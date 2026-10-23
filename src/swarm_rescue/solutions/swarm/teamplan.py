"""Planification d'équipe : une allocation des tâches PAR RÉSEAU CONNEXE, calculée par chaque drone du réseau.

Tous les drones d'un même réseau radio exécutent le même algorithme sur les mêmes données (gossip, positions à
quelques pas près dans le réseau) : ils obtiennent la même allocation et chacun applique sa ligne. Pour cela, rien
dans le calcul ne dépend du drone qui calcule :
  - grille d'équipe construite depuis la BASE, sans surcoût autour des drones (chaque drone s'excluait lui-même) ;
  - pas de liste noire locale (un drone qui a échoué sur une cible y renonce lui-même, cf. brain) ;
  - carte topologique recalculée au même pas pour tous (t // topo_period) ;
  - seuls les drones de MON réseau connexe sont affectés (positions fraîches, extrapolées). Les drones des autres
    réseaux ne sont connus qu'approximativement : leur cible / position « réserve » les frontières voisines.

Ordre de décision (chaque étape prend ses drones dans ce qui reste) :
  1. dépose : un explorateur dont la pièce est finie et contient un blessé le dépose devant la porte (côté parent),
     un autre drone le ramènera à la base ;
  2. meneurs : une voie d'exploration par frontière, prise par le drone le PLUS PROCHE (glouton sur le coût de
     chemin), en profondeur : bonus pour finir la pièce où l'on est et garder sa voie ; environ la moitié des drones ;
  3. relais : chaîne base -> meneur tous les relay_spacing px, jamais au-delà du meneur, décalée hors des passages
     et des routes des explorateurs (la radio traverse les murs) ; une voie qu'on ne peut pas relier explore seule ;
  4. suiveurs : le drone libre le plus proche de chaque meneur ;
  5. blessés, relais, base : méthode hongroise sur les drones restants (les blessés passent avant les relais).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Params
from .network import build_network
from .pathing import PlanGrid
from .schema import Mission, MissionKind, ObjKind, ObjState, Pose
from .topo import CORRIDOR, TopoMap
from .world import WorldModel

BIG = 1e7
_BUSY = (MissionKind.CARRY, MissionKind.RECEIVE, MissionKind.DROP)

# type de créneau -> type de mission annoncée
_SLOT_MISSION = {"wounded": MissionKind.GOTO_WOUNDED, "leader": MissionKind.EXPLORE, "follower": MissionKind.FOLLOW,
                 "relay": MissionKind.RELAY, "base": MissionKind.RELAY, "drop": MissionKind.DROP}


@dataclass
class Slot:
    kind: str                 # wounded | leader | follower | relay | base | drop
    xy: np.ndarray
    cell: tuple
    pair: int = -1            # follower : index du créneau meneur
    dest: Optional[np.ndarray] = None   # drop : point de dépose


@dataclass
class TeamPlan:
    t: int
    root: Optional[np.ndarray]
    slots: list = field(default_factory=list)
    missions: dict = field(default_factory=dict)      # origin -> Mission
    roles: dict = field(default_factory=dict)         # origin -> kind de créneau
    topo: Optional[TopoMap] = None                    # carte topologique utilisée (pour le debug)
    component: tuple = ()                             # drones de mon réseau connexe
    stats: dict = field(default_factory=dict)         # debug / mesures

    def taken_by_others(self, me: int) -> list:
        """Cibles (blessés, dépose, frontières) que ce plan donne à d'autres drones."""
        return [np.array(m.target) for o, m in self.missions.items() if o != me and m.target is not None
                and m.kind in (MissionKind.GOTO_WOUNDED, MissionKind.DROP, MissionKind.EXPLORE)]


class TeamPlanner:
    def __init__(self, world: WorldModel, params: Params, planner):
        self.w, self.p = world, params
        self.mp = planner                 # MissionPlanner : base, blessés
        self.last: Optional[TeamPlan] = None
        self._topo: Optional[TopoMap] = None
        self._topo_epoch = None
        self._axis = None
        self._D = None
        self.grid: Optional[PlanGrid] = None

    # ------------------------------------------------------------------ réseau
    def component(self, t: int, pose: Pose):
        """(drones de mon réseau connexe [(origin, position extrapolée, mission)], états des autres drones connus).
        Liens : entendus (AgentState.heard) ou prévus (distance < comp_link entre positions extrapolées)."""
        p, w = self.p, self.w
        net = build_network(w, t, fresh=p.comp_fresh, link_range=p.comp_link)
        net.nodes[w.me] = (float(pose.x), float(pose.y))
        for o, a in w.agents.items():                           # positions extrapolées (quelques pas d'âge)
            if o != w.me and o in net.nodes:
                age = min(t - a.t, 10)
                net.nodes[o] = (a.pos[0] + a.vel[0] * age, a.pos[1] + a.vel[1] * age)
        for o in list(net.nodes):
            if o != w.me and math.hypot(net.nodes[o][0] - pose.x, net.nodes[o][1] - pose.y) < p.comp_link:
                net.predicted.add((min(o, w.me), max(o, w.me)))
        comp = next((c for c in net.components() if w.me in c), {w.me})
        inside, outside = [], []
        for o in sorted(comp):
            if o == w.me:
                continue
            a = w.agents[o]
            if a.health > 0:
                inside.append((o, np.array(net.nodes[o], float), a.mission))
        for a in w.others():
            if a.origin not in comp and t - a.t < 600 and a.health > 0:
                outside.append(a)
        return inside, outside, tuple(sorted(comp))

    # ------------------------------------------------------------------ plan
    def plan(self, t: int, pose: Pose, current: Mission) -> Optional[TeamPlan]:
        p, w = self.p, self.w
        inside, outside, comp = self.component(t, pose)
        agents = [(w.me, pose.pos.copy(), current)] + [x for x in inside if x[2].kind not in _BUSY]
        agents.sort(key=lambda x: x[0])
        n = len(agents)
        root = self.mp.rescue_point()
        if root is None:
            root = np.mean([a[1] for a in agents], axis=0)
        # ---- grille d'équipe : depuis la ZONE DE RETOUR (espace libre connu où tous les drones ont démarré, collé au
        # centre de secours), sans rien qui dépende du drone qui calcule. Pas depuis le centre de secours lui-même :
        # son intérieur, inconnu, forme une poche fermée dont le Dijkstra ne sort pas (plus aucune frontière visible)
        home = self.mp.home()
        G = PlanGrid(w, p, home if home is not None else root, extra_cost=None, free_start=False)
        G.solve()
        self.grid = G
        m = G.m
        root_cell = G.start
        root_xy = m.cell_to_world(np.array([root_cell[0]]), np.array([root_cell[1]]))[0]
        dist_root, pred_root = G._dist, G._pred
        frontiers = G.frontiers(p.frontier_min_cells)
        epoch = t // p.topo_period                           # même instant de recalcul pour tous les drones
        if self._topo is None or epoch != self._topo_epoch:
            self._topo = TopoMap(G, root_xy, r_room=p.topo_room_core, prev=self._topo)
            self._topo_epoch = epoch
            self._axis = self._topo.axis_mask()
        topo = self._topo
        topo.attach_frontiers(frontiers)
        self._D = topo.D
        plan = TeamPlan(t, root_xy, topo=topo, component=comp)

        # distances depuis chaque drone (un Dijkstra multi-sources)
        acells = []
        for _, pos, _ in agents:
            c = G.nearest_reachable(pos, radius=4)
            acells.append(c if c is not None else G.cell(pos))
        dm = G.dists_from(acells)
        pool = list(range(n))
        slots: list[Slot] = []
        assign: dict[int, int] = {}                          # index agent -> index créneau

        def take(i, slot):
            slots.append(slot)
            assign[i] = len(slots) - 1
            pool.remove(i)

        # ---- 1. dépose à la porte (explorateur dans une pièce finie qui contient un blessé)
        wounded = self._wounded(t, G, outside)
        dropped = set()
        for i, (o, pos, mis) in enumerate(agents):
            if mis.kind != MissionKind.EXPLORE or n < p.small_net:
                continue                                 # petit réseau : personne d'autre ne viendrait le prendre
            rid = topo.region_at(pos, search=3)
            reg = topo.regions.get(rid)
            if reg is None or reg.kind == CORRIDOR or reg.parent < 0 or reg.link < 0:
                continue
            # Une frontière peut encore exister dans la pièce alors que le meneur
            # vise déjà une autre région. Dans ce cas le meneur doit prendre le
            # blessé trouvé avant de quitter la pièce, plutôt que de laisser un
            # suiveur le récupérer par le Hungarian.
            if reg.open and (mis.target is None or topo.region_at(mis.target, search=3) == rid):
                continue
            door = topo.links[reg.link].pos
            inroom = [(k, wp) for k, wp in enumerate(wounded) if k not in dropped
                      and topo.region_at(wp, search=2) == rid and np.linalg.norm(wp - door) > self._at_door()]
            if not inroom:
                continue
            k, wp = min(inroom, key=lambda x: float(PlanGrid.lookup(dm[i], G.cell(x[1]), radius=3)))
            dest = self._drop_point(G, topo, reg, door)
            if dest is None:
                continue
            dropped.add(k)
            take(i, Slot("drop", wp, G.cell(wp), dest=dest))

        # ---- 2. meneurs : voies d'exploration, le plus proche d'abord, en profondeur
        fr = self._frontier_points(G, frontiers, dist_root, topo, outside)
        free_wounded = [wp for k, wp in enumerate(wounded) if k not in dropped]
        n_base = 1 if len(pool) >= 6 else 0
        # Petit réseau (drone isolé, binôme : souvent coupé du reste après une perte radio) : réserver un drone pour un
        # blessé lui retirait sa voie d'exploration (k_lanes = 0 pour un drone seul) et l'envoyait sur un blessé connu
        # par gossip, parfois à l'autre bout de la carte et déjà pris en charge par le réseau principal. On ne réserve
        # que pour un blessé proche ; les autres restent aux drones sans voie (étape 5).
        reserve_cand = free_wounded
        if len(pool) < p.small_net and free_wounded:
            reserve_cand = [wp for wp in free_wounded
                            if min(PlanGrid.lookup(dm[i], G.cell(wp), radius=2) for i in pool) <= p.small_net_wounded_cost]
        reserve_w = min(len(reserve_cand), max(len(pool) // 3, 1)) if reserve_cand else 0
        k_lanes = min(p.max_lanes, len(fr), max(1, math.ceil(p.lane_share * len(pool))),
                      max(len(pool) - n_base - reserve_w, 0))
        # Hystérésis : les voies déjà actives (meneurs annoncés) restent ouvertes tant qu'il y a des frontières. Sinon un
        # blessé qui apparaît (réserve) fermait une voie et rebattait tous les rôles (meneur -> relais -> suiveur ...) ;
        # on garde au moins un drone libre pour les blessés.
        n_cur = sum(1 for i in pool if agents[i][2].kind == MissionKind.EXPLORE)
        keep = min(n_cur, max(len(pool) - n_base - min(reserve_w, 1), 0))
        k_lanes = min(p.max_lanes, len(fr), max(k_lanes, keep))
        leaders = []                                         # (index agent, index frontière)
        if k_lanes > 0:
            cost = np.full((n, len(fr)), np.inf)
            for i in pool:
                o, pos, mis = agents[i]
                my_reg = topo.region_at(pos, search=3)
                for j, f in enumerate(fr):
                    c = PlanGrid.lookup(dm[i], f["cell"], radius=1)
                    if not math.isfinite(c):
                        continue
                    c += f["penalty"] - p.frontier_size_gain * f["size"]
                    if f["region"] >= 0 and f["region"] == my_reg:
                        c -= p.lane_region_bonus             # finir la pièce où l'on est
                    if mis.kind == MissionKind.EXPLORE and mis.target is not None \
                            and math.hypot(mis.target[0] - f["xy"][0], mis.target[1] - f["xy"][1]) < p.lane_sticky_radius:
                        c -= p.lane_sticky_bonus             # garder sa voie
                    cost[i, j] = c
            chosen_xy = []
            while len(leaders) < k_lanes:
                ok = np.isfinite(cost)
                if not ok.any():
                    break
                i, j = np.unravel_index(int(np.argmin(np.where(ok, cost, np.inf))), cost.shape)
                leaders.append((int(i), int(j)))
                chosen_xy.append(fr[j]["xy"])
                cost[i, :] = np.inf
                for j2, f2 in enumerate(fr):                 # pas deux voies sur la même zone
                    if np.linalg.norm(f2["xy"] - fr[j]["xy"]) < p.frontier_spacing:
                        cost[:, j2] = np.inf
            plan.stats["leader_ranks"] = []
            for i, j in leaders:                              # rang du meneur parmi les drones libres (mesure)
                col = [PlanGrid.lookup(dm[a], fr[j]["cell"], radius=1) for a in pool]
                plan.stats["leader_ranks"].append(int(sum(1 for c in col
                                                          if c < PlanGrid.lookup(dm[i], fr[j]["cell"], radius=1) - 1e-6)))
            for i, j in leaders:
                take(i, Slot("leader", fr[j]["xy"], fr[j]["cell"]))

        # ---- 3. relais : chaîne base -> meneur, derrière lui, hors des passages
        budget = max(len(pool) - n_base - reserve_w, 0)
        lane_paths = []
        for i, j in leaders:
            path = G.path_cells(fr[j]["cell"], pred_root, root_cell)
            lane_paths.append(path)
        on_path = np.zeros((G.W, G.H), np.uint8)
        for path in lane_paths:
            if path:
                a = np.array(path)
                on_path[a[:, 0], a[:, 1]] = 1
        on_path = cv2.dilate(on_path, np.ones((5, 5), np.uint8)) > 0
        relays: list = []
        for (i, j), path in zip(leaders, lane_paths):
            if path is None:
                continue
            lead_d = float(PlanGrid.lookup(dist_root, acells[i], radius=2))
            new = self._relays_for(G, path, root_xy, relays, lead_d, dist_root, on_path)
            if new is not None and len(new) <= budget - len(relays):
                relays += new
        # ---- 4. suiveurs : le drone libre le plus proche de chaque meneur (s'il en reste au-delà des relais)
        n_follow = max(budget - len(relays), 0)
        for i, j in leaders:
            if n_follow <= 0 or not pool:
                break
            li = agents[i][0]
            lc = acells[i]

            def fcost(a):
                c = PlanGrid.lookup(dm[a], lc, radius=2)
                mis = agents[a][2]
                if mis.kind == MissionKind.FOLLOW and mis.peer == li:
                    c -= p.lane_sticky_bonus                 # garder son binôme
                return c
            a = min(pool, key=fcost)
            if not math.isfinite(fcost(a)):
                continue
            take(a, Slot("follower", fr[j]["xy"], fr[j]["cell"], pair=assign[i]))
            n_follow -= 1
        # ---- drones en trop (relais et suiveurs servis) : ils explorent au lieu d'attendre à la base
        extra = len(pool) - reserve_w - n_base - len(relays)
        if extra > 0 and leaders:
            cost[[i for i in range(n) if i not in pool], :] = np.inf
            while extra > 0:
                ok = np.isfinite(cost)
                if not ok.any():
                    break
                i, j = np.unravel_index(int(np.argmin(np.where(ok, cost, np.inf))), cost.shape)
                leaders.append((int(i), int(j)))
                take(int(i), Slot("leader", fr[j]["xy"], fr[j]["cell"]))
                cost[i, :] = np.inf
                for j2, f2 in enumerate(fr):
                    if np.linalg.norm(f2["xy"] - fr[j]["xy"]) < p.frontier_spacing:
                        cost[:, j2] = np.inf
                extra -= 1
        # ---- 5. blessés, relais, base : hongrois sur les drones restants
        rest_slots = [Slot("wounded", wp, c) for wp in free_wounded if (c := G.nearest_reachable(wp)) is not None]
        rest_slots += [Slot("relay", r, G.cell(r)) for r in relays]
        n_base_slots = max(len(pool) - len(rest_slots), 0) if pool else 0
        rest_slots += self._standby(G, root_xy, dist_root, n_base_slots)
        if pool and rest_slots:
            C = np.full((len(pool), len(rest_slots)), BIG)
            for r_, i in enumerate(pool):
                o, pos, mis = agents[i]
                for c_, s in enumerate(rest_slots):
                    c = PlanGrid.lookup(dm[i], s.cell, radius=1)
                    if not math.isfinite(c):
                        continue
                    if s.kind == "wounded":
                        if mis.kind == MissionKind.EXPLORE and mis.dest is not None                                 and math.hypot(mis.dest[0] - s.xy[0], mis.dest[1] - s.xy[1]) < 60.0:
                            continue                         # il vient de le déposer devant la porte : un autre ira
                        c -= p.wounded_bonus
                        if mis.kind == MissionKind.EXPLORE:
                            c += p.explore_switch_penalty   # un explorateur continue d'explorer, un autre ira
                    if mis.kind == _SLOT_MISSION[s.kind] and mis.target is not None \
                            and np.linalg.norm(np.array(mis.target) - s.xy) < p.sticky_radius:
                        c = c * p.sticky_gain - p.sticky_bonus      # hystérésis : on garde sa tâche annoncée
                    C[r_, c_] = c
            rows, cols = linear_sum_assignment(C)
            for r_, c_ in sorted(zip(rows, cols)):
                if C[r_, c_] < BIG / 2:
                    slots.append(rest_slots[c_])
                    assign[pool[r_]] = len(slots) - 1
        # aucun drone du réseau sans ligne (sinon il repartait en choix individuel, souvent sur le premier blessé) :
        # faute de poste, il attend là où il est
        for i in pool:
            if i not in assign:
                slots.append(Slot("base", agents[i][1].copy(), acells[i]))
                assign[i] = len(slots) - 1
        # ---- missions
        for i, sidx in assign.items():
            s = slots[sidx]
            kind = _SLOT_MISSION[s.kind]
            peer = -1
            if s.kind == "follower":
                peer = agents[[a for a, b in assign.items() if b == s.pair][0]][0]
            dest = (float(s.dest[0]), float(s.dest[1])) if s.dest is not None else None
            plan.missions[agents[i][0]] = Mission(kind, (float(s.xy[0]), float(s.xy[1])), t, peer, dest)
            plan.roles[agents[i][0]] = s.kind
        plan.slots = slots
        plan.stats.update({"n": n, "lanes": len(leaders), "relays": len(relays), "outside": len(outside)})
        self.last = plan
        return plan

    def drop_task(self, t: int, pos, leaving_to=None) -> Optional[Mission]:
        """Dépose décidée par le meneur lui-même au moment de QUITTER une pièce finie (son prochain bord ouvert est
        ailleurs) : s'il y reste un blessé libre loin de la porte, il le prend et le pose devant la porte, côté pièce
        parente ; un autre drone le ramènera à la base sans avoir à fouiller la pièce."""
        topo, G = self._topo, self.grid
        if topo is None or G is None:
            return None
        if self.last is not None and len(self.last.component) < self.p.small_net:
            return None                                       # petit réseau : personne d'autre ne viendrait le prendre
        rid = topo.region_at(pos, search=3)
        reg = topo.regions.get(rid)
        if reg is None or reg.kind == CORRIDOR or reg.parent < 0 or reg.link < 0:
            return None
        if leaving_to is not None and topo.region_at(leaving_to, search=3) == rid:
            return None                                       # on reste dans la pièce : pas encore
        door = topo.links[reg.link].pos
        inroom = [wp for wp in self._wounded(t, G) if topo.region_at(wp, search=2) == rid
                  and np.linalg.norm(wp - door) > self._at_door()]
        if not inroom:
            return None
        wp = min(inroom, key=lambda q: float(np.linalg.norm(q - np.asarray(pos))))
        dest = self._drop_point(G, topo, reg, door)
        if dest is None:
            return None
        return Mission(MissionKind.DROP, (float(wp[0]), float(wp[1])), t, -1, (float(dest[0]), float(dest[1])))

    # ------------------------------------------------------------------ entrées
    def _wounded(self, t: int, G: PlanGrid, outside=()) -> list[np.ndarray]:
        w = self.w
        carriers = [a for a in w.others() if t - a.t < 400 and a.mission.kind in (MissionKind.CARRY, MissionKind.DROP)]
        # Blessé visé par un drone d'un AUTRE réseau (dernier état connu) : il s'en occupe, même s'il ne m'entend plus
        # (zone sans radio). Sinon chaque changement de composition du réseau me le faisait réattribuer, et un drone
        # traversait la carte vers un blessé déjà porté. Comme claim_penalty pour les frontières.
        claims = [np.array(a.mission.target) for a in outside if t - a.t < self.p.wounded_claim_time
                  and a.mission.kind == MissionKind.GOTO_WOUNDED and a.mission.target is not None]
        out = []
        for o in w.objects_view(ObjKind.WOUNDED):
            if o.state in (ObjState.RESCUED, ObjState.TAKEN) or (o.state == ObjState.CARRIED and carriers):
                continue
            pos = np.array(o.pos)
            if any(a.mission.kind == MissionKind.DROP and a.mission.target is not None
                   and np.linalg.norm(np.array(a.mission.target) - pos) < 40.0 for a in carriers):
                continue                                     # un explorateur est en train de le déposer
            if any(np.linalg.norm(q - pos) < 40.0 for q in claims):
                continue
            if math.isfinite(G.cost_to(pos)):
                out.append(pos)
        return out

    def _frontier_points(self, G: PlanGrid, frontiers, dist_root, topo: TopoMap, outside) -> list[dict]:
        """Point de chaque frontière : sa cellule la plus proche de la BASE (le même pour tous les drones), sa région,
        et le surcoût si un drone d'un autre réseau l'explore déjà (sa cible ou sa position)."""
        p, m = self.p, G.m
        claims = []
        for a in outside:
            if a.mission.kind in (MissionKind.EXPLORE, MissionKind.FOLLOW) and a.mission.target is not None:
                claims.append(np.array(a.mission.target))
            claims.append(np.array(a.pos))
        out = []
        for f in frontiers:
            cell = f.cell
            if f.cells is not None:
                dr = dist_root[f.cells[0], f.cells[1]]
                if np.isfinite(dr).any():
                    j = int(np.argmin(dr))
                    cell = (int(f.cells[0][j]), int(f.cells[1][j]))
            if not math.isfinite(dist_root[cell]):
                continue
            xy = m.cell_to_world(np.array([cell[0]]), np.array([cell[1]]))[0]
            pen = p.claim_penalty if any(np.linalg.norm(xy - q) < p.claim_radius for q in claims) else 0.0
            out.append({"cell": cell, "xy": xy, "size": f.size, "region": topo.region_of_cell(cell), "penalty": pen})
        return out

    # ------------------------------------------------------------------ relais
    def _relays_for(self, G: PlanGrid, path, root_xy, existing, lead_d, dist_root, on_path) -> Optional[list]:
        """Maillons à ajouter le long du chemin base -> frontière pour que deux maillons consécutifs restent à moins
        de relay_link_max (distance euclidienne : la radio traverse les murs), seulement là où le meneur est déjà passé
        (jamais devant lui). Réutilise un maillon prévu s'il est proche (tronc commun)."""
        p = self.p
        L = p.relay_spacing
        a = np.array(path)
        pts = G.m.cell_to_world(a[:, 0], a[:, 1])
        new, last = [], root_xy
        for q, qc in zip(pts, a):
            if dist_root[qc[0], qc[1]] > lead_d - 0.5 * L:
                break                                         # au-delà : le meneur (et son suiveur) font le lien
            if np.linalg.norm(q - last) > L:
                reuse = [r for r in list(existing) + new if np.linalg.norm(r - q) < 0.4 * L
                         and np.linalg.norm(r - last) < p.relay_link_max]
                if reuse:
                    last = min(reuse, key=lambda r: np.linalg.norm(r - q))
                else:
                    last = self._snap_relay(G, tuple(qc), last, on_path)
                    new.append(last)
        return new

    def _snap_relay(self, G: PlanGrid, cell, prev_xy, on_path) -> np.ndarray:
        """Poste de relais : jusqu'à relay_snap_cells du chemin, hors des axes de passage et des routes des
        explorateurs, vers l'endroit le plus dégagé, à portée radio du maillon précédent."""
        p, m = self.p, G.m
        r = p.relay_snap_cells
        W, H = self._axis.shape
        x0, x1 = max(cell[0] - r, 0), min(cell[0] + r + 1, W)
        y0, y1 = max(cell[1] - r, 0), min(cell[1] + r + 1, H)
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        wxy = m.cell_to_world(gx, gy)
        in_range = np.hypot(wxy[..., 0] - prev_xy[0], wxy[..., 1] - prev_xy[1]) <= p.relay_link_max
        ok = (~self._axis[x0:x1, y0:y1]) & (~on_path[x0:x1, y0:y1]) & G.trav[x0:x1, y0:y1] \
            & np.isfinite(G._dist[x0:x1, y0:y1]) & (np.hypot(gx - cell[0], gy - cell[1]) <= r) & in_range
        if ok.any():
            score = np.where(ok, self._D[x0:x1, y0:y1] - 0.3 * np.hypot(gx - cell[0], gy - cell[1]), -np.inf)
            k = np.unravel_index(int(np.argmax(score)), score.shape)
            cell = (int(gx[k]), int(gy[k]))
        return m.cell_to_world(np.array([cell[0]]), np.array([cell[1]]))[0]

    def _standby(self, G: PlanGrid, root_xy, dist_root, n: int) -> list:
        """Postes d'attente DANS la zone de retour (tous les drones doivent y être à la fin pour que la mission
        s'arrête), espacés de 40 px, en laissant libre l'accès au centre de secours : pas à moins de 40 px du segment
        centre de la zone -> centre de secours ; de préférence loin du centre de secours. Complété par l'anneau."""
        if n <= 0:
            return []
        out = []
        box = self.mp.return_box()
        if box is not None:
            m = G.m
            lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
            ax_, ay_ = m.world_to_cell(lo + 12.0)
            bx_, by_ = m.world_to_cell(hi - 12.0)
            x0, x1, y0, y1 = int(ax_[0]), int(bx_[0]) + 1, int(ay_[0]), int(by_[0]) + 1
            if x1 > x0 and y1 > y0:
                gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
                ok = G.trav[x0:x1, y0:y1] & np.isfinite(dist_root[x0:x1, y0:y1])
                xy = m.cell_to_world(gx, gy)
                rc = self.mp.rescue_point()
                ac = 0.5 * (lo + hi)
                if rc is not None:
                    seg = rc - ac
                    L2 = max(float(seg @ seg), 1e-6)
                    u = np.clip(((xy - ac) @ seg) / L2, 0.0, 1.0)
                    dseg = np.linalg.norm(xy - (ac + u[..., None] * seg), axis=-1)
                    ok &= dseg > 40.0
                    far = np.linalg.norm(xy - rc, axis=-1)
                else:
                    far = np.zeros(gx.shape)
                idx = [(-round(float(far[k]), 1), int(gx[k]), int(gy[k])) for k in zip(*np.nonzero(ok))]
                idx.sort()
                picked = []
                for _, cx, cy in idx:
                    q = m.cell_to_world(np.array([cx]), np.array([cy]))[0]
                    if all(np.linalg.norm(q - r) >= 40.0 for r in picked):
                        picked.append(q)
                        out.append(Slot("base", q, (cx, cy)))
                        if len(out) >= n:
                            break
        if len(out) < n:
            out += self._ring(G, root_xy, dist_root, n - len(out))
        return out

    def _ring(self, G: PlanGrid, root_xy, dist_root, n_ring: int) -> list:
        """Postes d'attente en anneau autour du centre de secours, pas sur son accès ni sur un axe de passage."""
        if n_ring <= 0:
            return []
        p, m = self.p, G.m
        ring = []
        for k in range(12):
            a = 2 * math.pi * k / 12
            xy = root_xy + p.standby_ring * np.array((math.cos(a), math.sin(a)))
            cell = G.nearest_reachable(xy, radius=4)
            if cell is not None and math.isfinite(dist_root[cell]) and not self._axis[cell]:
                ring.append((float(dist_root[cell]), cell))
        ring.sort()
        picked, out = [], []
        for _, cell in ring:
            xy = m.cell_to_world(np.array([cell[0]]), np.array([cell[1]]))[0]
            if all(np.linalg.norm(xy - q) > 45.0 for q in picked):
                picked.append(xy)
                out.append(Slot("base", xy, cell))
            if len(out) >= n_ring:
                break
        return out

    def _at_door(self) -> float:
        """Distance à la porte en deçà de laquelle un blessé y est déjà déposé : décalage de dépose + blessé pendu au
        drone (25 px) + marge. À 50 px, un blessé tout juste déposé (~65 px) était « encore dans la pièce » : le
        déposeur le reprenait pour le déposer à nouveau."""
        return self.p.drop_offset + 25.0 + 30.0

    def _drop_point(self, G: PlanGrid, topo: TopoMap, reg, door) -> Optional[np.ndarray]:
        """Dépose : derrière la porte, côté pièce parente, à drop_offset px, sur une cellule libre hors axe de passage
        (le blessé ne doit pas boucher la porte)."""
        p, m = self.p, G.m
        u = np.asarray(door, float) - np.asarray(reg.interior, float)
        nu = float(np.linalg.norm(u))
        if nu < 1e-6:
            return None
        target = np.asarray(door, float) + p.drop_offset * u / nu
        cx, cy = G.cell(target)
        r = 6
        x0, x1 = max(cx - r, 0), min(cx + r + 1, G.W)
        y0, y1 = max(cy - r, 0), min(cy + r + 1, G.H)
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        ok = G.trav[x0:x1, y0:y1] & np.isfinite(G._dist[x0:x1, y0:y1]) & (~self._axis[x0:x1, y0:y1]) \
            & (topo.labels[x0:x1, y0:y1] != reg.rid)
        if not ok.any():
            ok = G.trav[x0:x1, y0:y1] & np.isfinite(G._dist[x0:x1, y0:y1]) & (topo.labels[x0:x1, y0:y1] != reg.rid)
        if not ok.any():
            return None
        d = np.where(ok, np.hypot(gx - cx, gy - cy), np.inf)
        k = np.unravel_index(int(np.argmin(d)), d.shape)
        return m.cell_to_world(np.array([gx[k]]), np.array([gy[k]]))[0]
