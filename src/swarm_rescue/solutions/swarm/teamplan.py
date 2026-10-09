"""Planification d'équipe : une allocation globale des tâches, calculée par CHAQUE drone sur SA vue du réseau.

Tous les drones exécutent le même algorithme sur des données presque identiques (gossip) : ils obtiennent la
même allocation et chacun applique sa ligne. Les écarts restants sont absorbés par l'hystérésis (un drone garde
sa tâche annoncée sauf gain net) et par la replanification périodique.

Forme visée du réseau = un arbre enraciné au centre de secours (la "base") :
  - base      : un drone posté près du centre (nœud racine, sauveteur de réserve) ;
  - binômes   : aux extrémités, un MENEUR explore une frontière, un SUIVEUR le suit à distance radio
                (plus tard : le suiveur détecte une kill zone quand il perd le meneur) ;
  - relais    : au milieu, placés le long des chemins base -> frontières pour que la radio passe de proche en proche
                (un tronc commun n'est payé qu'une fois) ;
  - blessés   : affectés en priorité au drone le plus proche (souvent l'extrémité qui l'a vu).
Les frontières sont prises dans l'ordre de distance à la base tant qu'on a assez de drones pour les relier.

Affectation : méthode hongroise (scipy) sur les coûts de chemin réels (Dijkstra depuis chaque tâche).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Params
from .pathing import PlanGrid
from .schema import AgentState, Mission, MissionKind, ObjKind, ObjState, Pose
from .world import WorldModel

BIG = 1e7

# type de créneau -> type de mission annoncée
_SLOT_MISSION = {"wounded": MissionKind.GOTO_WOUNDED, "leader": MissionKind.EXPLORE, "follower": MissionKind.FOLLOW,
                 "relay": MissionKind.RELAY, "base": MissionKind.RELAY}


@dataclass
class Slot:
    kind: str                 # wounded | leader | follower | relay | base
    xy: np.ndarray
    cell: tuple
    pair: int = -1            # leader/follower : index du créneau partenaire


@dataclass
class TeamPlan:
    t: int
    root: Optional[np.ndarray]
    slots: list = field(default_factory=list)
    missions: dict = field(default_factory=dict)      # origin -> Mission
    roles: dict = field(default_factory=dict)         # origin -> kind de créneau ("" si sans tâche)


class TeamPlanner:
    def __init__(self, world: WorldModel, params: Params, planner):
        self.w, self.p = world, params
        self.mp = planner                 # MissionPlanner : base, blessés libres, liste noire
        self.last: Optional[TeamPlan] = None

    # ------------------------------------------------------------------ données d'entrée
    def _agents(self, t: int, pose: Pose, current: Mission) -> list[tuple[int, np.ndarray, Mission]]:
        p, w = self.p, self.w
        out = [(w.me, pose.pos, current)]
        for a in w.others():
            if t - a.t > p.team_fresh or a.health <= 0 or a.mission.kind in (MissionKind.CARRY, MissionKind.RECEIVE):
                continue
            out.append((a.origin, np.array(a.pos), a.mission))
        out.sort(key=lambda x: x[0])
        return out

    def _wounded(self, t: int, grid: PlanGrid) -> list[np.ndarray]:
        w = self.w
        carriers = [a for a in w.others() if t - a.t < 400 and a.mission.kind == MissionKind.CARRY]
        out = []
        for o in w.objects_view(ObjKind.WOUNDED):
            if o.state == ObjState.RESCUED or (o.state == ObjState.CARRIED and carriers):
                continue
            pos = np.array(o.pos)
            if self.mp._banned(pos, t) or not math.isfinite(grid.cost_to(pos)):
                continue
            out.append(pos)
        return out

    # ------------------------------------------------------------------ plan
    def plan(self, t: int, pose: Pose, grid: PlanGrid, current: Mission) -> Optional[TeamPlan]:
        p, m = self.p, grid.m
        agents = self._agents(t, pose, current)
        n = len(agents)
        root = self.mp.rescue_point()
        if root is None:
            root = np.mean([a[1] for a in agents], axis=0)
        root_cell = grid.nearest_reachable(root, radius=15)
        if root_cell is None:
            root_cell = grid.start
        root_xy = m.cell_to_world(np.array([root_cell[0]]), np.array([root_cell[1]]))[0]
        dist_root, pred_root = grid.tree_from(root_cell)

        slots: list[Slot] = []
        for pos in self._wounded(t, grid)[:n]:
            cell = grid.nearest_reachable(pos)            # le blessé lui-même est un obstacle pour le lidar
            if cell is not None:
                slots.append(Slot("wounded", pos, cell))
        remaining = n - len(slots)
        n_base = 0
        if remaining >= p.team_min_for_base:
            n_base = 1                                  # posé sur l'anneau avec la réserve (plus bas)
            remaining -= 1

        # ---- frontières : dans l'ordre de distance à la base, tant qu'on peut les relier
        fr = []
        for f in grid.frontiers(p.frontier_min_cells):
            c = m.cell_to_world(np.array([f.cell[0]]), np.array([f.cell[1]]))[0]
            d = float(dist_root[f.cell])
            if math.isfinite(d) and not self.mp._banned(c, t):
                fr.append((d - p.frontier_size_gain * f.size, f, c))
        fr.sort(key=lambda x: x[0])
        chosen, relays = [], []
        for _, f, c in fr:
            if remaining < 2 or len(chosen) >= p.team_max_frontiers:
                break
            if any(np.linalg.norm(c - q) < p.frontier_spacing for q in chosen):
                continue
            new = self._relays_for(grid, pred_root, root_cell, root_xy, f.cell, relays)
            if new is None or 2 + len(new) > remaining:
                continue
            chosen.append(c)
            relays += new
            remaining -= 2 + len(new)
            i = len(slots)
            slots.append(Slot("leader", c, f.cell, pair=i + 1))
            slots.append(Slot("follower", c, f.cell, pair=i))
        # ---- excursions : drones restants sur les frontières qu'on ne peut pas relier (binôme, ou seul s'il n'en
        # reste qu'un). Hors réseau un moment, ils rapportent la carte en revenant à portée (gossip).
        for _, f, c in fr:
            if remaining < 1 or len(chosen) >= p.team_max_frontiers + p.team_max_excursions:
                break
            if any(np.linalg.norm(c - q) < p.frontier_spacing for q in chosen):
                continue
            chosen.append(c)
            i = len(slots)
            if remaining >= 2:
                slots.append(Slot("leader", c, f.cell, pair=i + 1))
                slots.append(Slot("follower", c, f.cell, pair=i))
                remaining -= 2
            else:
                slots.append(Slot("leader", c, f.cell))
                remaining -= 1
        for r in relays:
            slots.append(Slot("relay", r, grid.cell(r)))
        # base + drones en trop : en anneau autour du centre de secours, pas sur son point d'accès (sinon ils
        # bloquent les porteurs) ni empilés. Positions candidates sur l'anneau, on garde les plus accessibles.
        n_ring = n_base + max(remaining, 0)
        if n_ring:
            ring = []
            for k in range(12):
                a = 2 * math.pi * k / 12
                xy = root_xy + p.standby_ring * np.array((math.cos(a), math.sin(a)))
                cell = grid.nearest_reachable(xy, radius=4)
                if cell is not None and math.isfinite(dist_root[cell]):
                    ring.append((float(dist_root[cell]), cell))
            ring.sort()
            picked = []
            for _, cell in ring:
                xy = m.cell_to_world(np.array([cell[0]]), np.array([cell[1]]))[0]
                if all(np.linalg.norm(xy - q) > 45.0 for q in picked):
                    picked.append(xy)
                    slots.append(Slot("base", xy, cell))
                if len(picked) >= n_ring:
                    break

        plan = TeamPlan(t, root_xy, slots)
        if not slots:
            self.last = plan
            return plan

        # ---- coûts : Dijkstra depuis chaque cellule de créneau (une seule fois par cellule distincte)
        cells = sorted({s.cell for s in slots})
        dmaps = dict(zip(cells, grid.dists_from(cells)))
        C = np.full((n, len(slots)), BIG)
        for i, (origin, pos, mission) in enumerate(agents):
            acell = grid.cell(pos)
            for j, s in enumerate(slots):
                c = PlanGrid.lookup(dmaps[s.cell], acell)
                if not math.isfinite(c):
                    continue
                if s.kind == "wounded":
                    c -= p.wounded_bonus
                elif s.kind == "follower":
                    c += 1.0                                   # à coût égal, le plus proche mène
                if mission.kind == _SLOT_MISSION[s.kind] and mission.target is not None \
                        and np.linalg.norm(np.array(mission.target) - s.xy) < p.sticky_radius:
                    c = c * p.sticky_gain - p.sticky_bonus      # hystérésis : on garde sa tâche annoncée
                C[i, j] = c
        rows, cols = linear_sum_assignment(C)
        by_slot = {}
        for i, j in zip(rows, cols):
            if C[i, j] < BIG / 2:
                by_slot[j] = agents[i][0]

        for j, origin in by_slot.items():
            s = slots[j]
            kind = _SLOT_MISSION[s.kind]
            peer = -1
            if s.kind == "follower":
                peer = by_slot.get(s.pair, -1)
                if peer < 0:
                    kind = MissionKind.EXPLORE                 # meneur absent : on explore seul
            plan.missions[origin] = Mission(kind, (float(s.xy[0]), float(s.xy[1])), t, peer)
            plan.roles[origin] = s.kind
        self.last = plan
        return plan

    # ------------------------------------------------------------------ relais
    def _relays_for(self, grid: PlanGrid, pred, root_cell, root_xy, cell, existing) -> Optional[list]:
        """Points de relais à ajouter le long du chemin base -> `cell` pour que deux maillons consécutifs
        restent à moins de link_range (distance euclidienne : la radio traverse les murs). Réutilise un relais
        déjà prévu s'il est proche du point voulu (tronc commun)."""
        L = self.p.link_range
        path = grid.path_cells(cell, pred, root_cell)
        if path is None:
            return None
        a = np.array(path)
        pts = grid.m.cell_to_world(a[:, 0], a[:, 1])
        new, last, prev = [], root_xy, root_xy
        for q in pts:
            if np.linalg.norm(q - last) > L:
                reuse = [r for r in list(existing) + new if np.linalg.norm(r - prev) < 0.4 * L]
                if reuse:
                    last = min(reuse, key=lambda r: np.linalg.norm(r - prev))
                else:
                    new.append(prev.copy())
                    last = prev
            prev = q
        return new
