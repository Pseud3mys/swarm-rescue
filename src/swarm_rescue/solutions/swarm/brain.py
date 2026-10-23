"""Orchestrateur : SensorFrame -> Command, un pas à la fois. Aucune dépendance au simulateur.

Ordre d'un pas :  radio -> estimation (EKF + amers) -> perception (carte, objets) ->
                  mission + chemin (sur événement) -> asservissement -> publication de l'état aux voisins.
Chaque étape est dans son propre module ; ce fichier ne contient que la logique de séquencement
et les règles d'événements (quand replanifier, détection de blocage, saisie / dépose).
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import replace
from typing import Optional

import numpy as np

from .comms import Gossip, GossipConfig, extract_packets
from .config import Params
from .controller import PathFollower
from . import fusion
from .estimator import PoseEKF
from .gridmap import OCC_THR
from .landmarks import LandmarkTracker
from .missions import MissionPlanner
from .pathing import PlanGrid, extra_cost_from_agents
from .perception import ScanIntegrator, dynamic_ray_mask, observe_semantic
from .scanmatch import ScanMatcher
from .schema import Command, Mission, MissionKind, ObjKind, ObjState, Packet, Route, SensorFrame
from .teamplan import TeamPlanner
from .world import WorldModel
from .monitoring import MissionMonitor


class Brain:
    def __init__(self, me: int, size_area, params: Optional[Params] = None, t_limit: Optional[int] = None):
        self.p = p = params or Params()
        self.me = me
        self.t_limit = t_limit          # pas de fin de partie (misc_data) : retour à la zone de départ avant
        self._homing = False
        self._home_xy = None            # ma position de départ (dans la zone de retour)
        self.world = WorldModel(me, size_area, p.resolution, p.chunk)
        self.gossip = Gossip(self.world, GossipConfig())
        self.monitor = MissionMonitor(os.environ.get("SWARM_MONITOR"))
        self.ekf = PoseEKF(p)
        self.landmarks = LandmarkTracker(self.world, self.ekf, p, min_gap=30)
        self.scan = ScanIntegrator(self.world, p)
        self.matcher = ScanMatcher(self.world, p)
        self.planner = MissionPlanner(self.world, p)
        self.team = TeamPlanner(self.world, p, self.planner)
        self.follower = PathFollower(p)

        self.mission = Mission(MissionKind.IDLE)
        self.route: Optional[Route] = None
        self.theta_ref: Optional[float] = None
        self.last_cmd = Command()
        self._packet: Optional[Packet] = None
        self._last_plan = -10 ** 9
        self._plan_reason = ""         # drapeau d'événement en attente (consommé par _replan_reason)
        self.last_reason = "start"
        self._was_carrying = False
        self._grasp = 0
        self._grasp_since = 0
        self._stuck = 0
        self._sighting_t = -10 ** 9
        self._arrived_t: Optional[int] = None
        self._home_set = False
        self._carry_key = None
        self._pending_drop = None      # (clé de piste, échéance, instant de dépose)
        self._nudge = 0
        self._n_free_wounded = 0
        self._hist: deque = deque(maxlen=7)            # (t, position) tous les 10 pas : détection d'absence de progrès
        self._recover_until = -1
        self._recover_dir = None
        self._recover_theta = None      # porteur : cap visé pendant le dégagement (rotation bornée)
        self._n_recover = 0
        self._carry_since = 0
        self._handoff = None            # porteur : {"peer", "since", "ack"} passage de relais en cours
        self._handoff_ban: dict = {}    # receveur -> pas jusqu'auquel on ne le sollicite plus
        self._handoff_release = False   # le lâcher en cours est un passage de relais (pas une dépose)
        self._recv_since = 0
        self._wait_since = None         # début de l'attente en file (même sens, drone devant)
        # Pince = PivotJoint fixé dans le repère du drone : le blessé reste du côté où on l'a saisi.
        # Il devient invisible aux capteurs une fois saisi, mais cet angle (repère drone) reste valable.
        self._wounded_bearing = None    # relèvement (repère drone) du blessé visé pendant l'approche
        self._carry_bearing = math.pi   # relèvement du blessé porté (repère drone)
        self._rc_close_since = None
        self._no_wait_until = -1
        self._last_follow = -10 ** 9
        self._cleared: set = set()      # blessés dont les impacts lidar ont été effacés de ma couche
        self._drop_release = False      # le lâcher en cours est une dépose volontaire devant une porte
        self._take_full = False         # DROP pris par le meneur sur un blessé proche : après la saisie, sauvetage complet
        self._last_explore = -10 ** 9
        self._disc = None               # disque de rayon check_near (cf. _mark_near)
        self._lane = None               # dernière cible d'exploration (voie reprise après une dépose devant une porte)
        self._dropped = None            # (position, instant) de ma dernière dépose devant une porte, annoncée (mission.dest)
        self.mission_src = "start"
        self._no_follow: dict = {}      # meneur -> pas jusqu'auquel je ne le suis plus (un autre suiveur est mieux placé)
        self._wl_last: dict = {}        # blessé fixe -> dernier recalage de l'EKF dessus
        self._vanish: dict = {}         # blessé connu (clé de groupe) -> constats d'absence consécutifs
        self.timing = {"est": 0.0, "map": 0.0, "plan": 0.0, "ctl": 0.0, "n_plan": 0}

    # ------------------------------------------------------------------ interface
    def outgoing(self) -> Optional[Packet]:
        return self._packet

    def step(self, f: SensorFrame) -> Command:
        p, w, t = self.p, self.world, f.t
        T = time.perf_counter
        t0 = T()
        self.gossip.ingest(extract_packets(f.inbox), t)

        # ---- estimation
        self.ekf.predict(f.odom, f.gps is not None, f.compass is not None,
                         (self.last_cmd.forward, self.last_cmd.lateral), self.last_cmd.rotation,
                         accel=p.accel_per_cmd_carry if f.carrying else p.accel_per_cmd, carrying=f.carrying)
        self.ekf.update_gps(f.gps, f.compass)
        self.ekf.update_compass(f.compass)
        if not self.ekf.initialized:
            return Command()
        # amers : recalage dès qu'un coin connu est revu sans GPS ; avec GPS, entretien + recalage sur les coins convergés
        if f.lidar is not None and t % (2 if f.gps is None else 4) == 0:
            self.landmarks.process(t, f.lidar, f.angles)
        pose = self.ekf.pose()
        if self.theta_ref is None:
            self.theta_ref = pose.theta
        t1 = T()

        # ---- perception
        if not self._home_set and f.in_return_area:
            w.observe_object(ObjKind.RETURN_AREA, pose.pos, t)
            self._home_set = True
            self._home_xy = pose.pos.copy()
        if f.lidar is not None and t % 2 == 0:       # un scan sur deux suffit (on bouge de < 5 px par pas)
            mask = dynamic_ray_mask(f.lidar, f.angles, f.semantic, pose, self._bodies(t))
            # recalage sur la carte AVANT d'y graver ce scan. Le modèle de mouvement de l'EKF tient le cap à ~0,1°/pas :
            # se recaler sur SA carte (dont l'orientation garde le biais de boussole de son passé) le dégradait (mesuré).
            # Seuls cas utiles : la carte des AUTRES drones (indépendante, tous les sm_pose_period pas : coût), et, sans
            # GPS, ma carte quand le cap est vraiment incertain (en portant : odomètre seul, ~1°/pas)
            gps_mode = f.gps is not None
            use_own = not gps_mode and math.sqrt(max(float(pose.cov[2, 2]), 0.0)) > math.radians(p.sm_own_min_sth)
            if p.sm_enabled and (use_own or t % p.sm_pose_period == 0):
                fix = self.matcher.match(t, pose, f.lidar, f.angles, mask, compass_ok=f.compass is not None,
                                         want_pose=t % p.sm_pose_period == 0, use_own=use_own)
                if fix is not None and (self.ekf.update_pose(fix[1], fix[2]) if fix[0] == "pose"
                                        else self.ekf.update_heading(fix[1], fix[2])):
                    pose = self.ekf.pose()
            self.scan.integrate(pose, f.lidar, f.angles, mask)
        if t % 2 == 0:                                # amas suspects réglés par moi et mon réseau (cf. pathing)
            if pose.sigma < p.max_pose_sigma_for_mapping:
                self._mark_near(pose.pos)
            for a in w.others():
                if t - a.t <= 2 and a.pos_sigma < p.max_pose_sigma_for_mapping:
                    self._mark_near(a.pos)
        seen = observe_semantic(w, pose, f.semantic, t, p)
        self._wounded_landmarks(t, f, seen)
        if t % 2 == 1:
            self._check_vanished(t, pose, f, seen)
        if t % 20 == 0:
            self._clear_phantoms()
        t2 = T()

        # ---- mission / chemin
        if f.carrying != self._was_carrying:
            self._plan_reason = "carry-change"
            if f.carrying:                                         # prise : la piste du blessé passe à CARRIED (sur place)
                self._carry_bearing = self._wounded_bearing if self._wounded_bearing is not None else math.pi
                a = pose.theta + self._carry_bearing
                gp = pose.pos + 25.0 * np.array((math.cos(a), math.sin(a)))   # le blessé saisi est là
                tr = w.nearest_own(ObjKind.WOUNDED, gp, 80.0)
                if tr is None:
                    tr = w.observe_object(ObjKind.WOUNDED, gp, t, ObjState.SEEN)
                w.add_track(ObjKind.WOUNDED, tr.pos, t, ObjState.TAKEN, tr.cov)   # "pris ici" : efface sa position
                self._carry_key = tr.key
                self._carry_since = t
                w.set_state(tr.key, ObjState.CARRIED, t)
                self._clear_around_grasp(t, gp, seen)
                if self._take_full and self.mission.kind == MissionKind.DROP:
                    self.mission = Mission(MissionKind.IDLE)   # pris par le meneur : on le porte jusqu'au centre
                self._take_full = False
            elif self._handoff_release:                            # passé à un autre drone : sa piste prend le relais
                self._handoff_release = False
                self._carry_key = None
            elif self._drop_release and self._carry_key is not None:   # déposé exprès devant la porte : reste à prendre
                self._drop_release = False
                w.move_track(self._carry_key, pose.pos + 25.0 * np.array((math.cos(pose.theta + self._carry_bearing),
                                                                          math.sin(pose.theta + self._carry_bearing))), t)
                w.set_state(self._carry_key, ObjState.SEEN, t)
                xy = w.objects[self._carry_key].pos
                self._carry_key = None
                # Je reprends ma voie d'exploration, en annonçant ma dépose (dest de la mission, diffusée) : le plan
                # d'équipe, le même chez tous, ne me redonne pas ce blessé (teamplan, étape 5) ; un autre viendra.
                # Sans ça j'étais le plus proche et je le reprenais au pas suivant : la dépose ne servait à rien.
                self._dropped = ((float(xy[0]), float(xy[1])), t)
                self.mission = Mission(MissionKind.EXPLORE, self._lane, t, -1, self._dropped[0])
            elif self._carry_key is not None:                      # dépose : verdict après quelques pas (cf. _control)
                # Sauvé jusqu'à preuve du contraire : resté « porté » sans porteur, la piste était libre pour les planners
                # (teamplan._wounded, missions.candidates) et je repartais chercher le blessé que je venais de livrer
                w.set_state(self._carry_key, ObjState.RESCUED, t)
                self._pending_drop = (self._carry_key, t + self.p.drop_check_steps, t)
                self._carry_key = None
            self._grasp = 1 if f.carrying else 0
            self._was_carrying = f.carrying
        if f.carrying and self._carry_key is not None and t % p.carry_track_period == 0:
            a = pose.theta + self._carry_bearing                # la piste suit le blessé, donc le porteur
            w.move_track(self._carry_key, pose.pos + 25.0 * np.array((math.cos(a), math.sin(a))), t)
        self._handoff_logic(t, pose, f)
        if t % 3 == 0 and not f.carrying:
            self._resolve_conflicts(t, pose)
        reason = self._replan_reason(t, pose, f, seen)
        if reason:
            self.last_reason = reason
            self._replan(t, pose, f)
        elif self.mission.kind in (MissionKind.FOLLOW, MissionKind.RECEIVE) and t - self._last_follow >= p.follow_replan:
            self._refresh_follow(t, pose, f)
        elif self.mission.kind == MissionKind.EXPLORE and not f.carrying and t - self._last_explore >= p.explore_refresh:
            self._refresh_explore(t, pose, f)
        t3 = T()

        # ---- asservissement
        cmd = self._control(t, pose, f, seen)
        self.last_cmd = cmd
        t4 = T()

        # ---- publication
        w.publish_self(t, pose.pos, pose.vel, f.health, self.mission,
                       self.route.waypoints() if self.route is not None else (), self.gossip.heard, pose.sigma)
        self._packet = self.gossip.build_packet(t)
        self.monitor.observe(self, t)
        for k, a, b in (("est", t0, t1), ("map", t1, t2), ("plan", t2, t3), ("ctl", t3, t4)):
            self.timing[k] += b - a
        return cmd

    # ------------------------------------------------------------------ carte : objets mobiles
    def _bodies(self, t) -> list:
        """Corps mobiles connus à ne pas graver comme murs : autres drones (position annoncée extrapolée, porteurs avec
        leur blessé) et blessés connus. Couvre le lidar au-delà des 200 px du capteur sémantique."""
        out = []
        for a in self.world.others():
            age = t - a.t
            if age > 5 or a.health <= 0:
                continue
            xy = (a.pos[0] + a.vel[0] * age, a.pos[1] + a.vel[1] * age)
            rad = 20.0 + 2.0 * a.pos_sigma + 1.5 * age
            if a.mission.kind == MissionKind.CARRY:
                rad += 25.0                                     # le blessé pend derrière le porteur
            out.append((xy, rad))
        for o in self.world.objects_view(ObjKind.WOUNDED):
            if o.state == ObjState.SEEN:
                out.append((o.pos, 14.0 + 2.0 * o.sigma))
        return out

    def _clear_phantoms(self) -> None:
        """Les blessés vus par le lidar avant d'être reconnus (portée 300 px contre 200) ont laissé des impacts dans MA
        couche : on les efface une fois par blessé connu (je ne touche qu'à ma couche, propriétaire unique)."""
        m = self.world.map
        for o in self.world.objects_view(ObjKind.WOUNDED):
            if o.state == ObjState.CARRIED:
                continue
            k = (int(o.pos[0] // 16), int(o.pos[1] // 16))
            if k in self._cleared:
                continue
            self._cleared.add(k)
            r = 13.0 + 1.5 * o.sigma                            # pas plus : un mur peut toucher le blessé
            g = np.arange(-r, r + 1e-6, m.res / 2)
            gx, gy = np.meshgrid(g, g, indexing="ij")
            inside = gx ** 2 + gy ** 2 <= r * r
            ix, iy = m.world_to_cell(np.stack((o.pos[0] + gx[inside], o.pos[1] + gy[inside]), axis=1))
            m.clamp_max(self.me, ix, iy, 0)

    # ------------------------------------------------------------------ blessés fixes comme amers
    def _wounded_landmarks(self, t, f, seen) -> None:
        """Sans GPS : un blessé FIXE (vu longtemps sans bouger, cf. world.motion_class) est un amer comme un coin :
        recalage de l'EKF sur la distance et le relèvement mesurés. Position de référence : celle des AUTRES drones
        s'ils l'ont vu (indépendante), sinon la mienne, figée depuis qu'il est classé fixe (cf. observe_object)."""
        p, w = self.p, self.world
        if f.gps is not None or not seen["wounded_rb"]:
            return
        view = w.objects_view(ObjKind.WOUNDED)
        for key, rng, brg, n in seen["wounded_rb"]:
            if w.motion_class(key) != "fixed" or t - self._wl_last.get(key, -10 ** 9) < p.wounded_lm_gap:
                continue
            o = next((o for o in view if key in o.sources), None)
            if o is None or o.state != ObjState.SEEN or w.wounded_class(o) != "fixed":
                continue
            if o.others_pos is not None:
                ref, cov = o.others_pos, fusion.cov_unpack(o.others_cov)
            else:
                own = w._own.get(key)
                if own is None:
                    continue
                ref, cov = own[0], own[1]
            if self.ekf.update_landmark(ref, rng, brg, sigma_r=p.sem_sigma_r, sigma_b=p.sem_sigma_b / math.sqrt(n),
                                        cov_l=cov):
                self._wl_last[key] = t

    # ------------------------------------------------------------------ blessés imaginaires
    def _wounded_near(self, xy, radius: float) -> bool:
        """Un blessé libre connu est-il encore à moins de `radius` de xy ?"""
        xy = np.asarray(xy, float)
        return any(o.state == ObjState.SEEN and np.linalg.norm(np.array(o.pos) - xy) < radius
                   for o in self.world.objects_view(ObjKind.WOUNDED))

    def _forget_wounded_near(self, xy, radius: float, t: int, seen=None) -> None:
        """Retire les blessés libres connus à moins de `radius` de xy, sauf ceux que je vois en ce moment."""
        xy = np.asarray(xy, float)
        vis = [q for q, _, st in seen["wounded"] if st == ObjState.SEEN] if seen is not None else []
        for o in list(self.world.objects_view(ObjKind.WOUNDED)):
            if o.state != ObjState.SEEN or np.linalg.norm(np.array(o.pos) - xy) >= radius:
                continue
            if any(np.linalg.norm(q - np.array(o.pos)) < 40.0 for q in vis):
                continue
            self.world.mark_gone(o, t)

    def _clear_around_grasp(self, t, gp, seen) -> None:
        """Juste après une saisie : les autres blessés « connus » autour du blessé saisi et que je ne vois pas sont des
        doublons de celui-ci (blessé mobile suivi en plusieurs pistes, positions d'autres drones décalées)."""
        self._forget_wounded_near(gp, self.p.grasp_clear_radius, t, seen)

    def _check_vanished(self, t, pose, f, seen) -> None:
        """Vérification négative : un blessé connu, proche et en vue directe, qui n'est plus là. Tous les rayons lidar
        qui couvrent son disque (incertitudes comprises) vont plus loin que lui, et le capteur sémantique ne voit aucun
        blessé près de sa position : s'il y était, son corps arrêterait au moins un rayon (un drone ou un mur devant
        l'arrête aussi : pas de conclusion). Après quelques constats consécutifs, la piste est retirée (blessé mobile
        parti, blessé imaginaire, déjà pris par un autre)."""
        if f.lidar is None or f.angles is None:
            self._vanish = {}
            return
        p = self.p
        r = np.asarray(f.lidar, float)
        rel = np.asarray(f.angles, float) + pose.theta
        vis = [q for q, _, _ in seen["wounded"]]
        keep = {}
        for o in self.world.objects_view(ObjKind.WOUNDED):
            if o.state != ObjState.SEEN:
                continue
            op = np.array(o.pos)
            to = op - pose.pos
            d = float(np.linalg.norm(to))
            rad = p.wounded_radius + 2.0 * (pose.sigma + o.sigma)
            if d > p.vanish_check_dist or d < rad + 15.0:
                continue
            if any(np.linalg.norm(q - op) < 40.0 for q in vis):
                continue
            dev = (rel - math.atan2(to[1], to[0]) + np.pi) % (2 * np.pi) - np.pi
            rays = np.abs(dev) < math.asin(rad / d) + 0.02
            if rays.sum() < 2 or not (np.isfinite(r[rays]) & (r[rays] > d + rad)).all():
                continue
            k = o.sources[0]
            n = self._vanish.get(k, 0) + 1
            if n >= p.vanish_checks:
                self.world.mark_gone(o, t)
            else:
                keep[k] = n
        self._vanish = keep

    # ------------------------------------------------------------------ événements de replanification
    def _replan_reason(self, t, pose, f, seen) -> Optional[str]:
        p = self.p
        if self._plan_reason in ("carry-change", "stuck", "conflict"):
            r, self._plan_reason = self._plan_reason, ""
            return r
        if not self._homing and t % 10 == 0 and self._time_to_go_home(t, pose):
            self._homing = True
            return "homing"
        if self.route is None and t - self._last_plan >= 10:
            return "no-route"
        if self.mission.kind == MissionKind.FOLLOW and self.mission.peer >= 0 and t % 3 == 0:
            lead = self.world.agents.get(self.mission.peer)
            if lead is not None and t - lead.t < 10 and lead.mission.kind != MissionKind.EXPLORE:
                return "leader-changed"                  # le meneur prend un blessé : je reprends la voie
        if t - self._last_plan >= p.replan_period:
            return "period"
        if t % 5 == 0:                                   # un blessé apparaît / disparaît : l'équipe se réorganise
            n = sum(1 for o in self.world.objects_view(ObjKind.WOUNDED) if o.state == ObjState.SEEN)
            if n != self._n_free_wounded:
                self._n_free_wounded = n
                return "wounded-change"
        if self.route is not None and t % 6 == 0:
            if self._route_blocked(pose):
                return "blocked"
            goal = np.array(self.route.goal) if self.route.goal else None
            if goal is not None and self.mission.kind in (MissionKind.EXPLORE, MissionKind.RETURN) \
                    and np.linalg.norm(goal - pose.pos) < 22.0:
                if self.mission.kind == MissionKind.EXPLORE:
                    self._last_explore = -10 ** 9          # meneur arrivé : il recible le bord ouvert suivant
                    return None
                return "arrived"
        return None

    def _time_to_go_home(self, t, pose) -> bool:
        """Fin de partie : seuls les drones DANS la zone de retour comptent pour le score de retour (20 %). On rentre
        quand le temps restant approche la durée du trajet (route depuis la base sur la grille d'équipe, à vitesse
        réduite) plus une marge."""
        p = self.p
        if self.t_limit is None or self._home_xy is None or t < self.t_limit - p.home_window:
            return False
        G = self.team.grid
        r = G.route_to(pose.pos, t, p.v_cruise) if G is not None else None
        L = r.length if r is not None else 1.5 * float(np.linalg.norm(pose.pos - self._home_xy))
        return t + L / (p.home_speed_frac * p.v_cruise) + p.home_margin >= self.t_limit

    def _route_blocked(self, pose) -> bool:
        """Les 150 prochains px de la route traversent-ils un obstacle connu ?"""
        P, m = self.route.pts, self.world.map
        proj, seg, _ = self.follower._project(pose.pos)
        pts, left, cur = [], 150.0, proj
        for i in range(seg, len(P) - 1):
            nxt = P[i + 1]
            d = float(np.linalg.norm(nxt - cur))
            n = max(int(d / 6.0), 1)
            for k in range(1, n + 1):
                pts.append(cur + (nxt - cur) * (k / n))
            left -= d
            cur = nxt
            if left <= 0:
                break
        if not pts:
            return False
        ix, iy = m.world_to_cell(np.array(pts))
        return bool((m.view()[ix, iy] >= OCC_THR).any())

    # ------------------------------------------------------------------ plan
    def _speed(self, carrying: bool) -> float:
        return self.p.v_carry if carrying else self.p.v_cruise

    def _replan(self, t, pose, f) -> None:
        tic = time.perf_counter()
        self._last_plan = t
        p, w = self.p, self.world
        if self.mission.kind == MissionKind.RECEIVE and not f.carrying:
            self._refresh_follow(t, pose, f)                   # passage de relais en cours : on ne réaffecte pas
            return
        extra = extra_cost_from_agents(w, t, w.map, (w.map.W, w.map.H), self.me)
        route, chosen = None, None
        if self._homing and not f.carrying and self._home_xy is not None:     # fin de partie : retour au départ
            for k in (None, p.inflate_cells_tight):
                grid = PlanGrid(w, p, pose.pos, inflate=k, extra_cost=extra)
                grid.solve()
                route = grid.route_to(self._home_xy, t, self._speed(False))
                if route is not None:
                    self.mission = Mission(MissionKind.RETURN, (float(self._home_xy[0]), float(self._home_xy[1])), t)
                    self.mission_src = "homing"
                    self.route = route
                    self.follower.set_route(route)
                    self._arrived_t = None
                    self._stuck = 0
                    self.timing["n_plan"] += 1
                    return
        # porteur : le blessé dépasse du drone, on planifie avec plus de marge aux murs (repli si injoignable)
        inflates = (p.inflate_cells_carry, None, p.inflate_cells_tight) if f.carrying else (None, p.inflate_cells_tight)
        # dépose en cours : on la termine (aller prendre le blessé, puis le porter devant la porte), sauf si le blessé à
        # prendre n'existe plus (blessé imaginaire, parti, pris par un autre) ou y a été abandonné : retour à l'équipe
        if self.mission.kind == MissionKind.DROP and not f.carrying and self.mission.target is not None \
                and (not self._wounded_near(self.mission.target, 50.0)
                     or self.planner._banned(np.array(self.mission.target), t)):
            self.mission = Mission(MissionKind.IDLE)
        if self.mission.kind == MissionKind.DROP and self.mission.dest is not None:
            goal = self.mission.dest if f.carrying else self.mission.target
            for k in inflates:
                grid = PlanGrid(w, p, pose.pos, inflate=k, extra_cost=extra)
                grid.solve()
                route = grid.route_to(goal, t, self._speed(f.carrying))
                if route is not None:
                    break
            if route is not None:
                self.route = route
                self.follower.set_route(route)
                self._arrived_t = None
                self._stuck = 0
                self.mission_src = "drop"
                self.timing["n_plan"] += 1
                return
            if not f.carrying:
                self.planner.ban(self.mission.target, t + 400)     # injoignable : on renonce, l'équipe s'en chargera
        team_mission = None
        taken = []
        if not f.carrying:                                     # allocation d'équipe (grille commune, calculée à part)
            plan = self.team.plan(t, pose, self.mission)
            team_mission = plan.missions.get(self.me) if plan is not None else None
            taken = plan.taken_by_others(self.me) if plan is not None else []
            if team_mission is not None and team_mission.kind == MissionKind.FOLLOW \
                    and self._no_follow.get(team_mission.peer, -1) > t:
                team_mission = None             # un autre suiveur est plus près de ce meneur : je laisse
            if team_mission is not None and team_mission.target is not None \
                    and team_mission.kind in (MissionKind.EXPLORE, MissionKind.GOTO_WOUNDED, MissionKind.DROP) \
                    and self.planner._banned(np.array(team_mission.target), t):
                team_mission = None             # j'y ai déjà échoué : je laisse cette tâche (choix individuel)
            if team_mission is not None and team_mission.kind == MissionKind.EXPLORE \
                    and self.mission.kind == MissionKind.EXPLORE and self.mission.target is not None \
                    and team_mission.target is not None and not self.planner._banned(np.array(self.mission.target), t) \
                    and math.dist(team_mission.target, self.mission.target) < p.frontier_spacing:
                # même voie : on garde le bord ouvert affiné par _refresh_explore (le plan d'équipe vise la cellule de
                # la frontière la plus proche de la base ; les deux alternaient à chaque replanification)
                team_mission = replace(team_mission, target=self.mission.target)
        for k in inflates:
            grid = PlanGrid(w, p, pose.pos, inflate=k, extra_cost=extra)
            grid.solve()
            cands = [team_mission] if team_mission is not None and k is None else []
            cands += self.planner.candidates(t, pose, grid, f.carrying, self.mission, taken)   # repli individuel
            for cand in cands:
                if cand.target is None:
                    if cand.kind == MissionKind.IDLE:
                        chosen, route = cand, None
                        break
                    continue
                r = self._route_for(grid, cand, t, pose, f.carrying)
                if r is not None:
                    chosen, route = cand, r
                    break
                if cand.kind == MissionKind.EXPLORE:
                    self.planner.ban(cand.target, t + 300)
            if chosen is not None:
                break
        self.mission = chosen or Mission(MissionKind.IDLE)
        if self.mission.kind == MissionKind.EXPLORE and self.mission.target is not None:
            self._lane = self.mission.target
        if self.mission.kind == MissionKind.EXPLORE and self._dropped is not None                 and t - self._dropped[1] < p.drop_self_ban:
            self.mission = replace(self.mission, dest=self._dropped[0])   # dépose encore annoncée (cf. step)
        # origine de la mission (debug / mesures) : plan d'équipe, repli individuel faute de route, ou pas de ligne
        self.mission_src = ("carry" if f.carrying else "team" if chosen is not None and chosen is team_mission
                            else "fallback" if team_mission is not None else "no-team")
        if f.carrying and self._handoff is not None and self.mission.kind == MissionKind.CARRY:
            self.mission = replace(self.mission, peer=self._handoff["peer"])   # demande de relais toujours annoncée
        self.route = route
        self.follower.set_route(route)
        self._arrived_t = None
        self._stuck = 0
        self._last_follow = t
        self._last_explore = t
        self.timing["n_plan"] += 1

    def _route_for(self, grid, cand, t, pose, carrying) -> Optional[Route]:
        if cand.kind not in (MissionKind.FOLLOW, MissionKind.RECEIVE):
            return grid.route_to(cand.target, t, self._speed(carrying))
        peer = self.world.agents.get(cand.peer)
        if peer is None or t - peer.t > 30:
            return None                                       # meneur / porteur perdu : candidat suivant
        r = grid.route_to(peer.pos, t, self._speed(carrying))
        back = self.p.follow_dist if cand.kind == MissionKind.FOLLOW else 25.0
        return None if r is None else _truncate(r, back, pose.pos)

    # ------------------------------------------------------------------ passage de relais du blessé
    def _handoff_logic(self, t, pose, f) -> None:
        """Porteur : demande à un relais plus proche du centre de prendre le blessé, puis le lâche quand ce
        relais l'a saisi. Receveur : répond à la demande. Tout passe par les missions annoncées (gossip)."""
        p, w = self.p, self.world
        if not p.handoff_enabled:
            return
        if f.carrying:
            h = self._handoff
            if h is None:
                if t % 10 or t - self._carry_since < 20 or self.mission.kind != MissionKind.CARRY:
                    return
                R = self.planner.rescue_point()
                if R is None:
                    return
                d_me = float(np.linalg.norm(R - pose.pos))
                best, bd = None, p.handoff_range
                for a in w.others():
                    if t - a.t > 5 or a.health <= 5 or self._handoff_ban.get(a.origin, -1) > t \
                            or a.mission.kind not in (MissionKind.RELAY, MissionKind.IDLE):
                        continue
                    d = float(np.linalg.norm(np.array(a.pos) - pose.pos))
                    if d < bd and float(np.linalg.norm(R - np.array(a.pos))) < d_me - p.handoff_gain:
                        best, bd = a.origin, d
                if best is not None:
                    self._handoff = {"peer": best, "since": t, "ack": False}
                    self.mission = replace(self.mission, peer=best)
                return
            a = w.agents.get(h["peer"])
            fresh = a is not None and t - a.t <= 10
            if fresh and a.mission.kind == MissionKind.RECEIVE and a.mission.peer == self.me:
                h["ack"] = True
            if fresh and h["ack"] and a.mission.kind == MissionKind.CARRY \
                    and np.linalg.norm(np.array(a.pos) - pose.pos) < 80.0:
                self._grasp = 0                                # il tient le blessé : on lâche
                self._handoff_release = True
                self._handoff = None
                return
            if not fresh or t - h["since"] > p.handoff_timeout or (not h["ack"] and t - h["since"] > 60):
                self._handoff_ban[h["peer"]] = t + 400
                self._handoff = None
                if self.mission.kind == MissionKind.CARRY:
                    self.mission = replace(self.mission, peer=-1)
            return
        self._handoff = None
        if self.mission.kind != MissionKind.RECEIVE:
            if self.mission.kind not in (MissionKind.RELAY, MissionKind.IDLE):
                return
            req = [a for a in w.others() if t - a.t <= 5 and a.mission.kind == MissionKind.CARRY
                   and a.mission.peer == self.me]
            if req:
                a = min(req, key=lambda x: x.origin)
                self.mission = Mission(MissionKind.RECEIVE, a.pos, t, a.origin)
                self._recv_since = t
                self._last_follow = -10 ** 9                   # route vers le porteur au prochain pas
            return
        g = w.agents.get(self.mission.peer)
        if g is None or t - g.t > 10 or g.mission.kind != MissionKind.CARRY or g.mission.peer != self.me \
                or t - self._recv_since > p.handoff_timeout:
            self.mission = Mission(MissionKind.IDLE)          # demande retirée / échouée : retour à l'équipe
            self._plan_reason = "stuck"

    # ------------------------------------------------------------------ priorités de passage
    @staticmethod
    def _prio(kind) -> int:
        return {MissionKind.CARRY: 5, MissionKind.DROP: 4, MissionKind.RECEIVE: 4, MissionKind.GOTO_WOUNDED: 3,
                MissionKind.EXPLORE: 2, MissionKind.FOLLOW: 2, MissionKind.RETURN: 1,
                MissionKind.RELAY: 1}.get(kind, 0)

    def _my_dir(self, pose) -> Optional[np.ndarray]:
        if self.route is None or len(self.route.pts) < 2:
            return None
        proj, seg, _ = self.follower._project(pose.pos)
        c = self.follower._carrot(proj, seg, self.p.lookahead) - pose.pos
        n = float(np.linalg.norm(c))
        return c / n if n > 1e-6 else None

    def _yield_dir(self, t, pose, f) -> Optional[np.ndarray]:
        """Direction d'évitement si un drone plus prioritaire arrive sur moi (sa route annoncée passe près de moi).
        Priorité : porteur > receveur > va-chercher > explorateur/suiveur > relais/réserve, puis plus petit identifiant."""
        p = self.p
        mine = 5 if f.carrying else self._prio(self.mission.kind)
        me = pose.pos
        best, bd = None, p.yield_radius
        for a in self.world.others():
            if t - a.t > 3:
                continue
            ap = np.array(a.pos)
            d = float(np.linalg.norm(ap - me))
            if d >= bd:
                continue
            pa = self._prio(a.mission.kind)
            if pa < mine or (pa == mine and a.origin > self.me):
                continue
            if self.mission.kind in (MissionKind.FOLLOW, MissionKind.RECEIVE) and a.origin == self.mission.peer:
                continue                                       # on ne fuit pas celui qu'on rejoint
            if a.mission.kind == MissionKind.RECEIVE and a.mission.peer == self.me:
                continue                                       # ni celui qui vient prendre notre blessé
            pts = [ap] + [np.array((w.x, w.y)) for w in a.route if w.eta >= t]
            if len(pts) < 2:
                continue                                       # il ne va nulle part
            q, dq = _closest_on_polyline(np.array(pts), me)
            if dq >= p.yield_path_dist:
                continue
            their = pts[1] - pts[0]
            their = their / max(float(np.linalg.norm(their)), 1e-6)
            mine_dir = self._my_dir(pose)
            if mine_dir is not None and float(mine_dir @ their) > 0.3:
                # même sens : pas d'esquive. S'il est devant moi, j'attends derrière (file d'attente).
                if float((ap - me) @ mine_dir) > 0 and d < 60.0:
                    best, bd = ("wait", None), d
                continue
            best, bd = (q, ap), d
        if best is None:
            return None
        q, ap = best
        if isinstance(q, str):
            return np.zeros(2)                                 # attendre : vitesse nulle
        perp, away = me - q, me - ap
        v = away / max(np.linalg.norm(away), 1e-6) * 0.6
        if np.linalg.norm(perp) > 3.0:
            v = v + perp / np.linalg.norm(perp)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None

    def _resolve_conflicts(self, t, pose) -> None:
        """Doublons vus dans les missions ANNONCÉES par les autres (chaque drone publie à chaque pas sa mission : type,
        cible, pair ; à jour à quelques pas près dans le réseau) : deux drones vers le même blessé, deux suiveurs du
        même meneur. Règle commune à tous : le plus proche de la cible (à égalité, le plus petit identifiant) garde la
        tâche, l'autre la laisse et replanifie. Filet de sécurité quand les plans divergent (connaissances
        différentes, plans calculés à des instants différents)."""
        m = self.mission
        me = pose.pos
        if m.kind in (MissionKind.GOTO_WOUNDED, MissionKind.DROP) and m.target is not None:
            tgt = np.array(m.target)
            mine = float(np.linalg.norm(tgt - me))
            # Le plan d'équipe de mon réseau (longueur de chemin) me donne cette cible : il fait foi face aux drones de ce
            # réseau, qui calculent le même plan. Sinon la distance à vol d'oiseau ci-dessous désignait parfois l'autre
            # (mur entre les deux) : je laissais la cible, le plan me la redonnait, et ainsi de suite.
            plan = self.team.last
            mine_in_plan = None
            if plan is not None and t - plan.t <= self.p.replan_period:
                mp = plan.missions.get(self.me)
                if mp is not None and mp.target is not None and math.dist(mp.target, m.target) < 40.0:
                    mine_in_plan = set(plan.component)
            for a in self.world.others():
                if t - a.t > 10 or a.mission.target is None \
                        or a.mission.kind not in (MissionKind.GOTO_WOUNDED, MissionKind.DROP):
                    continue
                if mine_in_plan is not None and a.origin in mine_in_plan:
                    continue
                if np.linalg.norm(np.array(a.mission.target) - tgt) > 40.0:
                    continue
                theirs = float(np.linalg.norm(np.array(a.pos) - tgt))
                if theirs < mine - 5.0 or (abs(theirs - mine) <= 5.0 and a.origin < self.me):
                    self.planner.ban(tgt, t + 60)     # le plan ne me le redonnera pas tant que l'autre y va
                    self._grasp = 0
                    self.mission = Mission(MissionKind.IDLE)
                    self._plan_reason = "conflict"
                    return
        elif m.kind == MissionKind.FOLLOW and m.peer >= 0:
            lead = self.world.agents.get(m.peer)
            if lead is None:
                return
            lp = np.array(lead.pos)
            mine = float(np.linalg.norm(lp - me))
            for a in self.world.others():
                if t - a.t > 10 or a.mission.kind != MissionKind.FOLLOW or a.mission.peer != m.peer:
                    continue
                theirs = float(np.linalg.norm(np.array(a.pos) - lp))
                if theirs < mine - 5.0 or (abs(theirs - mine) <= 5.0 and a.origin < self.me):
                    self._no_follow[m.peer] = t + 60
                    self.mission = Mission(MissionKind.IDLE)
                    self._plan_reason = "conflict"
                    return

    def _refresh_explore(self, t, pose, f) -> None:
        """Meneur : vise en continu le bord ouvert le plus proche de lui plutôt qu'un point fixe. Au fil de
        l'avancée, la frontière recule et la cible avec elle : on suit l'ouvert. En profondeur : bonus pour finir la
        pièce où l'on est et pour rester sur sa voie ; les bords visés par un autre meneur sont évités."""
        self._last_explore = t
        p, w = self.p, self.world
        if self._take_nearby_wounded(t, pose):
            return
        grid = PlanGrid(w, p, pose.pos, extra_cost=extra_cost_from_agents(w, t, w.map, (w.map.W, w.map.H), self.me))
        grid.solve()
        fr = grid.frontiers(p.frontier_min_cells)
        if not fr:
            self._plan_reason = "stuck"                       # plus rien d'ouvert d'ici : nouvelle allocation
            return
        topo = self.team._topo
        my_reg = topo.region_at(pose.pos, search=3) if topo is not None else -1
        others = [np.array(a.mission.target) for a in w.others() if t - a.t < 60
                  and a.mission.kind == MissionKind.EXPLORE and a.mission.target is not None]
        lane = np.array(self.mission.target) if self.mission.target is not None else None
        best, bc = None, float("inf")
        cur, cc = None, float("inf")                          # meilleure frontière de ma voie actuelle
        for fx in fr:
            cxy = grid.m.cell_to_world(np.array([fx.cell[0]]), np.array([fx.cell[1]]))[0]
            if self.planner._banned(cxy, t):
                continue
            c = fx.cost - p.frontier_size_gain * fx.size
            if topo is not None and my_reg >= 0 and topo.region_of_cell(fx.cell) == my_reg:
                c -= p.lane_region_bonus
            if lane is not None and np.linalg.norm(cxy - lane) < p.lane_sticky_radius:
                c -= p.lane_sticky_bonus
            if any(np.linalg.norm(cxy - q) < p.frontier_spacing for q in others):
                c += p.claim_penalty
            if lane is not None and np.linalg.norm(cxy - lane) < p.lane_sticky_radius and c < cc:
                cur, cc = cxy, c
            if c < bc:
                best, bc = cxy, c
        if best is None:
            return
        # La carte est fusionnée par gossip avec quelques pas de retard. Sans hystérésis ici, deux frontières presque
        # équivalentes font repartir le meneur alternativement dans les deux directions. On garde la frontière ACTUELLE
        # de sa voie (pas l'ancien point visé : arrivé dessus, il s'y arrêterait) ; marge relative valable aussi pour des
        # coûts négatifs (bonus de voie et de taille).
        if cur is not None and bc > cc - (1.0 - p.switch_gain) * abs(cc):
            best = cur
        # on quitte une pièce finie qui contient encore un blessé : on le dépose d'abord devant la porte
        drop = self.team.drop_task(t, pose.pos, leaving_to=best)
        if drop is not None and not self.planner._banned(np.array(drop.target), t):
            self.mission = drop
            self._take_full = False
            self._plan_reason = "stuck"                       # route vers le blessé au pas suivant (cf. _replan)
            return
        r = grid.route_to(best, t, self._speed(f.carrying))
        if r is None:
            return
        self.mission = replace(self.mission, target=(float(best[0]), float(best[1])))
        self._lane = self.mission.target
        self.route = r
        self.follower.set_route(r)

    def _mark_near(self, xy) -> None:
        """world.near : cellules passées à moins de check_near d'un drone (moi, ou un drone de mon réseau d'après sa
        position annoncée). Un amas suspect ainsi couvert est réglé (cf. PlanGrid.frontiers)."""
        m = self.world.map
        if self.world.near is None:
            self.world.near = np.zeros((m.W, m.H), bool)
            k = int(self.p.check_near // m.res)
            g = np.arange(-k, k + 1)
            gx, gy = np.meshgrid(g, g, indexing="ij")
            self._disc = (gx * gx + gy * gy) * m.res ** 2 <= self.p.check_near ** 2
        k = self._disc.shape[0] // 2
        ix, iy = m.world_to_cell(np.asarray(xy, float))
        cx, cy = int(ix[0]), int(iy[0])
        x0, x1, y0, y1 = max(cx - k, 0), min(cx + k + 1, m.W), max(cy - k, 0), min(cy + k + 1, m.H)
        self.world.near[x0:x1, y0:y1] |= self._disc[x0 - cx + k:x1 - cx + k, y0 - cy + k:y1 - cy + k]

    def _take_nearby_wounded(self, t, pose) -> bool:
        """Meneur à moins de explore_take_dist d'un blessé libre : il le prend lui-même et le ramène au centre. Sinon
        il explorait l'ombre lidar du blessé en s'y collant, et gênait le drone envoyé le chercher. Mission DROP (tâche
        en cours, hors allocation d'équipe, cf. teamplan._BUSY) convertie en sauvetage complet à la saisie ; son
        suiveur, voyant que son meneur n'explore plus, replanifie et reprend la voie."""
        p, me = self.p, pose.pos
        best, bd = None, p.explore_take_dist
        for o in self.world.objects_view(ObjKind.WOUNDED):
            if o.state != ObjState.SEEN:
                continue
            op = np.array(o.pos)
            d = float(np.linalg.norm(op - me))
            if d >= bd or self.planner._banned(op, t):
                continue
            if any(t - a.t < 30 and a.mission.target is not None
                   and a.mission.kind in (MissionKind.GOTO_WOUNDED, MissionKind.DROP, MissionKind.CARRY)
                   and np.linalg.norm(np.array(a.mission.target) - op) < 40.0
                   and (a.mission.kind != MissionKind.GOTO_WOUNDED or np.linalg.norm(np.array(a.pos) - op) < d)
                   for a in self.world.others()):
                continue                                      # quelqu'un s'en charge déjà (ou est plus près)
            best, bd = op, d
        if best is None:
            return False
        xy = (float(best[0]), float(best[1]))
        self.mission = Mission(MissionKind.DROP, xy, t, -1, xy)
        self.mission_src = "take"
        self._take_full = True
        self._plan_reason = "stuck"                           # route vers le blessé au pas suivant (cf. _replan)
        return True

    def _refresh_follow(self, t, pose, f) -> None:
        """Suiveur : la route vers le meneur est recalculée souvent, sans refaire l'allocation d'équipe."""
        self._last_follow = t
        w = self.world
        grid = PlanGrid(w, self.p, pose.pos, extra_cost=extra_cost_from_agents(w, t, w.map, (w.map.W, w.map.H), self.me))
        grid.solve()
        r = self._route_for(grid, self.mission, t, pose, f.carrying)
        if r is None:
            self._plan_reason = "stuck"                       # meneur perdu ou injoignable : nouvelle allocation
            return
        self.route = r
        self.follower.set_route(r)

    # ------------------------------------------------------------------ contrôle
    def _control(self, t, pose, f, seen) -> Command:
        p = self.p
        v_est = pose.vel
        override_dir, override_speed = None, 0.0
        hold = False
        repulse = True
        spin = False
        lidar = f.lidar

        if self._pending_drop is not None:
            key, deadline, t_drop = self._pending_drop
            if any(st == ObjState.SEEN and d < 70.0 for _, d, st in seen["wounded"]) and t > t_drop + 2:
                self.world.set_state(key, ObjState.SEEN, t)           # toujours là : non sauvé
                self._pending_drop = None
            elif t >= deadline:                                       # disparu : sauvé (déjà marqué au lâcher)
                self._pending_drop = None

        if self.mission.kind in (MissionKind.GOTO_WOUNDED, MissionKind.DROP) and not f.carrying and self.mission.target is not None:
            tgt = np.array(self.mission.target)
            vis = [(pos, d) for pos, d, st in seen["wounded"] if st == ObjState.SEEN and np.linalg.norm(pos - tgt) < 60.0]
            if vis:
                self._sighting_t = t
                pos, d = min(vis, key=lambda x: x[1])
                if d < 90.0:                                       # approche finale à vue, jusqu'au contact
                    override_dir, override_speed, lidar = self._approach(pose, pos, d, f)
                    self._wounded_bearing = math.atan2(override_dir[1], override_dir[0]) - pose.theta
                    if d < p.grasp_close_dist:
                        self._grasp, self._grasp_since = 1, t
            elif t % 5 == 0 and not self._wounded_near(tgt, 50.0):
                self._plan_reason = "stuck"                    # cible retirée (ou blessé mobile parti) : réaffectation
            else:
                close = np.linalg.norm(tgt - pose.pos) < 35.0
                if close or (self.route is not None and self.follower.remaining(pose.pos) < 25.0):
                    if self._arrived_t is None:
                        self._arrived_t = t
                    elif t - self._arrived_t > 25 and t - self._sighting_t > 25:
                        if close:                              # sur place et rien en vue : il n'est plus là
                            self._forget_wounded_near(tgt, 50.0, t, seen)
                        # sinon fin de route loin de la cible (injoignable) : on y renonce, sans l'effacer
                        self.planner.ban(tgt, t + (600 if close else 300))
                        self._arrived_t = None
                        self._plan_reason = "stuck"
            if self._grasp and not f.carrying and t - self._grasp_since > 40:
                self._grasp = 0                                    # saisie ratée : on réessaie

        if self.mission.kind == MissionKind.RECEIVE and not f.carrying:
            g = self.world.agents.get(self.mission.peer)
            if g is not None:                                  # blessé porté par le donneur, vu près de lui
                gp = np.array(g.pos)
                vis = [(pos, d) for pos, d, st in seen["wounded"] if st == ObjState.CARRIED
                       and np.linalg.norm(pos - gp) < 60.0]
                if vis:
                    pos, d = min(vis, key=lambda x: x[1])
                    if d < 90.0:
                        override_dir, override_speed, lidar = self._approach(pose, pos, d, f)
                        if d < p.grasp_close_dist:
                            self._grasp, self._grasp_since = 1, t
                elif np.linalg.norm(gp - pose.pos) < 80.0:
                    # blessé caché derrière le porteur (la vue sémantique s'arrête au premier drone) : on tourne
                    # autour du porteur à ~50 px jusqu'à le voir
                    rel = pose.pos - gp
                    a = math.atan2(rel[1], rel[0]) + math.radians(50.0)
                    goal = gp + 50.0 * np.array((math.cos(a), math.sin(a)))
                    to = goal - pose.pos
                    override_dir = to / max(float(np.linalg.norm(to)), 1e-6)
                    override_speed = 1.0
            if self._grasp and t - self._grasp_since > 40:
                self._grasp = 0

        if f.carrying and self._handoff is not None:           # le receveur arrive : on l'attend sur place
            g = self.world.agents.get(self._handoff["peer"])
            if g is not None and np.linalg.norm(np.array(g.pos) - pose.pos) < 90.0:
                hold = True

        if f.carrying and self.mission.kind == MissionKind.DROP and self.mission.dest is not None:
            # dépose devant la porte : on lâche au point prévu, la piste reste « à prendre » (cf. step)
            self._grasp = 1
            if np.linalg.norm(np.array(self.mission.dest) - pose.pos) < 25.0:
                self._grasp = 0
                self._drop_release = True
        elif f.carrying:
            # Le blessé est sauvé quand IL TOUCHE le centre de secours : on garde la pince fermée et on fonce
            # dessus à vue. Le lidar voit le centre comme un mur : sans ça, marge et répulsion nous arrêtent avant.
            self._grasp = 1
            near = [(pos, d) for pos, d in seen["rescue"] if d < p.rescue_approach_dist]
            if near:
                pos, d = min(near, key=lambda x: x[1])
                to = pos - pose.pos
                override_dir = to / max(float(np.linalg.norm(to)), 1e-6)
                override_speed = p.rescue_approach_speed
                repulse = False
                # Le blessé est tourné VERS le centre (cap réglé plus bas) : c'est lui qui touche en premier.
                # Secours : si on est collé depuis longtemps sans sauvetage, on tourne sur place (le pivot fait
                # orbiter le blessé, qui balaie le centre).
                if d < p.rescue_spin_dist:
                    self._rc_close_since = t if self._rc_close_since is None else self._rc_close_since
                    spin = t - self._rc_close_since > p.rescue_spin_after
                else:
                    self._rc_close_since = None

        # meneur d'un binôme : il attend son suiveur s'il le distance (la radio doit passer entre eux)
        if self.mission.kind == MissionKind.EXPLORE and not f.carrying:
            fol = [a for a in self.world.others() if a.mission.kind == MissionKind.FOLLOW
                   and a.mission.peer == self.me and t - a.t < 30]
            if fol and min(np.linalg.norm(np.array(a.pos) - pose.pos) for a in fol) > p.pair_max:
                hold = True

        # ---- blocage : on devrait avancer mais on n'a presque pas bougé depuis 60 pas -> manœuvre de dégagement
        if t % 10 == 0:
            self._hist.append((t, pose.pos.copy()))
        expect_move = (not hold and not spin and self.route is not None
                       and (override_dir is not None or self.follower.remaining(pose.pos) > 25.0))
        ydir = None if t < self._recover_until else self._yield_dir(t, pose, f)
        if ydir is not None and not ydir.any():      # même sens, il est juste devant : on attend (pas trop longtemps)
            if self._wait_since is None:
                self._wait_since = t
            if t - self._wait_since > p.queue_max_wait or t < self._no_wait_until:
                if t - self._wait_since > p.queue_max_wait:
                    self._no_wait_until, self._wait_since = t + 60, None   # il est bloqué devant : on n'attend plus
                ydir = None
        else:
            self._wait_since = None
        if ydir is not None and not ydir.any():
            hold = True
            override_dir, spin = None, False
            expect_move = False
            self._hist.clear()
            ydir = None
        if ydir is not None:                        # un drone plus prioritaire arrive : on lui laisse le passage
            override_dir, override_speed, repulse, lidar = ydir, p.yield_speed, True, f.lidar
            expect_move = False
            self._hist.clear()
        if t < self._recover_until:
            override_dir, override_speed, repulse = self._recover_dir, 1.0, True
        elif self._recover_until >= 0:              # fin de manœuvre : on replanifie depuis la nouvelle position
            self._recover_until = -1
            self._plan_reason = "stuck"
        elif expect_move and len(self._hist) == self._hist.maxlen                 and np.linalg.norm(self._hist[-1][1] - self._hist[0][1]) < 10.0:
            self._start_recovery(t, pose, f)
            if self.mission.target is not None and self.mission.kind == MissionKind.EXPLORE:
                self.planner.ban(self.mission.target, t + 400)
            override_dir, override_speed = self._recover_dir, 1.0

        recovering = t < self._recover_until
        if f.carrying and recovering and self._recover_theta is not None:
            # en tournant, le pivot fait orbiter le blessé : il se décroche. Rotation BORNÉE (une rotation libre
            # pendant toute la manœuvre faisait un tour complet)
            self.theta_ref = self._recover_theta
        elif f.carrying and not spin:
            # cap du porteur : blessé DERRIÈRE nous dans le sens de la marche (il suit la trajectoire au lieu
            # d'accrocher les murs dans les virages), DEVANT nous à l'approche du centre de secours
            if override_dir is not None and not repulse:
                self.theta_ref = _wrap_angle(math.atan2(override_dir[1], override_dir[0]) - self._carry_bearing)
            else:
                u = override_dir if override_dir is not None else self._my_dir(pose)
                if override_dir is None and self.route is not None and self.follower.remaining(pose.pos) < 30.0:
                    u = None                        # fin de route : la direction de la carotte n'est que du bruit
                if u is not None:
                    want = _wrap_angle(math.atan2(-u[1], -u[0]) - self._carry_bearing)
                    if abs(_wrap_angle(want - self.theta_ref)) > math.radians(p.carry_heading_tol):
                        self.theta_ref = want

        end_stop = self.mission.kind in (MissionKind.EXPLORE, MissionKind.CARRY, MissionKind.GOTO_WOUNDED, MissionKind.DROP,
                                         MissionKind.RETURN, MissionKind.IDLE, MissionKind.RELAY, MissionKind.FOLLOW)
        cmd = self.follower.command(pose, v_est, lidar, f.angles, self.theta_ref, self._speed(f.carrying),
                                    grasp=self._grasp, hold=hold or self.route is None,
                                    override_dir=override_dir, override_speed=override_speed, end_stop=end_stop,
                                    repulse=repulse,
                                    accel=p.accel_per_cmd_carry if f.carrying else p.accel_per_cmd)
        if spin:
            cmd = Command(cmd.forward, cmd.lateral, 1.0, cmd.grasper)
            self.theta_ref = pose.theta            # on garde le cap atteint une fois la manœuvre faite
        return cmd

    def _approach(self, pose, pos, d, f):
        """Approche finale d'un blessé : droit dessus, vitesse limitée par le freinage mais jamais nulle (le heurter
        ne coûte rien), rayons lidar vers lui ignorés (sinon la répulsion et le freinage nous en tiennent à l'écart)."""
        p = self.p
        to = pos - pose.pos
        dirv = to / max(float(np.linalg.norm(to)), 1e-6)
        a_b = 0.5 * p.brake_frac * p.accel_per_cmd
        speed = min(p.v_cruise, max(p.approach_min_speed, math.sqrt(2 * a_b * max(d - 15.0, 0.0))))
        lidar = f.lidar
        if lidar is not None:
            rel = np.asarray(f.angles) + pose.theta - math.atan2(dirv[1], dirv[0])
            rel = (rel + np.pi) % (2 * np.pi) - np.pi
            lidar = np.where(np.abs(rel) < math.radians(p.approach_mask_deg), np.inf, lidar)
        return dirv, speed, lidar

    def _start_recovery(self, t, pose, f) -> None:
        """Recul à l'opposé de l'obstacle le plus proche, dévié de ±40° (alterné à chaque tentative)."""
        self._n_recover += 1
        self._hist.clear()
        self._recover_until = t + self.p.recover_steps
        away = -np.array((math.cos(pose.theta), math.sin(pose.theta)))
        if f.lidar is not None:
            r = np.where(np.isfinite(f.lidar), f.lidar, np.inf)
            i = int(np.argmin(r))
            a = pose.theta + float(f.angles[i])
            away = -np.array((math.cos(a), math.sin(a)))
        dev = math.radians(40.0) * (1 if self._n_recover % 2 else -1)
        c, s_ = math.cos(dev), math.sin(dev)
        self._recover_dir = np.array((c * away[0] - s_ * away[1], s_ * away[0] + c * away[1]))
        turn = math.radians(self.p.recover_carry_turn) * (1 if self._n_recover % 2 else -1)
        self._recover_theta = _wrap_angle(pose.theta + turn) if f.carrying else None


def _truncate(r: Route, back: float, here) -> Route:
    """Raccourcit la route de `back` px depuis sa fin (le suiveur s'arrête derrière le meneur).
    Si elle est plus courte que ça, route réduite au point courant : on attend sur place."""
    P, E = r.pts, r.eta
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    left = back
    i = len(P) - 1
    while i > 0 and seg[i - 1] <= left:
        left -= seg[i - 1]
        i -= 1
    if i == 0:
        p0 = np.asarray(here, float)
        return Route(np.array([p0, p0]), np.array([r.t0, r.t0]), (float(p0[0]), float(p0[1])), r.t0)
    a, b = P[i - 1], P[i]
    end = b + (a - b) * (left / max(seg[i - 1], 1e-9))
    pts = np.vstack([P[:i], end[None, :]])
    eta = np.r_[E[:i], E[i]]
    return Route(pts, eta, (float(end[0]), float(end[1])), r.t0)


def _closest_on_polyline(P: np.ndarray, x: np.ndarray):
    """Point de la polyligne P (N, 2) le plus proche de x, et sa distance."""
    best, bd = P[0], float(np.linalg.norm(P[0] - x))
    for a, b in zip(P[:-1], P[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        s = 0.0 if L2 < 1e-9 else float(np.clip((x - a) @ ab / L2, 0.0, 1.0))
        q = a + s * ab
        d = float(np.linalg.norm(x - q))
        if d < bd:
            best, bd = q, d
    return best, bd


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
