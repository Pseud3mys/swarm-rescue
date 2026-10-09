"""Orchestrateur : SensorFrame -> Command, un pas à la fois. Aucune dépendance au simulateur.

Ordre d'un pas :  radio -> estimation (EKF + amers) -> perception (carte, objets) ->
                  mission + chemin (sur événement) -> asservissement -> publication de l'état aux voisins.
Chaque étape est dans son propre module ; ce fichier ne contient que la logique de séquencement
et les règles d'événements (quand replanifier, détection de blocage, saisie / dépose).
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from .comms import Gossip, GossipConfig, extract_packets
from .config import Params
from .controller import PathFollower
from .estimator import PoseEKF
from .gridmap import OCC_THR
from .landmarks import LandmarkTracker
from .missions import MissionPlanner
from .pathing import PlanGrid, extra_cost_from_agents
from .perception import ScanIntegrator, drone_ray_mask, observe_semantic
from .schema import Command, Mission, MissionKind, ObjKind, ObjState, Packet, Route, SensorFrame
from .teamplan import TeamPlanner
from .world import WorldModel


class Brain:
    def __init__(self, me: int, size_area, params: Optional[Params] = None):
        self.p = p = params or Params()
        self.me = me
        self.world = WorldModel(me, size_area, p.resolution, p.chunk)
        self.gossip = Gossip(self.world, GossipConfig())
        self.ekf = PoseEKF(p)
        self.landmarks = LandmarkTracker(self.world, self.ekf, p, min_gap=30)
        self.scan = ScanIntegrator(self.world, p)
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
        self._n_recover = 0
        self._last_follow = -10 ** 9
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
                         (self.last_cmd.forward, self.last_cmd.lateral), self.last_cmd.rotation)
        self.ekf.update_gps(f.gps, f.compass)
        self.ekf.update_compass(f.compass)
        if not self.ekf.initialized:
            return Command()
        if f.lidar is not None and t % 2 == 0:
            self.landmarks.process(t, f.lidar, f.angles)
        pose = self.ekf.pose()
        if self.theta_ref is None:
            self.theta_ref = pose.theta
        t1 = T()

        # ---- perception
        if not self._home_set and f.in_return_area:
            w.observe_object(ObjKind.RETURN_AREA, pose.pos, t)
            self._home_set = True
        if f.lidar is not None:
            self.scan.integrate(pose, f.lidar, f.angles, drone_ray_mask(f.lidar, f.angles, f.semantic))
        seen = observe_semantic(w, pose, f.semantic, t)
        t2 = T()

        # ---- mission / chemin
        if f.carrying != self._was_carrying:
            self._plan_reason = "carry-change"
            if f.carrying:                                         # prise : la piste du blessé passe à CARRIED (sur place)
                tr = w.nearest_own(ObjKind.WOUNDED, pose.pos, 80.0)
                if tr is None:
                    tr = w.observe_object(ObjKind.WOUNDED, pose.pos, t, ObjState.SEEN)
                self._carry_key = tr.key
                w.set_state(tr.key, ObjState.CARRIED, t)
            elif self._carry_key is not None:                      # dépose : verdict après quelques pas (cf. _control)
                self._pending_drop = (self._carry_key, t + self.p.drop_check_steps, t)
                self._carry_key = None
            self._grasp = 1 if f.carrying else 0
            self._was_carrying = f.carrying
        reason = self._replan_reason(t, pose, f, seen)
        if reason:
            self.last_reason = reason
            self._replan(t, pose, f)
        elif self.mission.kind == MissionKind.FOLLOW and t - self._last_follow >= p.follow_replan:
            self._refresh_follow(t, pose, f)
        t3 = T()

        # ---- asservissement
        cmd = self._control(t, pose, f, seen)
        self.last_cmd = cmd
        t4 = T()

        # ---- publication
        w.publish_self(t, pose.pos, pose.vel, f.health, self.mission,
                       self.route.waypoints() if self.route is not None else (), self.gossip.heard)
        self._packet = self.gossip.build_packet(t)
        for k, a, b in (("est", t0, t1), ("map", t1, t2), ("plan", t2, t3), ("ctl", t3, t4)):
            self.timing[k] += b - a
        return cmd

    # ------------------------------------------------------------------ événements de replanification
    def _replan_reason(self, t, pose, f, seen) -> Optional[str]:
        p = self.p
        if self._plan_reason in ("carry-change", "stuck"):
            r, self._plan_reason = self._plan_reason, ""
            return r
        if self.route is None and t - self._last_plan >= 10:
            return "no-route"
        if t - self._last_plan >= p.replan_period:
            return "period"
        if t % 5 == 0:                                   # un blessé apparaît / disparaît : l'équipe se réorganise
            n = sum(1 for o in self.world.objects_view(ObjKind.WOUNDED) if o.state == ObjState.SEEN)
            if n != self._n_free_wounded:
                self._n_free_wounded = n
                return "wounded-change"
        if self.route is not None and t % 3 == 0:
            if self._route_blocked(pose):
                return "blocked"
            goal = np.array(self.route.goal) if self.route.goal else None
            if goal is not None and self.mission.kind in (MissionKind.EXPLORE, MissionKind.RETURN) \
                    and np.linalg.norm(goal - pose.pos) < 22.0:
                return "arrived"
        return None

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
        extra = extra_cost_from_agents(w, t, w.map, (w.map.W, w.map.H), self.me)
        route, chosen = None, None
        team_mission = None
        for k in (None, p.inflate_cells_tight):
            grid = PlanGrid(w, p, pose.pos, inflate=k, extra_cost=extra)
            grid.solve()
            cands = []
            if not f.carrying:
                if k is None:                                  # allocation d'équipe : une fois, sur la grille nominale
                    plan = self.team.plan(t, pose, grid, self.mission)
                    team_mission = plan.missions.get(self.me) if plan is not None else None
                if team_mission is not None:
                    cands.append(team_mission)
            cands += self.planner.candidates(t, pose, grid, f.carrying, self.mission)   # repli : choix individuel
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
        self.route = route
        self.follower.set_route(route)
        self._arrived_t = None
        self._stuck = 0
        self._last_follow = t
        self.timing["n_plan"] += 1

    def _route_for(self, grid, cand, t, pose, carrying) -> Optional[Route]:
        if cand.kind != MissionKind.FOLLOW:
            return grid.route_to(cand.target, t, self._speed(carrying))
        leader = self.world.agents.get(cand.peer)
        if leader is None or t - leader.t > 30:
            return None                                       # meneur perdu : on passe au candidat suivant
        r = grid.route_to(leader.pos, t, self._speed(carrying))
        return None if r is None else _truncate(r, self.p.follow_dist, pose.pos)

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

        if self._pending_drop is not None:
            key, deadline, t_drop = self._pending_drop
            if any(st == ObjState.SEEN and d < 70.0 for _, d, st in seen["wounded"]) and t > t_drop + 2:
                self.world.set_state(key, ObjState.SEEN, t)           # toujours là : non sauvé
                self._pending_drop = None
            elif t >= deadline:
                self.world.set_state(key, ObjState.RESCUED, t)        # disparu : sauvé
                self._pending_drop = None

        if self.mission.kind == MissionKind.GOTO_WOUNDED and not f.carrying and self.mission.target is not None:
            tgt = np.array(self.mission.target)
            vis = [(pos, d) for pos, d, st in seen["wounded"] if st == ObjState.SEEN and np.linalg.norm(pos - tgt) < 60.0]
            if vis:
                self._sighting_t = t
                pos, d = min(vis, key=lambda x: x[1])
                if d < 90.0:                                       # approche finale à vue
                    dirv = (pos - pose.pos) / max(np.linalg.norm(pos - pose.pos), 1e-6)
                    a_b = 0.5 * p.brake_frac * p.accel_per_cmd     # freinage prudent : la boucle de vitesse a du retard
                    override_dir = dirv
                    override_speed = min(p.v_cruise, max(0.3, math.sqrt(2 * a_b * max(d - 0.6 * p.grasp_dist, 0.0))))
                    if d < p.grasp_dist:
                        self._grasp, self._grasp_since = 1, t
            elif np.linalg.norm(tgt - pose.pos) < 35.0:
                if self._arrived_t is None:
                    self._arrived_t = t
                elif t - self._arrived_t > 25 and t - self._sighting_t > 25:   # il n'est plus là
                    self.world.observe_object(ObjKind.WOUNDED, tgt, t, ObjState.RESCUED)
                    self.planner.ban(tgt, t + 600)
                    self._plan_reason = "stuck"
            if self._grasp and not f.carrying and t - self._grasp_since > 40:
                self._grasp = 0                                    # saisie ratée : on réessaie

        if f.carrying:
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
                # Collé au centre : le blessé (PivotJoint) traîne derrière nous et ne le touche pas.
                # En tournant sur nous-mêmes, le point de pivot fait orbiter le blessé, qui balaie le centre.
                spin = d < p.rescue_spin_dist

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
        if t < self._recover_until:
            override_dir, override_speed, repulse = self._recover_dir, 1.0, True
            spin = f.carrying                       # en tournant, le pivot fait orbiter le blessé : il se décroche
        elif self._recover_until >= 0:              # fin de manœuvre : on replanifie depuis la nouvelle position
            self._recover_until = -1
            self._plan_reason = "stuck"
        elif expect_move and len(self._hist) == self._hist.maxlen                 and np.linalg.norm(self._hist[-1][1] - self._hist[0][1]) < 10.0:
            self._start_recovery(t, pose, f)
            if self.mission.target is not None and self.mission.kind == MissionKind.EXPLORE:
                self.planner.ban(self.mission.target, t + 400)
            override_dir, override_speed = self._recover_dir, 1.0
            spin = f.carrying

        end_stop = self.mission.kind in (MissionKind.EXPLORE, MissionKind.CARRY, MissionKind.GOTO_WOUNDED,
                                         MissionKind.RETURN, MissionKind.IDLE, MissionKind.RELAY, MissionKind.FOLLOW)
        cmd = self.follower.command(pose, v_est, f.lidar, f.angles, self.theta_ref, self._speed(f.carrying),
                                    grasp=self._grasp, hold=hold or self.route is None,
                                    override_dir=override_dir, override_speed=override_speed, end_stop=end_stop,
                                    repulse=repulse)
        if spin:
            cmd = Command(cmd.forward, cmd.lateral, 1.0, cmd.grasper)
            self.theta_ref = pose.theta            # on garde le cap atteint une fois la manœuvre faite
        return cmd

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
