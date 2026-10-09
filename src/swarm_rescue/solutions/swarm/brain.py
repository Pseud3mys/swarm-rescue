"""Orchestrateur : SensorFrame -> Command, un pas à la fois. Aucune dépendance au simulateur.

Ordre d'un pas :  radio -> estimation (EKF + amers) -> perception (carte, objets) ->
                  mission + chemin (sur événement) -> asservissement -> publication de l'état aux voisins.
Chaque étape est dans son propre module ; ce fichier ne contient que la logique de séquencement
et les règles d'événements (quand replanifier, détection de blocage, saisie / dépose).
"""
from __future__ import annotations

import math
import time
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
        t3 = T()

        # ---- asservissement
        cmd = self._control(t, pose, f, seen)
        self.last_cmd = cmd
        t4 = T()

        # ---- publication
        w.publish_self(t, pose.pos, pose.vel, f.health, self.mission,
                       self.route.waypoints() if self.route is not None else ())
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
        for k in (None, p.inflate_cells_tight):
            grid = PlanGrid(w, p, pose.pos, inflate=k, extra_cost=extra)
            grid.solve()
            for cand in self.planner.candidates(t, pose, grid, f.carrying, self.mission):
                if cand.target is None:
                    if cand.kind == MissionKind.IDLE:
                        chosen, route = cand, None
                        break
                    continue
                r = grid.route_to(cand.target, t, self._speed(f.carrying))
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
        self.timing["n_plan"] += 1

    # ------------------------------------------------------------------ contrôle
    def _control(self, t, pose, f, seen) -> Command:
        p = self.p
        v_est = pose.vel
        override_dir, override_speed = None, 0.0
        hold = False

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
                    override_dir, override_speed = dirv, min(0.7, max(0.15, d * 0.012))
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
            self._grasp = 1
            arrived = (self.mission.kind == MissionKind.CARRY and self.mission.target is not None
                       and np.linalg.norm(np.array(self.mission.target) - pose.pos) < 14.0)
            if arrived or any(d < p.release_dist for _, d in seen["rescue"]):
                self._grasp = 0                                    # dépose (verdict dans _pending_drop)
                hold = True

        end_stop = self.mission.kind in (MissionKind.EXPLORE, MissionKind.CARRY, MissionKind.GOTO_WOUNDED,
                                         MissionKind.RETURN, MissionKind.IDLE)
        cmd = self.follower.command(pose, v_est, f.lidar, f.angles, self.theta_ref, self._speed(f.carrying),
                                    grasp=self._grasp, hold=hold or self.route is None,
                                    override_dir=override_dir, override_speed=override_speed, end_stop=end_stop)

        # blocage : on commande de l'accélération mais on ne bouge pas
        moving_cmd = self.follower.last_cmd_mag > 0.6 and self.route is not None and override_dir is None
        if moving_cmd and np.linalg.norm(v_est) < 0.1:
            self._stuck += 1
        else:
            self._stuck = max(self._stuck - 2, 0)
        if self._stuck > 90:
            if self.mission.target is not None and self.mission.kind == MissionKind.EXPLORE:
                self.planner.ban(self.mission.target, t + 400)
            self._plan_reason = "stuck"
            self._stuck = 0
        return cmd
