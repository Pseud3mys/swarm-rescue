"""Asservissement : (pose, vitesse, route, lidar) -> commandes.

Le drone est une masse commandée en FORCE avec très peu de frottement et une rotation très lente :
  - on pilote en vitesse (holonome : avant + latéral), jamais en tournant sur place ;
  - on respecte la distance de freinage : v <= sqrt(2 * a_frein * distance_libre) vers la fin de route,
    dans les virages et devant les obstacles vus au lidar ;
  - le cap est simplement maintenu (rotation ~ 0).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .schema import Command, Pose, Route


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PathFollower:
    def __init__(self, params: Params):
        self.p = params
        self.route: Optional[Route] = None
        self.seg = 0
        self.last_cmd_mag = 0.0

    def set_route(self, route: Optional[Route]) -> None:
        self.route = route
        self.seg = 0

    # ---------------------------------------------------------------------------
    def remaining(self, pos) -> float:
        """Longueur de route restante à partir de la projection de `pos`."""
        if self.route is None or len(self.route) == 0:
            return 0.0
        P = self.route.pts
        proj, seg, _ = self._project(pos)
        if seg + 1 >= len(P):
            return float(np.linalg.norm(P[-1] - pos))
        rest = np.linalg.norm(np.diff(P[seg + 1:], axis=0), axis=1).sum() if seg + 2 <= len(P) - 1 else 0.0
        return float(np.linalg.norm(P[seg + 1] - proj) + rest)

    def _project(self, pos):
        P = self.route.pts
        if len(P) == 1:
            return P[0], 0, 0.0
        lo, hi = self.seg, min(self.seg + 8, len(P) - 1)
        best = (1e18, None, lo, 0.0)
        for i in range(lo, hi):
            a, b = P[i], P[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            s = 0.0 if L2 < 1e-9 else float(np.clip((pos - a) @ ab / L2, 0.0, 1.0))
            q = a + s * ab
            d = float(np.linalg.norm(pos - q))
            if d < best[0]:
                best = (d, q, i, s)
        self.seg = best[2]
        return best[1], best[2], best[0]

    def _carrot(self, proj, seg, dist_ahead):
        P = self.route.pts
        pt, left = proj, dist_ahead
        for i in range(seg, len(P) - 1):
            nxt = P[i + 1]
            d = float(np.linalg.norm(nxt - pt))
            if d >= left:
                return pt + (nxt - pt) * (left / max(d, 1e-9))
            left -= d
            pt = nxt
        return P[-1]

    # ---------------------------------------------------------------------------
    def command(self, pose: Pose, v_est: np.ndarray, ranges: Optional[np.ndarray], angles: Optional[np.ndarray],
                theta_ref: float, v_cruise: float, grasp: int = 0, hold: bool = False,
                override_dir: Optional[np.ndarray] = None, override_speed: float = 0.0,
                end_stop: bool = True) -> Command:
        p = self.p
        a_b = p.brake_frac * p.accel_per_cmd
        pos = pose.pos
        v_des = np.zeros(2)
        u = None

        if override_dir is not None:                                   # approche finale à vue
            u, vt = override_dir, override_speed
            v_des = u * vt
        elif self.route is not None and len(self.route) > 0 and not hold:
            proj, seg, off = self._project(pos)
            P = self.route.pts
            carrot = self._carrot(proj, seg, p.lookahead)
            to = carrot - pos
            nrm = float(np.linalg.norm(to))
            if nrm > 1e-6:
                u = to / nrm
            s_rem = self.remaining(pos)
            vt = v_cruise
            if end_stop:
                vt = min(vt, math.sqrt(2 * a_b * max(s_rem - 6.0, 0.0)))
            # virages à venir
            s_acc = float(np.linalg.norm(P[min(seg + 1, len(P) - 1)] - proj))
            for i in range(seg + 1, min(seg + 6, len(P) - 1)):
                d1, d2 = P[i] - P[i - 1], P[i + 1] - P[i]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-6 and n2 > 1e-6:
                    ang = math.acos(float(np.clip(d1 @ d2 / (n1 * n2), -1, 1)))
                    vc = v_cruise * (1.0 - ang / math.pi) ** 2
                    vt = min(vt, math.sqrt(vc * vc + 2 * a_b * s_acc))
                s_acc += float(n2)
                if s_acc > 400:
                    break
            # distance libre devant (lidar)
            if ranges is not None and u is not None:
                vt = min(vt, self._clearance_cap(pose, ranges, angles, u, a_b))
            v_des = (u * vt) if u is not None else np.zeros(2)

        # répulsion douce des obstacles proches
        if ranges is not None and angles is not None:
            v_des = v_des + self._repulsion(pose, ranges, angles)

        a_des = (v_des - v_est) / p.vel_time_const
        cmd_w = a_des / p.accel_per_cmd
        n = float(np.linalg.norm(cmd_w))
        if n > 1.0:
            cmd_w /= n
        self.last_cmd_mag = min(n, 1.0)
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        fwd = c * cmd_w[0] + s * cmd_w[1]
        lat = -s * cmd_w[0] + c * cmd_w[1]
        rot = float(np.clip(_wrap(theta_ref - pose.theta) * 8.0, -1.0, 1.0)) if abs(_wrap(theta_ref - pose.theta)) > 0.03 else 0.0
        return Command(float(fwd), float(lat), rot, grasp)

    def _clearance_cap(self, pose, ranges, angles, u, a_b) -> float:
        p = self.p
        rel = np.asarray(angles) + pose.theta - math.atan2(u[1], u[0])
        rel = (rel + np.pi) % (2 * np.pi) - np.pi
        ok = (np.abs(rel) < math.radians(40)) & np.isfinite(ranges)
        if not ok.any():
            return 99.0
        along = ranges[ok] * np.cos(rel[ok])
        # largeur du corridor : on ne s'inquiète que des rayons qui passent dans la largeur du drone
        lateral = np.abs(ranges[ok] * np.sin(rel[ok]))
        blocking = lateral < (p.robot_radius + 4.0)
        if not blocking.any():
            return 99.0
        clear = float(along[blocking].min()) - p.wall_margin
        return math.sqrt(2 * a_b * max(clear, 0.0))

    def _repulsion(self, pose, ranges, angles) -> np.ndarray:
        p = self.p
        r = np.asarray(ranges, float)
        near = np.isfinite(r) & (r < p.repulse_dist)
        if not near.any():
            return np.zeros(2)
        a = np.asarray(angles)[near] + pose.theta
        w = (1.0 - r[near] / p.repulse_dist) ** 2
        v = -(w[:, None] * np.stack((np.cos(a), np.sin(a)), axis=1)).sum(0) * (1.2 / max(near.sum() ** 0.5, 1.0))
        n = float(np.linalg.norm(v))
        return v if n < 0.8 else v * (0.8 / n)
