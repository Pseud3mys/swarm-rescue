"""Perception : capteurs bruts + pose -> écritures dans le WorldModel (carte, objets).

Ne connaît ni la planification ni la radio. Tout est vectorisé numpy (pas de boucle sur les cellules).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from . import fusion
from .config import Params
from .schema import ObjKind, ObjState, Pose
from .world import WorldModel


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ScanIntegrator:
    """Lancer de rayons lidar -> preuves libre/occupé dans MA couche de carte.

    - une seule preuve "libre" par cellule et par scan, une seule "occupé" par cellule et par scan ;
    - la cellule touchée n'est jamais marquée libre par un rayon rasant ;
    - rien n'est écrit si la pose est trop incertaine (on ne grave pas du bruit dans la carte).
    """

    def __init__(self, world: WorldModel, params: Params):
        self.w, self.p = world, params
        step = world.map.res * 0.5
        self.samples = np.arange(0.0, params.lidar_max_range, step)
        m = world.map
        self._free = np.zeros((m.W, m.H), bool)
        self._hit = np.zeros((m.W, m.H), bool)

    def integrate(self, pose: Pose, ranges: np.ndarray, angles: np.ndarray,
                  ignore: Optional[np.ndarray] = None) -> int:
        """`ignore` : masque booléen des rayons à ne pas prendre comme obstacle (ex. autre drone)."""
        p, m = self.p, self.w.map
        if pose.sigma > p.max_pose_sigma_for_mapping:
            return 0
        r = np.asarray(ranges, float)
        ok = np.isfinite(r)
        hit = ok & (r < 0.985 * p.lidar_max_range) & (r > 3.0)
        if ignore is not None:
            hit &= ~ignore
        a = pose.theta + np.asarray(angles, float)
        c, s = np.cos(a), np.sin(a)
        reach = np.where(ok, np.where(hit, r - 0.7 * m.res, 0.985 * p.lidar_max_range), 0.0)
        if ignore is not None:       # derrière/sur un autre drone : on ne sait rien de plus que sa distance
            reach = np.where(ignore & ok, np.maximum(r - 25.0, 0.0), reach)
        S = self.samples[None, :]
        valid = S <= reach[:, None]
        xs = pose.x + c[:, None] * S
        ys = pose.y + s[:, None] * S
        ix, iy = m.world_to_cell(np.stack((xs[valid], ys[valid]), axis=1))
        free, hits = self._free, self._hit
        free[:] = False
        hits[:] = False
        free[ix, iy] = True
        if hit.any():
            hx, hy = m.world_to_cell(np.stack((pose.x + c[hit] * r[hit], pose.y + s[hit] * r[hit]), axis=1))
            hits[hx, hy] = True
        free &= ~hits
        fx, fy = np.nonzero(free)
        hx, hy = np.nonzero(hits)
        m.add_evidence(self.w.me, fx, fy, p.ev_free)
        m.add_evidence(self.w.me, hx, hy, p.ev_hit)
        return int(fx.size + hx.size)


def drone_ray_mask(ranges, angles, semantic: Iterable) -> Optional[np.ndarray]:
    """Rayons lidar qui touchent probablement un autre drone (vu par le capteur sémantique)."""
    mask = None
    for det in semantic:
        if det.kind != "drone" or det.distance < 1.0:
            continue
        half = math.asin(min(1.0, 20.0 / det.distance)) + 0.05
        near = (np.abs(_wrap(angles - det.angle)) < half) & (np.abs(ranges - det.distance) < 30.0)
        mask = near if mask is None else (mask | near)
    return mask


def observe_semantic(world: WorldModel, pose: Pose, semantic: Sequence, t: int,
                     params: Optional[Params] = None) -> dict:
    """Détections sémantiques -> pistes d'objets avec covariance. Renvoie ce qui est visible maintenant
    (pour les missions) : (position du centre, distance mesurée brute, état)."""
    p = params or Params()
    seen = {"wounded": [], "rescue": []}
    pts = []
    for det in semantic:
        if det.kind not in seen or not np.isfinite(det.distance):
            continue
        ang = pose.theta + det.angle
        if det.kind == "wounded":            # surface vue -> centre du blessé, biais du capteur retiré
            reach = det.distance - p.sem_dist_bias + p.wounded_radius
        else:                                # centre de secours : 10 px à l'intérieur du bord vu
            reach = det.distance + 10.0
        pos = pose.pos + reach * np.array((math.cos(ang), math.sin(ang)))
        pts.append((det.kind, pos, det, reach))
    # un objet est vu par plusieurs rayons : on regroupe à 25 px
    merged = []
    for kind, pos, det, reach in pts:
        for g in merged:
            if g[0] == kind and np.linalg.norm(g[1] - pos) < 25.0:
                g[2].append((det, pos, reach)); break
        else:
            merged.append([kind, pos, [(det, pos, reach)]])
    floor = 0.5 * float(pose.cov[0, 0] + pose.cov[1, 1])      # incertitude de MA pose : commune à mes observations
    for kind, _, items in merged:
        det, _, _ = min(items, key=lambda x: x[0].distance)
        pos = np.mean([q for _, q, _ in items], axis=0)       # rayons voisins : moyenne des centres
        reach = float(np.mean([r for _, _, r in items]))
        bearing = float(np.mean([x.angle for x, _, _ in items]))
        if kind == "wounded":
            carried = any(x.grasped for x, _, _ in items)
            st = ObjState.CARRIED if carried else ObjState.SEEN
            if not carried:                    # un blessé porté par un autre : son porteur gère la piste
                R = fusion.meas_point_cov(pose.cov, pose.theta, reach, bearing, p.sem_sigma_r,
                                          p.sem_sigma_b / math.sqrt(len(items)))
                world.observe_object(ObjKind.WOUNDED, pos, t, st, cov=R, floor_var=floor)
            seen["wounded"].append((pos, det.distance, st))
        elif kind == "rescue":
            # objet étendu (200 x 100 px) : chaque rayon voit un point différent du bord -> incertitude large
            R = fusion.meas_point_cov(pose.cov, pose.theta, reach, bearing, 30.0, p.sem_sigma_b) + 900.0 * np.eye(2)
            world.observe_object(ObjKind.RESCUE_CENTER, pos, t, cov=R, floor_var=floor)
            seen["rescue"].append((pos, det.distance))
    return seen
