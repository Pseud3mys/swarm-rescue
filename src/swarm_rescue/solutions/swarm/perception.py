"""Perception : capteurs bruts + pose -> écritures dans le WorldModel (carte, objets).

Ne connaît ni la planification ni la radio. Tout est vectorisé numpy (pas de boucle sur les cellules).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

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


def observe_semantic(world: WorldModel, pose: Pose, semantic: Sequence, t: int) -> dict:
    """Détections sémantiques -> pistes d'objets. Renvoie ce qui est visible maintenant (pour les missions)."""
    seen = {"wounded": [], "rescue": []}
    pts = []
    for det in semantic:
        if det.kind not in seen or not np.isfinite(det.distance):
            continue
        ang = pose.theta + det.angle
        reach = det.distance + (10.0 if det.kind == "rescue" else 0.0)    # centre de secours : 10 px à l'intérieur du bord vu
        pos = pose.pos + reach * np.array((math.cos(ang), math.sin(ang)))
        pts.append((det.kind, pos, det))
    # un blessé est vu par plusieurs rayons : on regroupe à 25 px
    merged = []
    for kind, pos, det in pts:
        for g in merged:
            if g[0] == kind and np.linalg.norm(g[1] - pos) < 25.0:
                g[2].append(det); break
        else:
            merged.append([kind, pos, [det]])
    for kind, pos, dets in merged:
        d = min(dets, key=lambda x: x.distance)
        if kind == "wounded":
            carried = any(x.grasped for x in dets)
            st = ObjState.CARRIED if carried else ObjState.SEEN
            if not carried:                    # un blessé porté par un autre : son porteur gère la piste
                world.observe_object(ObjKind.WOUNDED, pos, t, st)
            seen["wounded"].append((pos, d.distance, st))
        elif kind == "rescue":
            world.observe_object(ObjKind.RESCUE_CENTER, pos, t)
            seen["rescue"].append((pos, d.distance))
    return seen
