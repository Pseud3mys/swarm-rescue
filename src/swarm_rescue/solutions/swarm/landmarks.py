"""Amers géométriques (coins de mur) : extraction dans le scan lidar + association + recalage de l'EKF.

Les coins sont stockés comme des objets du WorldModel (ObjKind.CORNER) : ils se partagent donc entre
drones par le même gossip que le reste, et tous les drones se recalent sur les MÊMES points du monde.

Précautions contre l'auto-confirmation :
  - un amer ne sert au recalage qu'après `min_obs` observations espacées d'au moins `min_gap` pas
    (le GPS est corrélé sur ~50 pas : deux observations rapprochées ne sont pas indépendantes) ;
  - incertitude d'un amer = f(nombre d'observations) ; elle est ajoutée au bruit de mesure de l'EKF ;
  - association par plus proche voisin avec test de rapport (ambiguïté -> on ignore) ;
  - on ne crée d'amer que si la pose est suffisamment sûre.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .estimator import PoseEKF
from .schema import ObjKind
from .world import WorldModel


def extract_corners(ranges: np.ndarray, angles: np.ndarray, max_range: float,
                    k: int = 6, rms_max: float = 2.6, min_arc: float = 14.0,
                    r_min: float = 30.0, r_max: float = 120.0) -> np.ndarray:
    """Coins du scan, dans le repère du drone. Renvoie un tableau (M, 2).

    Un coin = deux segments droits (fenêtres de k+1 points de part et d'autre) quasi perpendiculaires,
    contigus (pas de saut de profondeur) ; position = intersection des deux droites.
    """
    r = np.asarray(ranges, float)[:-1]          # le dernier rayon (+pi) double le premier (-pi)
    a = np.asarray(angles, float)[:-1]
    good = np.isfinite(r) & (r > r_min * 0.5) & (r < 0.97 * max_range)
    x, y = r * np.cos(a), r * np.sin(a)
    out = []
    n = len(r)
    i = 0
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and good[j + 1] and abs(r[j + 1] - r[j]) < 10.0 + 0.06 * r[j]:
            j += 1
        if j - i + 1 >= 2 * k + 1:
            out.extend(_corners_in_segment(x[i:j + 1], y[i:j + 1], k, rms_max, min_arc, r_min, r_max))
        i = j + 1
    return np.array(out).reshape(-1, 2)


def _window_lines(x, y, k):
    """Droite ajustée sur chaque fenêtre glissante de k+1 points : (centre, direction, rms)."""
    w = k + 1
    ker = np.ones(w)
    sx, sy = np.convolve(x, ker, "valid"), np.convolve(y, ker, "valid")
    sxx, syy, sxy = np.convolve(x * x, ker, "valid"), np.convolve(y * y, ker, "valid"), np.convolve(x * y, ker, "valid")
    mx, my = sx / w, sy / w
    cxx, cyy, cxy = sxx / w - mx * mx, syy / w - my * my, sxy / w - mx * my
    th = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    lam_min = 0.5 * (cxx + cyy) - np.sqrt(0.25 * (cxx - cyy) ** 2 + cxy ** 2)
    return mx, my, np.cos(th), np.sin(th), np.sqrt(np.maximum(lam_min, 0.0))


def _corners_in_segment(x, y, k, rms_max, min_arc, r_min, r_max):
    n = len(x)
    mx, my, ux, uy, rms = _window_lines(x, y, k)      # fenêtre m couvre les points m..m+k
    idx = np.arange(k, n - k)                          # point candidat i : gauche = fenêtre i-k, droite = fenêtre i
    L, R = idx - k, idx
    arc_l = np.hypot(x[idx] - x[idx - k], y[idx] - y[idx - k])
    arc_r = np.hypot(x[idx + k] - x[idx], y[idx + k] - y[idx])
    cross = ux[L] * uy[R] - uy[L] * ux[R]
    ok = (rms[L] < rms_max) & (rms[R] < rms_max) & (arc_l > min_arc) & (arc_r > min_arc) & (np.abs(cross) > 0.8)
    cand = np.nonzero(ok)[0]
    if cand.size == 0:
        return []
    score = rms[L][cand] + rms[R][cand]
    order = np.argsort(score)
    taken = np.zeros(len(idx), bool)
    res = []
    for oc in order:
        c = cand[oc]
        if taken[max(c - k, 0):c + k + 1].any():
            continue
        taken[c] = True
        l, rr = L[c], R[c]
        # intersection des deux droites : m1 + s u1 = m2 + t u2
        det = ux[l] * (-uy[rr]) - uy[l] * (-ux[rr])
        if abs(det) < 1e-6:
            continue
        bx, by = mx[rr] - mx[l], my[rr] - my[l]
        s = (bx * (-uy[rr]) - by * (-ux[rr])) / det
        px, py = mx[l] + s * ux[l], my[l] + s * uy[l]
        i = idx[c]
        if math.hypot(px - x[i], py - y[i]) > 9.0:
            continue
        rho = math.hypot(px, py)
        if r_min <= rho <= r_max:
            res.append((px, py))
    return res


class LandmarkTracker:
    def __init__(self, world: WorldModel, ekf: PoseEKF, params: Params,
                 min_obs: int = 2, min_gap: int = 25, max_obs: int = 14,
                 use_gap: int = 12, sigma_use: float = 6.0, sigma_create: float = 4.5):
        """Recalage seulement si la pose est mauvaise (GPS coupé ou incertitude > sigma_use) : avec un bon GPS, les amers
        (dont l'erreur est commune à toutes leurs observations) rendraient le filtre trop confiant. Un amer n'est
        réutilisé qu'après `use_gap` pas. On ne crée / affine des amers que lorsque la pose est sûre (sigma_create)."""
        self.w, self.ekf, self.p = world, ekf, params
        self.min_obs, self.min_gap, self.max_obs = min_obs, min_gap, max_obs
        self.use_gap, self.sigma_use, self.sigma_create = use_gap, sigma_use, sigma_create
        self._last_use: dict = {}
        self._ver = -1
        self._pos = np.zeros((0, 2))
        self._rev = np.zeros(0, int)
        self._tseen = np.zeros(0, int)
        self.n_used = 0
        self.n_created = 0

    def _db(self):
        if self._ver != self.w.version:
            tr = [o for o in self.w.objects.values() if o.kind == ObjKind.CORNER]
            self._pos = np.array([o.pos for o in tr]).reshape(-1, 2)
            self._rev = np.array([o.rev for o in tr], int)
            self._tseen = np.array([o.t_seen for o in tr], int)
            self._ver = self.w.version
        return self._pos, self._rev, self._tseen

    @staticmethod
    def sigma_l(rev: int) -> float:
        return 2.5 + 6.0 / math.sqrt(max(rev, 1))

    def process(self, t: int, ranges, angles) -> int:
        """Renvoie le nombre de recalages appliqués à l'EKF."""
        p = self.p
        corners = extract_corners(ranges, angles, p.lidar_max_range)
        used = 0
        for c in corners:
            pose = self.ekf.pose()
            ct, st = math.cos(pose.theta), math.sin(pose.theta)
            wpos = np.array((pose.x + ct * c[0] - st * c[1], pose.y + st * c[0] + ct * c[1]))
            pos, rev, tseen = self._db()
            gate = min(25.0, 3.0 * math.sqrt(pose.sigma ** 2 + 4.0) + 4.0)
            if len(pos):
                d = np.linalg.norm(pos - wpos, axis=1)
                near = np.nonzero(d < gate)[0]
            else:
                near = np.zeros(0, int)
            if near.size == 0:
                if pose.sigma < self.sigma_create:
                    self.w.observe_object(ObjKind.CORNER, wpos, t)
                    self.n_created += 1
                continue
            near = near[np.argsort(d[near])]
            i0 = near[0]
            same = near[np.linalg.norm(pos[near] - pos[i0], axis=1) < 14.0]
            others = np.setdiff1d(near, same)
            if others.size and d[others].min() < max(1.7 * d[i0], d[i0] + 6.0):
                continue                                    # ambigu : on s'abstient
            best = same[np.argmax(rev[same])]
            lm = pos[best]
            rng, brg = math.hypot(c[0], c[1]), math.atan2(c[1], c[0])
            key = int(best)
            need_fix = (not self.ekf.gps_ok) or pose.sigma > self.sigma_use
            if (need_fix and rev[best] >= self.min_obs and t - self._last_use.get(key, -10 ** 9) >= self.use_gap
                    and self.ekf.update_landmark(lm, rng, brg, self.sigma_l(int(rev[best])))):
                self._last_use[key] = t
                used += 1
                pose = self.ekf.pose()
                ct, st = math.cos(pose.theta), math.sin(pose.theta)
                wpos = np.array((pose.x + ct * c[0] - st * c[1], pose.y + st * c[0] + ct * c[1]))
            # affinage de l'amer (borné pour ne pas inonder la radio)
            if rev[best] < self.max_obs and t - tseen[best] >= self.min_gap and pose.sigma < self.sigma_create and self.ekf.gps_ok:
                self.w.observe_object(ObjKind.CORNER, wpos, t)
        self.n_used += used
        return used
