"""Amers géométriques (coins de mur) : extraction dans le scan lidar + association + recalage de l'EKF.

Les coins sont des objets du WorldModel (ObjKind.CORNER) partagés par le gossip. Chaque drone publie SA contribution
pour un coin : son estimation fusionnée (Kalman sur toutes ses observations) avec sa covariance, dont le plancher est
l'incertitude de sa pose (le biais GPS est commun à toutes ses observations). La lecture (world.fuse_tracks) associe
les contributions des drones (porte de Mahalanobis) et les somme en forme information : l'estimation globale d'un
coin est la même chez tous les drones et converge quand plusieurs drones l'ont vu.

Règles :
  - recalage de l'EKF sur l'estimation globale MOINS ma contribution (mes observations partagent l'erreur de ma
    pose : me recaler dessus serait de l'auto-confirmation) ; sur la mienne seule uniquement sans GPS (elle date
    d'un moment où ma pose était bonne) ;
  - sans GPS (ou pose incertaine) : recalage dès qu'un coin connu est revu (délai `use_gap` par coin) ; avec GPS :
    seulement sur des coins bien convergés vus par d'autres (délai plus long `use_gap_gps`) : information
    indépendante, qui aide l'EKF à estimer le biais GPS ;
  - contribution seulement avec une pose sûre (GPS + sigma < sigma_create), observations espacées de `min_gap` pas
    (bruit GPS corrélé sur ~50 pas), et pas de nouvelle contribution sur un coin déjà convergé (volume radio borné) ;
  - association par Mahalanobis avec test de rapport (ambiguïté -> on ignore).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .estimator import PoseEKF
from . import fusion
from .schema import ObjKind
from .world import WorldModel


def extract_corners(ranges: np.ndarray, angles: np.ndarray, max_range: float,
                    k: int = 6, rms_max: float = 3.3, min_arc: float = 14.0,
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
                 min_gap: int = 25, use_gap: int = 12, use_gap_gps: int = 40,
                 sigma_use: float = 6.0, sigma_create: float = 4.5, converged_sigma: float = 2.0,
                 gps_use_sigma: float = 3.0, **_):
        self.w, self.ekf, self.p = world, ekf, params
        self.min_gap, self.use_gap, self.use_gap_gps = min_gap, use_gap, use_gap_gps
        self.sigma_use, self.sigma_create = sigma_use, sigma_create
        self.converged_sigma, self.gps_use_sigma = converged_sigma, gps_use_sigma
        self._last_use: dict = {}
        self._last_obs: dict = {}
        self._view, self._view_t = None, -10 ** 9
        self._cand: list = []          # coins vus une fois : publiés seulement s'ils sont revus (pas de bruit sur la radio)
        self.n_used = 0
        self.n_created = 0
        self.n_contrib = 0

    def _candidate(self, t: int, wpos, Rm, floor: float) -> None:
        """Un coin inconnu n'est publié qu'à sa 2e observation, au moins 6 pas plus tard (autre point de vue) : les
        détections isolées (bruit, objets) restent locales et sont oubliées au bout de 150 pas."""
        self._cand = [c for c in self._cand if t - c[2] < 150]
        for i, c in enumerate(self._cand):
            if np.linalg.norm(c[0] - wpos) < 8.0:
                if t - c[2] >= 6:
                    self.w.observe_object(ObjKind.CORNER, c[0], c[2], cov=c[1], floor_var=floor)
                    self.w.observe_object(ObjKind.CORNER, wpos, t, cov=Rm, floor_var=floor)
                    del self._cand[i]
                    self.n_created += 1
                return
        self._cand.append((wpos, Rm, t))

    def process(self, t: int, ranges, angles) -> int:
        """Renvoie le nombre de recalages appliqués à l'EKF."""
        p = self.p
        corners = extract_corners(ranges, angles, p.lidar_max_range)
        if len(corners) == 0:
            return 0
        if self._view is None or t - self._view_t >= 8:   # vue fusionnée des coins : reconstruite au plus tous les 8 pas
            view = self.w.objects_view(ObjKind.CORNER)
            self._view, self._view_t = (view, np.array([o.pos for o in view], float).reshape(-1, 2)), t
        view, P = self._view
        used = 0
        for c in corners:
            pose = self.ekf.pose()
            rng, brg = math.hypot(c[0], c[1]), math.atan2(c[1], c[0])
            ct, st = math.cos(pose.theta), math.sin(pose.theta)
            wpos = np.array((pose.x + ct * c[0] - st * c[1], pose.y + st * c[0] + ct * c[1]))
            Rm = fusion.meas_point_cov(pose.cov, pose.theta, rng, brg, p.landmark_sigma_r, p.landmark_sigma_b)
            floor = 0.5 * float(pose.cov[0, 0] + pose.cov[1, 1])
            pose_ok = self.ekf.gps_ok and pose.sigma < self.sigma_create
            # ---- association (Mahalanobis, test de rapport)
            near = np.nonzero(np.linalg.norm(P - wpos, axis=1) < 15.0)[0] if len(P) else np.zeros(0, int)
            scored = []
            for i in near.tolist():
                m2 = fusion.mahalanobis2(view[i].pos, fusion.cov_unpack(view[i].cov), wpos, Rm)
                if m2 < p.gate_chi2:
                    scored.append((m2, i))
            if not scored:
                if pose_ok:
                    self._candidate(t, wpos, Rm, floor)
                continue
            scored.sort()
            if len(scored) > 1 and scored[1][0] < max(4.0 * scored[0][0], scored[0][0] + 4.0):
                continue                                    # ambigu : on s'abstient
            lm = view[scored[0][1]]
            key = lm.sources[0]                             # identifiant stable du coin (plus petite clé du groupe)
            # ---- recalage
            if lm.others_pos is not None:
                est = (lm.others_pos, lm.others_cov)
            elif lm.mine is not None and not self.ekf.gps_ok:
                est = (lm.pos, lm.cov)
            else:
                est = None
            need_fix = (not self.ekf.gps_ok) or pose.sigma > self.sigma_use
            if est is not None:
                s_est = math.sqrt(0.5 * (est[1][0] + est[1][2]))
                gap = self.use_gap if need_fix else self.use_gap_gps
                allowed = need_fix or (lm.others_pos is not None and s_est < self.gps_use_sigma)
                if allowed and t - self._last_use.get(key, -10 ** 9) >= gap                         and self.ekf.update_landmark(est[0], rng, brg, cov_l=fusion.cov_unpack(est[1])):
                    self._last_use[key] = t
                    used += 1
            # ---- contribution (pose sûre, observations espacées, coin pas encore convergé)
            if pose_ok and t - self._last_obs.get(key, -10 ** 9) >= self.min_gap                     and (lm.mine is not None or lm.sigma > self.converged_sigma):
                pose = self.ekf.pose()
                ct, st = math.cos(pose.theta), math.sin(pose.theta)
                wpos = np.array((pose.x + ct * c[0] - st * c[1], pose.y + st * c[0] + ct * c[1]))
                Rm = fusion.meas_point_cov(pose.cov, pose.theta, rng, brg, p.landmark_sigma_r, p.landmark_sigma_b)
                self.w.observe_object(ObjKind.CORNER, wpos, t, cov=Rm, floor_var=floor)
                self._last_obs[key] = t
                self.n_contrib += 1
        self.n_used += used
        return used
