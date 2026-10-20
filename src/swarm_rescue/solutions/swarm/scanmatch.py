"""Recalage scan -> carte : mesure (x, y, cap) de l'EKF à partir du lidar et de la carte fusionnée.

Pourquoi : le cap est la principale source d'erreur de la carte (mesuré sur MapMedium02 : 2-3° médian, 6-8° au 95e
centile AVEC boussole ; à 250 px, 5° = 22 px : les rayons libres traversent les vrais murs et les effacent). Rien
d'autre ne mesure le cap de façon indépendante : la boussole a un biais AR(1) (~4°, corrélé sur 50 pas), l'odomètre
ajoute 1° de bruit blanc PAR PAS (marche aléatoire de 30° en 1000 pas sans boussole).

Méthode (type Hector SLAM) : champ d'occupation lissé (gaussienne, deux échelles), pondéré par les preuves, puis
Gauss-Newton amorti sur (x, y, cap) pour amener les impacts lidar au centre des murs connus. Un champ de distance aux
cellules « occupé » seuillées avait un plateau : les couches des drones, légèrement décalées, donnent des murs épais de
2-3 cellules où tout décalage est équivalent (8-12 px de jeu).

Deux références (même principe que les coins, cf. landmarks.py), choisies par brain.py :
  - murs des AUTRES drones seulement -> pose complète, tous les sm_pose_period pas, avec ou sans GPS : information
    indépendante (les biais GPS / boussole des drones et des instants différents s'y moyennent : gain avec le nombre
    de drones) ;
  - carte complète, ma couche comprise -> cap seul, et seulement sans GPS quand le cap est vraiment incertain
    (σθ > sm_own_min_sth : en portant, odomètre seul). Jamais en position : se recaler sur ses propres scans récents
    s'auto-confirme et fait glisser la pose. Pas non plus quand le modèle de mouvement tient le cap (~0,1°/pas) :
    l'orientation de ma carte garde le biais de boussole de mon passé et le recalage dégradait le cap (mesuré).
    Boussole présente : plancher sm_own_floor_th, elle garde la main sur le long terme.

Mesure renvoyée : covariance = variance des résidus x H^-1, gonflée (points corrélés) et avec planchers ; une
direction dégénérée (couloir) reçoit une grande variance et l'EKF n'y corrige presque rien.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .config import Params
from .schema import Pose
from .world import WorldModel


class ScanMatcher:
    def __init__(self, world: WorldModel, params: Params):
        self.w, self.p = world, params
        self._all = self._oth = None
        self._t = -10 ** 9
        self.stats = {"pose": 0, "heading": 0, "own": 0, "few": 0, "bad": 0}

    # ------------------------------------------------------------------ champs
    def _build(self, L: np.ndarray):
        """Champs lissés (grossier, fin) + gradients, à partir d'une somme de log-odds."""
        p, m = self.p, self.w.map
        o = np.clip(L.astype(np.float32), 0.0, p.sm_occ_sat) / p.sm_occ_sat
        if float(o.sum()) < 30.0:
            return None
        out = []
        for s in (2.0, 1.0):                                  # cellules
            k = int(2 * math.ceil(3 * s) + 1)
            M = cv2.GaussianBlur(o, (k, k), s, borderType=cv2.BORDER_CONSTANT)
            M = np.minimum(M * (math.sqrt(2 * math.pi) * s), 1.0)   # mur fin bien établi -> pic ~1
            gx, gy = np.gradient(M, m.res)
            out.append((M, gx.astype(np.float32), gy.astype(np.float32)))
        return out

    def _fields(self, t: int, use_own: bool) -> None:
        p, m = self.p, self.w.map
        if t - self._t < p.sm_field_period and (self._all is not None or not use_own):
            return
        self._t = t
        self._all = self._build(m.view()) if use_own else None
        others = [L for o, L in m.layers.items() if o != self.w.me]
        if others:
            acc = np.zeros((m.W, m.H), np.int16)
            for L in others:
                acc += L
            self._oth = self._build(acc)
        else:
            self._oth = None

    def _sample(self, F, q: np.ndarray):
        m = self.w.map
        u = (q[:, 0] + m.half[0]) / m.res - 0.5
        v = (q[:, 1] + m.half[1]) / m.res - 0.5
        i0 = np.clip(np.floor(u).astype(int), 0, m.W - 2)
        j0 = np.clip(np.floor(v).astype(int), 0, m.H - 2)
        a = np.clip(u - i0, 0.0, 1.0)
        b = np.clip(v - j0, 0.0, 1.0)
        return [(1 - a) * (1 - b) * G[i0, j0] + a * (1 - b) * G[i0 + 1, j0]
                + (1 - a) * b * G[i0, j0 + 1] + a * b * G[i0 + 1, j0 + 1] for G in F]

    # ------------------------------------------------------------------ recalage
    def match(self, t: int, pose: Pose, ranges, angles, ignore=None, compass_ok: bool = True,
              want_pose: bool = True, use_own: bool = True) -> Optional[tuple]:
        """("pose", z (x, y, cap), R 3x3) sur les murs des autres ; ("heading", cap, variance) sur la carte complète ;
        None si rien de fiable."""
        if ranges is None:
            return None
        self._fields(t, use_own)
        tries = []
        if self._oth is not None:
            tries.append(("pose" if want_pose else "heading", self._oth))
        if use_own and self._all is not None:
            tries.append(("own", self._all))
        for kind, F in tries:
            res = self._match(F, pose, ranges, angles, ignore)
            if res is None:
                continue
            z, R = res
            self.stats[kind] += 1
            if kind == "pose":
                return kind, z, R
            var = float(R[2, 2])
            if kind == "own" and compass_ok:                 # la boussole garde la main sur le long terme
                var = max(var, math.radians(self.p.sm_own_floor_th) ** 2)
            return "heading", float(z[2]), var
        return None

    def _match(self, F, pose: Pose, ranges, angles, ignore=None) -> Optional[tuple]:
        p = self.p
        r = np.asarray(ranges, float)[:-1]                   # le dernier rayon (+pi) double le premier
        a = np.asarray(angles, float)[:-1]
        ok = np.isfinite(r) & (r > 20.0) & (r < p.lidar_max_range - 3.0 * p.lidar_sigma)
        if ignore is not None:
            ok &= ~np.asarray(ignore, bool)[:-1]
        if ok.sum() < p.sm_min_points:
            self.stats["few"] += 1
            return None
        bx, by = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        xi = np.array((pose.x, pose.y, pose.theta), float)

        def project(xi, bx, by):
            c, s = math.cos(xi[2]), math.sin(xi[2])
            return np.stack((xi[0] + c * bx - s * by, xi[1] + s * bx + c * by), axis=1), c, s

        q, _, _ = project(xi, bx, by)
        sel = self._sample(F[0][:1], q)[0] > 0.05           # seulement les impacts près d'un mur déjà connu
        if sel.sum() < p.sm_min_points:
            self.stats["few"] += 1
            return None
        bx, by = bx[sel], by[sel]
        H = None
        for lvl, iters in ((0, 5), (1, 5)):                  # grossier puis fin
            for _ in range(iters):
                q, c, s = project(xi, bx, by)
                M, gx, gy = self._sample(F[lvl], q)
                res = 1.0 - M
                dqt_x, dqt_y = -s * bx - c * by, c * bx - s * by
                J = -np.stack((gx, gy, gx * dqt_x + gy * dqt_y), axis=1)
                H = J.T @ J
                g = J.T @ res
                Hd = H + 1e-2 * np.diag(np.maximum(np.diag(H), 1e-9)) + np.diag((1e-6, 1e-6, 1e-3))
                try:
                    step = -np.linalg.solve(Hd, g)
                except np.linalg.LinAlgError:
                    return None
                step[:2] = np.clip(step[:2], -4.0, 4.0)
                step[2] = float(np.clip(step[2], -0.03, 0.03))
                xi = xi + step
                if abs(step[2]) < 1e-4 and abs(step[0]) + abs(step[1]) < 0.05:
                    break
        q, _, _ = project(xi, bx, by)
        M = self._sample(F[1][:1], q)[0]
        inl = M > 0.3
        n = int(inl.sum())
        dth = (xi[2] - pose.theta + math.pi) % (2 * math.pi) - math.pi
        if n < p.sm_min_points or float(np.mean(M)) < p.sm_min_score or abs(dth) > math.radians(p.sm_max_dth) \
                or math.hypot(xi[0] - pose.x, xi[1] - pose.y) > p.sm_max_dxy:
            self.stats["bad"] += 1
            return None
        # covariance : variance des résidus x H^-1, gonflée (points corrélés : n / sm_n_eff), planchers ; directions
        # dégénérées (couloir) -> grande variance, l'EKF n'y corrige presque rien
        sig2 = max(float(np.mean((1.0 - M) ** 2)), p.sm_res_floor ** 2) * max(len(M) / p.sm_n_eff, 1.0)
        try:
            R = sig2 * np.linalg.inv(H + np.diag((1e-6, 1e-6, 1e-4)))
        except np.linalg.LinAlgError:
            return None
        R = 0.5 * (R + R.T) + np.diag((p.sm_floor_xy ** 2, p.sm_floor_xy ** 2, math.radians(p.sm_floor_th) ** 2))
        return np.array((xi[0], xi[1], pose.theta + dth)), R
