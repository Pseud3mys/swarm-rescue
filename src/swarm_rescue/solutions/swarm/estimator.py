"""Estimation de pose : filtre de Kalman étendu à 6 états + filtre de vitesse.

État  x = [px, py, th, bx, by, bth]
  p, th   : pose vraie du drone (repère monde)
  b, bth  : biais AR(1) du GPS / de la boussole  (mesure = vrai + biais, biais_k = phi * biais_{k-1} + bruit blanc)

Pourquoi ça marche mieux qu'un simple lissage du GPS :
  - le bruit GPS est corrélé sur ~50 pas (phi = 0.98) : moyenner ne le supprime pas.
  - l'odomètre (bruit blanc, précis à court terme) donne le déplacement vrai ; la différence
    "GPS - trajectoire odométrique" révèle le biais, qui est alors retranché.
  - hors zone GPS, le biais est gelé (le simulateur ne fait pas évoluer le bruit d'un capteur coupé)
    et la pose dérive seulement avec l'odomètre -> les amers (coins) prennent le relais.

Mises à jour possibles (toutes optionnelles, appelables dans n'importe quel ordre) :
  update_gps, update_compass, update_landmark (distance + relèvement vers un point connu du monde).
`update_landmark` est générique : coin de mur, centre de secours, ou toute cible dont on connaît la position.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .schema import Pose


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseEKF:
    def __init__(self, params: Optional[Params] = None):
        self.p = params or Params()
        self.x = np.zeros(6)
        self.P = np.diag([1e4, 1e4, 1.0, 4.0, 4.0, 0.05])   # incertain tant qu'on n'a pas de GPS
        self.initialized = False
        self.pos_init = False
        self.th_init = False
        self.gps_ok = False
        self.compass_ok = False
        self._rejects = 0
        # filtre de vitesse (monde), px/pas
        self.v = np.zeros(2)
        self.pv = 0.05
        self.stats = {"gps": 0, "gps_rej": 0, "landmark": 0, "landmark_rej": 0}

    # ------------------------------------------------------------------ initialisation
    def _maybe_init(self, gps, compass) -> None:
        """Première mesure disponible : on repart d'une covariance propre (pas de termes croisés hérités)."""
        gps_new = gps is not None and not self.pos_init
        cmp_new = compass is not None and not self.th_init
        if not (gps_new or cmp_new):
            return
        if gps_new:
            self.x[0:2], self.x[3:5] = gps, 0.0
            self.P[0:2, :] = 0.0; self.P[:, 0:2] = 0.0; self.P[3:5, :] = 0.0; self.P[:, 3:5] = 0.0
            self.P[0, 0] = self.P[1, 1] = 4.0
            self.P[3, 3] = self.P[4, 4] = 2.0
            self.pos_init = True
        if cmp_new:
            self.x[2], self.x[5] = compass, 0.0
            self.P[2, :] = 0.0; self.P[:, 2] = 0.0; self.P[5, :] = 0.0; self.P[:, 5] = 0.0
            self.P[2, 2] = 0.03 ** 2
            self.P[5, 5] = 0.02 ** 2
            self.th_init = True
        self.initialized = self.pos_init

    # ------------------------------------------------------------------ prédiction
    def predict(self, odom, gps_avail: bool, compass_avail: bool, last_cmd=(0.0, 0.0), last_rot: float = 0.0) -> None:
        """odom = (dist, alpha, dtheta) ou None. `last_cmd` = (forward, lateral) et `last_rot` = ordres du pas précédent.

        Cap : on intègre toujours la rotation mesurée par l'odomètre (bruit blanc 1°/pas, corrigé par la boussole).
        Le modèle "ordre de rotation x gain" ne voit pas les rotations subies (chocs, couple du blessé porté) :
        l'ignorer faisait dériver le cap de plusieurs dizaines de degrés pendant un portage."""
        p = self.p
        x, P = self.x, self.P
        F = np.eye(6)
        Q = np.zeros((6, 6))
        if odom is not None and np.all(np.isfinite(odom)):
            d, alpha, dth = float(odom[0]), float(odom[1]), float(odom[2])
            phi = x[2] + alpha
            c, s = math.cos(phi), math.sin(phi)
            x[0] += d * c
            x[1] += d * s
            F[0, 2] = -d * s
            F[1, 2] = d * c
            x[2] += dth
            # bruit (dist, alpha, theta) -> (px, py, th)
            G = np.array([[c, -d * s, 0.0], [s, d * c, 0.0], [0.0, 0.0, 1.0]])
            S = np.diag([p.odom_sigma_dist ** 2, p.odom_sigma_alpha ** 2, p.odom_sigma_theta ** 2])
            Q[:3, :3] = G @ S @ G.T
            Q[2, :] = 0.0; Q[:, 2] = 0.0
            Q[2, 2] = p.odom_sigma_theta ** 2
            vm = np.array([d * c, d * s])
        else:                                       # odomètre coupé : on inflate l'incertitude
            Q[:2, :2] = np.eye(2) * 0.5
            Q[2, 2] = (math.radians(1.0)) ** 2
            vm = None
        Q[:3, :3] += np.diag([0.01, 0.01, 1e-6])    # collisions, glissements
        if gps_avail:                                # le biais n'évolue que si le capteur fonctionne
            F[3, 3] = F[4, 4] = p.gps_phi
            Q[3, 3] = Q[4, 4] = p.gps_sigma ** 2 * (1 - p.gps_phi ** 2)
            x[3:5] *= p.gps_phi
        if compass_avail:
            F[5, 5] = p.compass_phi
            Q[5, 5] = p.compass_sigma ** 2 * (1 - p.compass_phi ** 2)
            x[5] *= p.compass_phi
        self.P = F @ P @ F.T + Q
        self._predict_velocity(last_cmd, vm)

    def _predict_velocity(self, cmd, vm) -> None:
        """Filtre scalaire (commun aux deux axes) : modèle d'accélération connu + mesure odométrique."""
        p = self.p
        th = self.x[2]
        fwd, lat = cmd
        a = p.accel_per_cmd * np.array([fwd * math.cos(th) - lat * math.sin(th),
                                        fwd * math.sin(th) + lat * math.cos(th)])
        self.v = p.drag * self.v + a
        self.pv = self.pv * p.drag ** 2 + p.accel_model_trust ** 2
        if vm is not None:
            r = p.odom_sigma_dist ** 2 + (np.linalg.norm(vm) * p.odom_sigma_alpha) ** 2 + 1e-4
            k = self.pv / (self.pv + r)
            self.v = self.v + k * (vm - self.v)
            self.pv = (1 - k) * self.pv

    # ------------------------------------------------------------------ mises à jour
    def _update(self, z, h, H, R, angle_rows=()) -> float:
        """Mise à jour EKF générique avec forme de Joseph. Renvoie la distance de Mahalanobis² (innovation)."""
        y = np.asarray(z, dtype=float) - np.asarray(h, dtype=float)
        for i in angle_rows:
            y[i] = wrap(y[i])
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf"), y, S, None
        m2 = float(y @ Sinv @ y)
        return m2, y, S, Sinv

    def _apply(self, y, H, R, Sinv) -> None:
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_gps(self, gps, compass=None) -> bool:
        self._maybe_init(gps, compass)
        if gps is None or not np.all(np.isfinite(gps)):
            self.gps_ok = False
            return False
        self.gps_ok = True
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = H[0, 3] = H[1, 4] = 1.0
        R = np.eye(2) * self.p.gps_meas_sigma ** 2
        h = self.x[0:2] + self.x[3:5]
        m2, y, S, Sinv = self._update(gps, h, H, R)
        if Sinv is None:
            return False
        if m2 > 18.4 and self._rejects < 20:      # chi2(2 ddl) 99.99 % ; on réaccepte après 20 rejets (filtre perdu)
            self._rejects += 1
            self.stats["gps_rej"] += 1
            return False
        self._rejects = 0
        self._apply(y, H, R, Sinv)
        self.stats["gps"] += 1
        return True

    def update_compass(self, compass) -> bool:
        if compass is None or not np.isfinite(compass):
            self.compass_ok = False
            return False
        self.compass_ok = True
        self._maybe_init(None, compass)
        H = np.zeros((1, 6))
        H[0, 2] = H[0, 5] = 1.0
        R = np.array([[self.p.compass_meas_sigma ** 2]])
        m2, y, S, Sinv = self._update([compass], [self.x[2] + self.x[5]], H, R, angle_rows=(0,))
        if m2 > 18.0:
            return False
        self._apply(y, H, R, Sinv)
        return True

    def update_landmark(self, landmark, rng: float, bearing: float, sigma_l: float = 0.0,
                        sigma_r: Optional[float] = None, sigma_b: Optional[float] = None) -> bool:
        """Mesure d'un point connu du monde `landmark` vu à distance `rng` et relèvement `bearing`
        (dans le repère du drone). `sigma_l` = incertitude sur la position de l'amer (px)."""
        if not self.initialized:
            return False
        p = self.p
        dx, dy = landmark[0] - self.x[0], landmark[1] - self.x[1]
        rho = math.hypot(dx, dy)
        if rho < 5.0:
            return False
        H = np.zeros((2, 6))
        H[0, 0], H[0, 1] = -dx / rho, -dy / rho
        H[1, 0], H[1, 1], H[1, 2] = dy / rho ** 2, -dx / rho ** 2, -1.0
        sr = (sigma_r if sigma_r is not None else p.landmark_sigma_r)
        sb = (sigma_b if sigma_b is not None else p.landmark_sigma_b)
        R = np.diag([sr ** 2 + sigma_l ** 2, sb ** 2 + (sigma_l / rho) ** 2])
        h = [rho, math.atan2(dy, dx) - self.x[2]]
        m2, y, S, Sinv = self._update([rng, bearing], h, H, R, angle_rows=(1,))
        if m2 > p.gate_chi2:
            self.stats["landmark_rej"] += 1
            return False
        self._apply(y, H, R, Sinv)
        self.stats["landmark"] += 1
        return True

    # ------------------------------------------------------------------ sortie
    def pose(self) -> Pose:
        cov = self.P[:3, :3].copy()
        return Pose(float(self.x[0]), float(self.x[1]), wrap(float(self.x[2])),
                    float(self.v[0]), float(self.v[1]), cov, self.gps_ok)
