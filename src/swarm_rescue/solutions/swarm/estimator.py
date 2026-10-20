"""Estimation de pose : filtre de Kalman étendu à 8 états.

État  x = [px, py, th, vx, vy, bx, by, bth]
  p, th   : pose vraie du drone (repère monde)
  v       : déplacement du dernier pas (px/pas, monde)
  b, bth  : biais AR(1) du GPS / de la boussole  (mesure = vrai + biais, biais_k = phi * biais_{k-1} + bruit blanc)

Modèle de mouvement (mesuré sur le vrai simulateur, MapIntermediate01, 1200 pas) :
  - rotation = commande x ang_per_cmd : résidu 0,07°/pas contre 1,0°/pas pour l'odomètre ; l'incrément de cap est la
    fusion des deux ; odomètre seul en cas de choc (écart > 4°) et en portant (le couple du blessé écarte la rotation
    du modèle de façon systématique : fusionner biaisait le cap de 30° sans boussole) ;
  - vitesse : v_k = drag v_{k-1} + accel R(th) commande : résidu 0,06 px/pas contre 0,30 px/pas pour l'odomètre ;
    l'odomètre est une MESURE de v (dans le repère du drone au pas précédent). Rejoué sur des parties enregistrées,
    faire confiance au modèle dégrade pourtant la position (petit biais intégré ; en portant, le blessé traîné tire
    le drone de côté et corrompt le cap) : vel_model_sigma est réglé large (odomètre dominant). La vitesse transmise
    au contrôle vient d'un filtre séparé qui, lui, utilise le modèle.
  Sans GPS, c'est ce qui borne la dérive entre deux recalages (amers, carte).

Mises à jour possibles (toutes optionnelles, appelables dans n'importe quel ordre) :
  update_gps, update_compass, update_landmark (distance + relèvement vers un point connu du monde),
  update_pose / update_heading (recalage scan -> carte, cf. scanmatch.py).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .schema import Pose

PX, PY, TH, VX, VY, BX, BY, BT = range(8)
N = 8


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseEKF:
    def __init__(self, params: Optional[Params] = None):
        self.p = params or Params()
        self.x = np.zeros(N)
        self.P = np.diag([1e4, 1e4, 1.0, 0.01, 0.01, 4.0, 4.0, 0.05])   # incertain tant qu'on n'a pas de GPS
        self.initialized = False
        self.pos_init = False
        self.th_init = False
        self.gps_ok = False
        self.compass_ok = False
        self._rejects = 0
        self.v = np.zeros(2)            # vitesse pour le contrôle (filtre séparé : modèle + odomètre), px/pas
        self.pv = 0.05
        self.stats = {"gps": 0, "gps_rej": 0, "landmark": 0, "landmark_rej": 0, "odo_shock": 0, "rot_shock": 0}

    # ------------------------------------------------------------------ initialisation
    def _maybe_init(self, gps, compass) -> None:
        """Première mesure disponible : on repart d'une covariance propre (pas de termes croisés hérités)."""
        gps_new = gps is not None and not self.pos_init
        cmp_new = compass is not None and not self.th_init
        if not (gps_new or cmp_new):
            return
        if gps_new:
            self.x[[PX, PY]], self.x[[BX, BY]] = gps, 0.0
            for i in (PX, PY, BX, BY):
                self.P[i, :] = 0.0
                self.P[:, i] = 0.0
            self.P[PX, PX] = self.P[PY, PY] = 4.0
            self.P[BX, BX] = self.P[BY, BY] = 2.0
            self.pos_init = True
        if cmp_new:
            self.x[TH], self.x[BT] = compass, 0.0
            for i in (TH, BT):
                self.P[i, :] = 0.0
                self.P[:, i] = 0.0
            self.P[TH, TH] = 0.03 ** 2
            self.P[BT, BT] = 0.02 ** 2
            self.th_init = True
        self.initialized = self.pos_init

    # ------------------------------------------------------------------ prédiction
    def predict(self, odom, gps_avail: bool, compass_avail: bool, last_cmd=(0.0, 0.0), last_rot: float = 0.0,
                accel: Optional[float] = None, carrying: bool = False) -> None:
        """odom = (dist, alpha, dtheta) ou None. `last_cmd` = (forward, lateral) et `last_rot` = ordres du pas précédent
        (appliqués entre le pas précédent et celui-ci, que l'odomètre mesure)."""
        p = self.p
        x = self.x
        odom_ok = odom is not None and np.all(np.isfinite(odom))
        # ---- cap : incrément = fusion modèle (commande) + odomètre
        d_model = float(last_rot) * p.ang_per_cmd
        s_model = p.rot_model_sigma_carry if carrying else p.rot_model_sigma
        if odom_ok:
            d_odo, s_odo = float(odom[2]), p.odom_sigma_theta
            if carrying:                             # couple du blessé : écart SYSTÉMATIQUE au modèle -> odomètre seul
                dth, var_th = d_odo, s_odo ** 2
            elif abs(wrap(d_odo - d_model)) > math.radians(p.rot_shock_deg):   # choc
                dth, var_th = d_odo, s_odo ** 2
                self.stats["rot_shock"] += 1
            else:
                w_m, w_o = 1.0 / s_model ** 2, 1.0 / s_odo ** 2
                dth, var_th = (w_m * d_model + w_o * d_odo) / (w_m + w_o), 1.0 / (w_m + w_o)
        else:
            dth, var_th = d_model, s_model ** 2 + math.radians(0.5) ** 2
        th_old = x[TH]
        # ---- vitesse : v = drag v + accel R(th_old) commande ; position : p += v
        acc = p.accel_per_cmd if accel is None else accel
        fwd, lat = float(last_cmd[0]), float(last_cmd[1])
        c, s = math.cos(th_old), math.sin(th_old)
        a = acc * np.array((fwd * c - lat * s, fwd * s + lat * c))
        da = acc * np.array((-fwd * s - lat * c, fwd * c - lat * s))     # d a / d th
        v_new = p.drag * x[[VX, VY]] + a
        F = np.eye(N)
        F[VX, VX] = F[VY, VY] = p.drag
        F[VX, TH], F[VY, TH] = da
        F[PX, VX] = F[PY, VY] = p.drag
        F[PX, TH], F[PY, TH] = da
        x[[VX, VY]] = v_new
        x[[PX, PY]] += v_new
        x[TH] = th_old + dth
        Q = np.zeros((N, N))
        sa = (p.vel_model_sigma_carry if carrying else p.vel_model_sigma) if odom_ok else p.vel_model_sigma_blind
        Q[np.ix_([PX, PY, VX, VY], [PX, PY, VX, VY])] = sa ** 2 * np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], float)
        Q[PX, PX] += 0.01
        Q[PY, PY] += 0.01
        Q[TH, TH] = var_th + 1e-6
        if gps_avail:                                # le biais n'évolue que si le capteur fonctionne
            F[BX, BX] = F[BY, BY] = p.gps_phi
            Q[BX, BX] = Q[BY, BY] = p.gps_sigma ** 2 * (1 - p.gps_phi ** 2)
            x[[BX, BY]] *= p.gps_phi
        if compass_avail:
            F[BT, BT] = p.compass_phi
            Q[BT, BT] = p.compass_sigma ** 2 * (1 - p.compass_phi ** 2)
            x[BT] *= p.compass_phi
        self.P = F @ self.P @ F.T + Q
        if odom_ok:
            self._update_odom(float(odom[0]), float(odom[1]), dth)
        vm = None
        if odom_ok:
            phi = th_old + float(odom[1])
            vm = float(odom[0]) * np.array((math.cos(phi), math.sin(phi)))
        self._predict_velocity(a, vm)

    def _predict_velocity(self, a, vm) -> None:
        """Vitesse pour le contrôle : filtre scalaire (modèle d'accélération connu + mesure odométrique) ; 3x plus
        précise que l'odomètre seul (mesuré), sans effet sur la position de l'EKF."""
        p = self.p
        self.v = p.drag * self.v + a
        self.pv = self.pv * p.drag ** 2 + p.accel_model_trust ** 2
        if vm is not None:
            r = p.odom_sigma_dist ** 2 + (np.linalg.norm(vm) * p.odom_sigma_alpha) ** 2 + 1e-4
            k = self.pv / (self.pv + r)
            self.v = self.v + k * (vm - self.v)
            self.pv = (1 - k) * self.pv

    def _update_odom(self, d: float, alpha: float, dth: float) -> None:
        """Odomètre = mesure du déplacement du pas, dans le repère du drone au pas précédent (cap = th - dth)."""
        p = self.p
        z = np.array((d * math.cos(alpha), d * math.sin(alpha)))
        th0 = self.x[TH] - dth
        c, s = math.cos(th0), math.sin(th0)
        vx, vy = self.x[VX], self.x[VY]
        h = np.array((c * vx + s * vy, -s * vx + c * vy))
        H = np.zeros((2, N))
        H[0, VX], H[0, VY] = c, s
        H[1, VX], H[1, VY] = -s, c
        if p.odo_th_coupling:
            H[0, TH], H[1, TH] = -s * vx + c * vy, -c * vx - s * vy
        J = np.array([[math.cos(alpha), -d * math.sin(alpha)], [math.sin(alpha), d * math.cos(alpha)]])
        R = J @ np.diag([p.odom_sigma_dist ** 2, p.odom_sigma_alpha ** 2]) @ J.T + 1e-4 * np.eye(2)
        m2, y, S, Sinv = self._update(z, h, H, R)
        if Sinv is None:
            return
        if m2 > 13.8:                                # choc : le modèle de vitesse ne vaut plus, on suit l'odomètre
            self.stats["odo_shock"] += 1
            self.P[np.ix_([VX, VY], [VX, VY])] += p.shock_vel_var * np.eye(2)
            m2, y, S, Sinv = self._update(z, h, H, R)
            if Sinv is None:
                return
        self._apply(y, H, R, Sinv)

    # ------------------------------------------------------------------ mises à jour
    def _update(self, z, h, H, R, angle_rows=()):
        """Innovation : (distance de Mahalanobis², innovation, S, S^-1)."""
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
        """Forme de Joseph."""
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        I_KH = np.eye(N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_gps(self, gps, compass=None) -> bool:
        self._maybe_init(gps, compass)
        if gps is None or not np.all(np.isfinite(gps)):
            self.gps_ok = False
            return False
        self.gps_ok = True
        H = np.zeros((2, N))
        H[0, PX] = H[1, PY] = H[0, BX] = H[1, BY] = 1.0
        R = np.eye(2) * self.p.gps_meas_sigma ** 2
        h = self.x[[PX, PY]] + self.x[[BX, BY]]
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
        H = np.zeros((1, N))
        H[0, TH] = H[0, BT] = 1.0
        R = np.array([[self.p.compass_meas_sigma ** 2]])
        m2, y, S, Sinv = self._update([compass], [self.x[TH] + self.x[BT]], H, R, angle_rows=(0,))
        if m2 > 18.0:
            return False
        self._apply(y, H, R, Sinv)
        return True

    def update_landmark(self, landmark, rng: float, bearing: float, sigma_l: float = 0.0,
                        sigma_r: Optional[float] = None, sigma_b: Optional[float] = None,
                        cov_l: Optional[np.ndarray] = None, gate: Optional[float] = None) -> bool:
        """Mesure d'un point connu du monde `landmark` vu à distance `rng` et relèvement `bearing`
        (dans le repère du drone). Incertitude de l'amer : `cov_l` (2x2, monde : estimation fusionnée de l'essaim),
        sinon `sigma_l` (px, isotrope). Elle s'ajoute au bruit de mesure (propagée par le jacobien h / amer)."""
        if not self.initialized:
            return False
        p = self.p
        dx, dy = landmark[0] - self.x[PX], landmark[1] - self.x[PY]
        rho = math.hypot(dx, dy)
        if rho < 5.0:
            return False
        H = np.zeros((2, N))
        H[0, PX], H[0, PY] = -dx / rho, -dy / rho
        H[1, PX], H[1, PY], H[1, TH] = dy / rho ** 2, -dx / rho ** 2, -1.0
        sr = (sigma_r if sigma_r is not None else p.landmark_sigma_r)
        sb = (sigma_b if sigma_b is not None else p.landmark_sigma_b)
        if cov_l is not None:
            J = -H[:, [PX, PY]]                              # dh / d(amer) = - dh / d(position du drone)
            R = np.diag([sr ** 2, sb ** 2]) + J @ np.asarray(cov_l, float) @ J.T
        else:
            R = np.diag([sr ** 2 + sigma_l ** 2, sb ** 2 + (sigma_l / rho) ** 2])
        h = [rho, math.atan2(dy, dx) - self.x[TH]]
        m2, y, S, Sinv = self._update([rng, bearing], h, H, R, angle_rows=(1,))
        if m2 > (gate if gate is not None else p.gate_chi2):
            self.stats["landmark_rej"] += 1
            return False
        self._apply(y, H, R, Sinv)
        self.stats["landmark"] += 1
        return True

    def update_pose(self, z, R) -> bool:
        """Mesure directe de (x, y, cap) avec covariance R (3x3) : recalage scan -> carte (scanmatch.py)."""
        if not self.initialized:
            return False
        H = np.zeros((3, N))
        H[0, PX] = H[1, PY] = H[2, TH] = 1.0
        R = np.asarray(R, float)
        m2, y, S, Sinv = self._update(z, self.x[[PX, PY, TH]], H, R, angle_rows=(2,))
        if Sinv is None or m2 > 16.3:                     # chi2(3 ddl) 99,9 %
            self.stats["scan_rej"] = self.stats.get("scan_rej", 0) + 1
            return False
        self._apply(y, H, R, Sinv)
        self.stats["scan"] = self.stats.get("scan", 0) + 1
        return True

    def update_heading(self, theta: float, var: float) -> bool:
        """Mesure du cap seul (recalage sur la carte complète, cf. scanmatch.py)."""
        if not self.initialized:
            return False
        H = np.zeros((1, N))
        H[0, TH] = 1.0
        R = np.array([[var]])
        m2, y, S, Sinv = self._update([theta], [self.x[TH]], H, R, angle_rows=(0,))
        if Sinv is None or m2 > 10.8:                     # chi2(1 ddl) 99,9 %
            self.stats["head_rej"] = self.stats.get("head_rej", 0) + 1
            return False
        self._apply(y, H, R, Sinv)
        self.stats["head"] = self.stats.get("head", 0) + 1
        return True

    # ------------------------------------------------------------------ sortie
    def pose(self) -> Pose:
        cov = self.P[:3, :3].copy()
        return Pose(float(self.x[PX]), float(self.x[PY]), wrap(float(self.x[TH])),
                    float(self.v[0]), float(self.v[1]), cov, self.gps_ok)
