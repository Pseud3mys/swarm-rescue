"""Fusion de perceptions avec incertitude (gaussiennes 2D) : la brique commune pour partager des ESTIMATIONS
fusionnées entre drones plutôt que des mesures brutes.

Convention : un point est une gaussienne (mu (2,), P (2, 2)) en "forme moment" pour le stockage et la radio
(5 nombres), convertie en "forme information" (Y = P^-1, y = P^-1 mu) pour fusionner :
  - fusion de sources INDÉPENDANTES = somme des informations (exacte, commutative, associative) ;
  - retirer sa propre contribution d'une estimation globale = soustraction (évite de se recaler sur soi-même) ;
  - sources possiblement corrélées (ex. info déjà relayée) = intersection de covariances (prudente, jamais
    trop confiante).
Les observations successives d'UN drone partagent le biais de sa pose (GPS AR(1)) : elles ne sont pas
indépendantes. `kalman_update` applique donc un plancher de covariance (le biais commun) au lieu de faire
tendre l'incertitude vers zéro.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

CHI2_2D_99 = 9.21          # porte de Mahalanobis² (2 ddl, 99 %)


def cov_pack(P: np.ndarray) -> tuple:
    return float(P[0, 0]), float(P[0, 1]), float(P[1, 1])


def cov_unpack(c) -> np.ndarray:
    return np.array([[c[0], c[1]], [c[1], c[2]]], float)


def meas_point_cov(pose_cov: np.ndarray, theta: float, rng: float, bearing: float,
                   sigma_r: float, sigma_b: float) -> np.ndarray:
    """Covariance (monde) d'un point vu à (rng, bearing) depuis une pose incertaine (x, y, theta).
    Jacobien de p = (x + r cos(theta + b), y + r sin(theta + b)) par rapport à (x, y, theta, r, b)."""
    a = theta + bearing
    c, s = math.cos(a), math.sin(a)
    Jp = np.array([[1.0, 0.0, -rng * s], [0.0, 1.0, rng * c]])
    Jm = np.array([[c, -rng * s], [s, rng * c]])
    return Jp @ pose_cov[:3, :3] @ Jp.T + Jm @ np.diag([sigma_r ** 2, sigma_b ** 2]) @ Jm.T


def to_info(mu, P) -> tuple[np.ndarray, np.ndarray]:
    Y = np.linalg.inv(P)
    return Y, Y @ np.asarray(mu, float)


def from_info(Y, y) -> tuple[np.ndarray, np.ndarray]:
    P = np.linalg.inv(Y)
    return P @ y, P


def fuse_independent(items: Iterable[tuple]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Fusion bayésienne de gaussiennes indépendantes : somme des informations."""
    Y, y, n = np.zeros((2, 2)), np.zeros(2), 0
    for mu, P in items:
        Yi, yi = to_info(mu, P)
        Y += Yi
        y += yi
        n += 1
    return from_info(Y, y) if n else None


def remove_contribution(mu, P, mu_i, P_i) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Estimation globale moins une contribution indépendante (ex. la mienne). None si mal conditionné."""
    Y, y = to_info(mu, P)
    Yi, yi = to_info(mu_i, P_i)
    Yr = Y - Yi
    if np.linalg.eigvalsh(Yr).min() <= 1e-9:
        return None
    return from_info(Yr, y - yi)


def covariance_intersection(a: tuple, b: tuple, n_grid: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Fusion prudente de deux estimations de corrélation inconnue (poids choisi pour minimiser det(P))."""
    Ya, ya = to_info(*a)
    Yb, yb = to_info(*b)
    best = None
    for w in np.linspace(0.0, 1.0, n_grid):
        Y = w * Ya + (1 - w) * Yb
        d = np.linalg.det(Y)
        if best is None or d > best[0]:
            best = (d, w)
    w = best[1]
    return from_info(w * Ya + (1 - w) * Yb, w * ya + (1 - w) * yb)


def kalman_update(mu, P, z, R, floor_var: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Mise à jour d'un point fixe par une mesure z ~ N(point, R), avec plancher de variance (biais commun)."""
    mu, P, z = np.asarray(mu, float), np.asarray(P, float), np.asarray(z, float)
    S = P + R
    K = P @ np.linalg.inv(S)
    mu2 = mu + K @ (z - mu)
    P2 = (np.eye(2) - K) @ P
    P2 = 0.5 * (P2 + P2.T)
    lo = float(np.linalg.eigvalsh(P2).min())
    if lo < floor_var:
        P2 = P2 + (floor_var - lo) * np.eye(2)
    return mu2, P2


def mahalanobis2(mu1, P1, mu2, P2) -> float:
    d = np.asarray(mu1, float) - np.asarray(mu2, float)
    return float(d @ np.linalg.inv(np.asarray(P1) + np.asarray(P2)) @ d)
