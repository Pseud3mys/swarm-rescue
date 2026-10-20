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
    - bruit réel du lidar (σ = 2,5 px, appliqué AUSSI aux rayons sans impact : 300 + bruit) : un rayon n'est un
      impact que s'il est à plus de 3σ sous la portée max (sinon il peut s'agir d'un lidar arrivé en portée max) ;
    - poids des preuves selon l'incertitude de MA pose : un impact vaut ev_hit si la pose est sûre, moins si elle
      dérive (la couche porte ainsi l'incertitude de la carte) ; rien n'est écrit au-delà de max_pose_sigma ;
    - rayons masqués (blessés, autres drones, porteurs) : ni impact ni preuve libre au-delà de l'objet.
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
        """`ignore` : masque booléen des rayons à ne pas prendre comme obstacle (blessé, autre drone)."""
        p, m = self.p, self.w.map
        sig = pose.sigma
        if sig > p.max_pose_sigma_for_mapping:
            return 0
        r = np.asarray(ranges, float)
        ang = np.asarray(angles, float)
        ok = np.isfinite(r)
        # Un rayon sans impact vaut 300 + bruit : seul un retour à plus de 3σ sous la portée max est un impact. Un mur
        # vers 290-300 px renvoie tantôt un impact, tantôt « portée max » : un rayon sans impact n'écrit donc du libre
        # que jusqu'à map_far_buffer avant cette limite (sinon la cellule du mur alterne mur / libre).
        r_max = p.lidar_max_range - 3.0 * p.lidar_sigma
        hit = ok & (r < r_max) & (r > 3.0)
        if ignore is not None:
            hit &= ~ignore
        a = pose.theta + ang
        c, s = np.cos(a), np.sin(a)
        reach = np.where(ok, np.where(hit, r - self._free_margin(pose, r, ang, hit), r_max - p.map_far_buffer), 0.0)
        if ignore is not None:       # derrière/sur un autre drone : on ne sait rien de plus que sa distance
            reach = np.where(ignore & ok, np.maximum(r - 25.0, 0.0), reach)
        # rayons fiables : impact continu avec ses voisins (pas un rayon qui a « glissé » à travers un mur), incidence
        # franche (en incidence rasante un rayon peut passer entre deux pixels de mur), portée moyenne
        cont, sinb = _surface(r, ang, hit)
        rel_hit = hit & cont & (r < p.map_rel_range)
        rel = rel_hit & (sinb >= p.map_rel_sin)
        S = self.samples[None, :]
        valid = S <= reach[:, None]
        xs = pose.x + c[:, None] * S
        ys = pose.y + s[:, None] * S
        free, hits = self._free, self._hit
        free[:] = False
        hits[:] = False
        ix, iy = m.world_to_cell(np.stack((xs[valid], ys[valid]), axis=1))
        free[ix, iy] = True
        hx_w, hy_w = pose.x + c * r, pose.y + s * r
        if hit.any():
            hx, hy = m.world_to_cell(np.stack((hx_w[hit], hy_w[hit]), axis=1))
            hits[hx, hy] = True
        free &= ~hits
        L = m.layers.get(self.w.me)
        if L is not None and p.map_lock <= 20:
            # cellules CERTAINES de ma couche (monde statique : drones et blessés sont masqués) : seuls des rayons
            # fiables peuvent les contredire. Sinon un drone garé efface petit à petit un mur vu en rasant ou en bout
            # de portée, et des impacts lointains bruités épaississent les murs dans du vide sûr.
            vr = valid & rel[:, None]
            free_rel = np.zeros_like(free)
            jx, jy = m.world_to_cell(np.stack((xs[vr], ys[vr]), axis=1))
            free_rel[jx, jy] = True
            free &= (L < p.map_lock) | free_rel
            hit_rel = np.zeros_like(hits)
            if rel_hit.any():
                kx, ky = m.world_to_cell(np.stack((hx_w[rel_hit], hy_w[rel_hit]), axis=1))
                hit_rel[kx, ky] = True
            hits &= (L > -p.map_lock) | hit_rel
        fx, fy = np.nonzero(free)
        hx, hy = np.nonzero(hits)
        ev_hit = p.ev_hit if sig < 6.0 else (2 if sig < 10.0 else 1)
        far = np.zeros_like(hits)
        if (hit & (r > p.map_far_hit)).any():         # impacts lointains : décalage latéral (cap) plus grand
            far_ray = hit & (r > p.map_far_hit)
            ux, uy = m.world_to_cell(np.stack((hx_w[far_ray], hy_w[far_ray]), axis=1))
            far[ux, uy] = True
            near_ray = hit & ~far_ray
            if near_ray.any():
                vx, vy = m.world_to_cell(np.stack((hx_w[near_ray], hy_w[near_ray]), axis=1))
                far[vx, vy] = False
        ev_h = np.where(far[hx, hy], min(ev_hit, 1), ev_hit)
        m.add_evidence(self.w.me, np.r_[fx, hx], np.r_[fy, hy], np.r_[np.full(fx.size, p.ev_free), ev_h])
        return int(fx.size + hx.size)

    def _free_margin(self, pose: Pose, r: np.ndarray, angles: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """Longueur de rayon (avant l'impact) sans preuve libre. Un rayon qui arrive sur un mur avec un angle beta
        passe à moins d'une cellule du mur sur res / sin(beta) avant l'impact : il traverse la partie vide de la
        cellule du mur, et un drone qui ne voit un mur qu'en rasant (garé) l'effacerait petit à petit. Optionnel : en
        plus, le décalage dû à la pose (sigma_xy + r * sigma_cap). Mur local estimé par les impacts voisins."""
        p, m = self.p, self.w.map
        base = 0.7 * m.res
        lat = np.full(r.shape, p.map_margin_cell * m.res)
        if p.map_margin_k > 0.0:
            sth = math.sqrt(max(float(pose.cov[2, 2]), 0.0))
            lat = lat + p.map_margin_k * (pose.sigma + np.where(np.isfinite(r), r, 0.0) * sth)
        if not lat.any():
            return np.full(r.shape, base)
        cont, sinb = _surface(r, angles, hit)
        sinb = np.where(cont, np.maximum(sinb, 0.1), 1.0)
        return np.maximum(base, np.minimum(lat / sinb, p.map_margin_max))


def _surface(r: np.ndarray, angles: np.ndarray, hit: np.ndarray):
    """Par rayon : impact continu avec ses deux voisins (même surface), et sinus de l'angle entre le rayon et cette
    surface (estimée par les impacts voisins ; 1 = incidence normale, ~0 = rasante)."""
    rr = np.where(np.isfinite(r), r, 0.0)
    x, y = rr * np.cos(angles), rr * np.sin(angles)
    n = len(r)
    prv, nxt = (np.arange(n) - 1) % n, (np.arange(n) + 1) % n
    cont = hit[prv] & hit[nxt] & (np.abs(rr[nxt] - rr) < 6.0 + 0.08 * rr) & (np.abs(rr - rr[prv]) < 6.0 + 0.08 * rr)
    tx, ty = x[nxt] - x[prv], y[nxt] - y[prv]
    sinb = np.abs(np.cos(angles) * ty - np.sin(angles) * tx) / np.maximum(np.hypot(tx, ty), 1e-6)
    return cont, sinb


def _mask_bodies(mask, ranges, angles, rel_bearing, dist, radius) -> np.ndarray:
    """Rayons qui touchent un corps de rayon `radius` centré à (dist, rel_bearing) dans le repère du drone."""
    if dist < 1.0:
        return mask
    half = math.asin(min(1.0, radius / dist)) + 0.03
    near = (np.abs(_wrap(angles - rel_bearing)) < half) & (ranges > dist - radius - 10.0) & (ranges < dist + radius + 6.0)
    return near if mask is None else (mask | near)


def dynamic_ray_mask(ranges, angles, semantic: Iterable, pose: Optional[Pose] = None,
                     bodies: Sequence = ()) -> Optional[np.ndarray]:
    """Rayons lidar qui touchent un objet mobile, à ne pas graver comme mur :
      - blessés et drones vus par le capteur sémantique (portée 200 px) ;
      - corps connus par la mémoire (`bodies` = [(xy monde, rayon)]) : autres drones d'après leur état annoncé,
        porteurs (avec leur blessé), blessés connus : couvre le lidar au-delà de la portée sémantique (200-300 px)."""
    if ranges is None or angles is None:
        return None
    ranges = np.asarray(ranges, float)
    angles = np.asarray(angles, float)
    mask = None
    for det in semantic:
        if det.distance < 1.0:
            continue
        if det.kind == "drone":
            mask = _mask_bodies(mask, ranges, angles, det.angle, det.distance + 10.0, 20.0)
        elif det.kind == "wounded":
            mask = _mask_bodies(mask, ranges, angles, det.angle, det.distance + 12.0, 16.0)
    if pose is not None:
        for xy, rad in bodies:
            dx, dy = xy[0] - pose.x, xy[1] - pose.y
            d = math.hypot(dx, dy)
            if d > 330.0:
                continue
            mask = _mask_bodies(mask, ranges, angles, math.atan2(dy, dx) - pose.theta, d, rad)
    return mask


def drone_ray_mask(ranges, angles, semantic: Iterable) -> Optional[np.ndarray]:
    """Compatibilité : masque des seuls corps vus par le capteur sémantique."""
    return dynamic_ray_mask(ranges, angles, semantic)


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
