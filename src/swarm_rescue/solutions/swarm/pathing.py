"""Planification de trajectoire sur la carte fusionnée (grille).

Un seul calcul de Dijkstra (scipy) depuis le drone donne le coût de TOUS les buts possibles :
frontières d'exploration, blessés, centre de secours. Le choix de la cible (missions.py) lit ces coûts.

  - obstacles gonflés du rayon du drone (cv2.dilate), fermeture morphologique pour boucher les trous entre impacts ;
  - coût de proximité des murs (reste au milieu des couloirs) ;
  - planification optimiste : le terrain inconnu est traversable à coût élevé (on cherche à "aller là-bas à travers
    les murs inconnus", on replanifie quand la carte se précise) ;
  - zones de coût supplémentaire fournies par l'appelant (autres drones, zones dangereuses).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .config import Params
from .gridmap import FREE_THR, OCC_THR
from .schema import Route
from .world import WorldModel

_NB = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)))


@dataclass
class Frontier:
    cell: tuple[int, int]
    cost: float
    size: int
    pos: np.ndarray


class PlanGrid:
    def __init__(self, world: WorldModel, params: Params, start_xy, inflate: Optional[int] = None,
                 extra_cost: Optional[np.ndarray] = None):
        self.world, self.p = world, params
        m = world.map
        self.m = m
        k = params.inflate_cells if inflate is None else inflate
        view = m.view()
        self.W, self.H = view.shape
        arena = np.zeros(view.shape, bool)
        arena[:m.w0, :m.h0] = True
        self.arena = arena
        occ = ((view >= OCC_THR) & arena).astype(np.uint8)
        self.free = (view <= FREE_THR) & arena
        closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.closed = closed
        if k > 0:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            blocked = cv2.dilate(closed, ker) > 0
        else:
            blocked = closed > 0
        blocked |= ~arena
        # la cellule du drone et son voisinage immédiat sont libres par définition
        self.start = tuple(int(v[0]) for v in m.world_to_cell(start_xy))
        sx, sy = self.start
        blocked[max(sx - 2, 0):sx + 3, max(sy - 2, 0):sy + 3] = False
        self.blocked = blocked
        unknown = arena & ~self.free & (closed == 0)
        self.unknown = unknown
        self.trav = ~blocked
        d = cv2.distanceTransform((1 - closed).astype(np.uint8), cv2.DIST_L2, 3)
        rw = params.wall_cost_cells
        prox = np.clip((rw - d) / rw, 0.0, 1.0) ** 2 * params.wall_cost
        self.los_max_cost = 1.0 + params.wall_cost * ((rw - params.los_wall_cells) / rw) ** 2
        cost = 1.0 + prox
        cost = np.where(unknown, cost * params.unknown_cost, cost)
        if extra_cost is not None:
            cost = cost + extra_cost
        self.cost = cost.astype(np.float32)
        self._dist: Optional[np.ndarray] = None
        self._pred: Optional[np.ndarray] = None
        self._g = None

    # ------------------------------------------------------------------ graphe + Dijkstra
    def solve(self) -> None:
        W, H = self.W, self.H
        idx = np.arange(W * H).reshape(W, H)
        rows, cols, wts = [], [], []
        trav, cost = self.trav, self.cost
        for dx, dy, ln in _NB:
            xs0, xs1 = (0, W - dx) if dx >= 0 else (-dx, W)
            ys0, ys1 = (0, H - dy) if dy >= 0 else (-dy, H)
            sl_s = (slice(xs0, xs1), slice(ys0, ys1))
            sl_d = (slice(xs0 + dx, xs1 + dx), slice(ys0 + dy, ys1 + dy))
            ok = trav[sl_s] & trav[sl_d]
            if dx != 0 and dy != 0:          # pas de coupe-coin à travers un obstacle
                ok &= trav[xs0 + dx:xs1 + dx, ys0:ys1] & trav[xs0:xs1, ys0 + dy:ys1 + dy]
            w = ln * 0.5 * (cost[sl_s] + cost[sl_d])
            rows.append(idx[sl_s][ok]); cols.append(idx[sl_d][ok]); wts.append(w[ok])
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        g = coo_matrix((w, (r, c)), shape=(W * H, W * H)).tocsr()
        self._g = g
        src = self.start[0] * H + self.start[1]
        dist, pred = dijkstra(g, directed=False, indices=src, return_predecessors=True)
        self._dist = dist.reshape(W, H)
        self._pred = pred

    # ------------------------------------------------------------------ Dijkstra depuis d'autres points
    def dists_from(self, cells) -> np.ndarray:
        """Distances (n, W, H) depuis plusieurs cellules sources, sur le même graphe (après solve())."""
        H = self.H
        idx = [c[0] * H + c[1] for c in cells]
        if not idx:
            return np.zeros((0, self.W, H))
        d = dijkstra(self._g, directed=False, indices=idx)
        return np.asarray(d).reshape(len(idx), self.W, H)

    def tree_from(self, cell):
        """(distances (W, H), prédécesseurs) depuis une cellule : pour tracer des chemins depuis un point fixe."""
        src = cell[0] * self.H + cell[1]
        d, pred = dijkstra(self._g, directed=False, indices=src, return_predecessors=True)
        return d.reshape(self.W, self.H), pred

    @staticmethod
    def lookup(dist: np.ndarray, cell, radius: int = 3) -> float:
        """Distance au point `cell` en tolérant une cellule bloquée (agent collé à un mur gonflé)."""
        W, H = dist.shape
        x0, x1 = max(cell[0] - radius, 0), min(cell[0] + radius + 1, W)
        y0, y1 = max(cell[1] - radius, 0), min(cell[1] + radius + 1, H)
        v = dist[x0:x1, y0:y1]
        return float(v.min()) if v.size else float("inf")

    # ------------------------------------------------------------------ requêtes
    def cell(self, xy) -> tuple[int, int]:
        ix, iy = self.m.world_to_cell(xy)
        return int(ix[0]), int(iy[0])

    def nearest_reachable(self, xy, radius: int = 6) -> Optional[tuple[int, int]]:
        cx, cy = self.cell(xy)
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, self.W)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, self.H)
        win = self._dist[x0:x1, y0:y1]
        if not np.isfinite(win).any():
            return None
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="ij")
        pen = np.where(np.isfinite(win), win + 12.0 * np.hypot(gx - cx, gy - cy), np.inf)
        i = np.unravel_index(np.argmin(pen), pen.shape)
        return int(gx[i]), int(gy[i])

    def cost_to(self, xy, radius: int = 6) -> float:
        c = self.nearest_reachable(xy, radius)
        return float("inf") if c is None else float(self._dist[c])

    def _trace(self, cell, pred=None, src_cell=None) -> Optional[list]:
        H = self.H
        pred = self._pred if pred is None else pred
        src_cell = self.start if src_cell is None else src_cell
        n = cell[0] * H + cell[1]
        src = src_cell[0] * H + src_cell[1]
        out = [n]
        for _ in range(self.W * self.H):
            if n == src:
                break
            n = pred[n]
            if n < 0:
                return None
            out.append(n)
        else:
            return None
        out.reverse()
        a = np.array(out)
        return list(zip((a // H).tolist(), (a % H).tolist()))

    def _los(self, a, b) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        t = np.linspace(0.0, 1.0, n + 1)
        xs = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
        ys = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
        return bool(self.trav[xs, ys].all() and not (self.cost[xs, ys] > self.los_max_cost).any())

    def path_cells(self, cell, pred, src_cell) -> Optional[list]:
        """Chemin (liste de cellules, non simplifié) de `src_cell` à `cell` d'après un arbre de tree_from()."""
        return self._trace(cell, pred, src_cell)

    def route_to(self, xy, t: int, speed: float, radius: int = 6) -> Optional[Route]:
        goal = self.nearest_reachable(xy, radius)
        if goal is None:
            return None
        cells = self._trace(goal)
        if not cells:
            return None
        if len(cells) == 1:
            cells = [cells[0], cells[0]]
        cells = self._simplify(cells)
        pts = self.m.cell_to_world(np.array([c[0] for c in cells]), np.array([c[1] for c in cells]))
        # le dernier point est le but réel s'il est proche de la cellule atteinte
        goal_xy = np.asarray(xy, float)
        if np.linalg.norm(pts[-1] - goal_xy) < 1.5 * radius * self.m.res:
            pts[-1] = goal_xy if self._los(self.cell(pts[-2]) if len(pts) > 1 else self.start, self.cell(goal_xy)) else pts[-1]
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        eta = (t + seg / max(speed, 1e-3)).astype(int)
        return Route(pts, eta, (float(pts[-1][0]), float(pts[-1][1])), t)

    def _simplify(self, cells: list) -> list:
        """Raccourcis par ligne de vue sur la grille gonflée (supprime le zigzag du 8-connexe)."""
        out = [cells[0]]
        i = 0
        n = len(cells)
        while i < n - 1:
            j = min(n - 1, i + 60)
            while j > i + 1 and not self._los(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return out

    # ------------------------------------------------------------------ frontières
    def frontiers(self, min_cells: int) -> list[Frontier]:
        free = self.free
        unk_core = self.unknown & ~(cv2.dilate(self.closed, np.ones((3, 3), np.uint8)) > 0)
        near_unk = cv2.dilate(unk_core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        reach = np.isfinite(self._dist)
        front = free & near_unk & self.trav & reach
        if not front.any():
            return []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(front.astype(np.uint8), connectivity=8)
        ix, iy = np.nonzero(labels)
        lab = labels[ix, iy]
        order = np.argsort(lab, kind="stable")
        ix, iy, lab = ix[order], iy[order], lab[order]
        starts = np.r_[0, np.nonzero(np.diff(lab))[0] + 1]
        out = []
        for s, e in zip(starts, np.r_[starts[1:], len(lab)]):
            if e - s < min_cells:
                continue
            d = self._dist[ix[s:e], iy[s:e]]
            j = int(np.argmin(d))
            cell = (int(ix[s + j]), int(iy[s + j]))
            pos = self.m.cell_to_world(np.array([ix[s:e].mean()]), np.array([iy[s:e].mean()]))[0]
            out.append(Frontier(cell, float(d[j]), int(e - s), pos))
        return out

    def explored_fraction(self) -> float:
        """Part de la zone (hors murs) connue : utile pour décider la fin de l'exploration."""
        a = self.arena
        return float((self.free & a).sum() / max(a.sum(), 1))


def extra_cost_from_agents(world: WorldModel, t: int, m, shape, me: int, horizon: int = 400,
                           radius_px: float = 35.0, weight: float = 2.5) -> Optional[np.ndarray]:
    """Surcoût autour des positions actuelles et futures (trajectoires annoncées) des autres drones."""
    others = world.others()
    if not others:
        return None
    img = np.zeros(shape, np.float32)
    rc = max(int(radius_px / m.res), 1)
    for a in others:
        if t - a.t > 600:             # information trop ancienne
            continue
        pts = [(a.pos, 1.0)]
        for w in a.route:
            if w.eta - t < horizon:
                pts.append(((w.x, w.y), 0.6))
        for pos, k in pts:
            ix, iy = m.world_to_cell(np.array(pos))
            cv2.circle(img, (int(iy[0]), int(ix[0])), rc, float(weight * k), -1)   # cv2 : (col, ligne) = (iy, ix)
    return img
