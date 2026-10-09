"""Carte d'occupation par tuiles, une couche par drone propriétaire.

Pourquoi une couche par origine plutôt qu'une carte unique :
  - additionner des log-odds reçus d'un voisin dans MA carte compte deux fois l'information
    quand elle circule en boucle (A->B->C->A). Ici chaque origine garde sa couche, on ne fait
    que remplacer une tuile par une révision plus récente, puis on SOMME les couches à la lecture.
  - si le GPS d'un drone dérive (bruit AR(1)), on peut plus tard réduire le poids de sa couche.
  - l'algo de fusion (somme + clip) est dans `view()`, remplaçable sans toucher au réseau.

Convention d'indices : cellule (ix, iy), ix selon x, iy selon y ; origine du monde au centre.
"""
from __future__ import annotations

import math

import numpy as np

from .schema import MAX_ORIGINS, MapPatch

CLIP = 20        # saturation par origine (permet de corriger une erreur ancienne)
VIEW_CLIP = 60   # saturation de la vue fusionnée
OCC_THR = 5      # vue >= OCC_THR  -> occupé
FREE_THR = -3    # vue <= FREE_THR -> libre


class ChunkedMap:
    def __init__(self, size_area, resolution: float = 8.0, chunk: int = 16,
                 max_origins: int = MAX_ORIGINS):
        self.res = float(resolution)
        self.C = int(chunk)
        self.half = np.asarray(size_area, dtype=float) / 2.0
        w = int(math.ceil(size_area[0] / self.res))
        h = int(math.ceil(size_area[1] / self.res))
        self.w0, self.h0 = w, h                      # taille réelle de l'arène (cellules)
        self.ncx = int(math.ceil(w / self.C))
        self.ncy = int(math.ceil(h / self.C))
        self.W, self.H = self.ncx * self.C, self.ncy * self.C   # taille paddée
        self.max_origins = max_origins
        self.layers: dict[int, np.ndarray] = {}
        self.rev = np.zeros((max_origins, self.ncx, self.ncy), dtype=np.uint32)
        self._view = np.zeros((self.W, self.H), dtype=np.int16)
        self._dirty: set[tuple[int, int]] = set()

    # ---- conversions ---------------------------------------------------
    def world_to_cell(self, xy) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        idx = np.floor((xy + self.half) / self.res).astype(int)
        return (np.clip(idx[:, 0], 0, self.W - 1), np.clip(idx[:, 1], 0, self.H - 1))

    def cell_to_world(self, ix, iy) -> np.ndarray:
        c = np.stack((np.asarray(ix), np.asarray(iy)), axis=-1).astype(float)
        return (c + 0.5) * self.res - self.half

    # ---- écriture locale ----------------------------------------------
    def _layer(self, origin: int) -> np.ndarray:
        if origin not in self.layers:
            self.layers[origin] = np.zeros((self.W, self.H), dtype=np.int8)
        return self.layers[origin]

    def add_evidence(self, origin: int, ix, iy, delta) -> None:
        """Ajoute `delta` (scalaire ou tableau) aux cellules (ix, iy) de la couche `origin`.
        Gère les doublons d'indices. Incrémente la révision des tuiles touchées."""
        ix = np.asarray(ix, dtype=np.int64).ravel()
        iy = np.asarray(iy, dtype=np.int64).ravel()
        if ix.size == 0:
            return
        flat = ix * self.H + iy
        uniq, inv = np.unique(flat, return_inverse=True)
        delta = np.broadcast_to(np.asarray(delta, dtype=float), flat.shape)
        sums = np.bincount(inv, weights=delta).astype(np.int16)
        cells = self._layer(origin).reshape(-1)          # vue (tableau contigu)
        cells[uniq] = np.clip(cells[uniq].astype(np.int16) + sums, -CLIP, CLIP).astype(np.int8)
        cx, cy = (uniq // self.H) // self.C, (uniq % self.H) // self.C
        chunks = np.unique(cx * self.ncy + cy)
        self.rev[origin].reshape(-1)[chunks] += 1
        for c in chunks:
            self._dirty.add((int(c // self.ncy), int(c % self.ncy)))

    # ---- synchronisation ----------------------------------------------
    def _sl(self, cx: int, cy: int):
        C = self.C
        return slice(cx * C, (cx + 1) * C), slice(cy * C, (cy + 1) * C)

    def export_patch(self, origin: int, cx: int, cy: int) -> MapPatch:
        data = self._layer(origin)[self._sl(cx, cy)].copy()
        data.flags.writeable = False     # le simulateur passe les messages par référence
        return MapPatch(origin, cx, cy, int(self.rev[origin, cx, cy]), data)

    def apply_patch(self, p: MapPatch) -> bool:
        """Renvoie True si la tuile était plus récente que la nôtre (et l'a remplacée)."""
        if not (0 <= p.origin < self.max_origins and 0 <= p.cx < self.ncx and 0 <= p.cy < self.ncy):
            return False
        if p.data.shape != (self.C, self.C) or p.rev <= self.rev[p.origin, p.cx, p.cy]:
            return False
        self._layer(p.origin)[self._sl(p.cx, p.cy)] = p.data
        self.rev[p.origin, p.cx, p.cy] = p.rev
        self._dirty.add((p.cx, p.cy))
        return True

    def newer_than(self, other_rev: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Tuiles que je possède en révision plus récente que `other_rev`.
        Renvoie (origin, cx, cy, retard) ; vide si les grilles sont incompatibles."""
        if other_rev.shape != self.rev.shape:
            return []
        lag = self.rev.astype(np.int64) - other_rev.astype(np.int64)
        return [(int(o), int(x), int(y), int(lag[o, x, y])) for o, x, y in np.argwhere(lag > 0)]

    # ---- lecture (pour planner / contrôle) -----------------------------
    def view(self) -> np.ndarray:
        """Log-odds fusionnés (int16, shape (W, H)). Recalcule seulement les tuiles modifiées."""
        for cx, cy in self._dirty:
            sl = self._sl(cx, cy)
            acc = np.zeros((self.C, self.C), dtype=np.int16)
            for layer in self.layers.values():
                acc += layer[sl]
            self._view[sl] = np.clip(acc, -VIEW_CLIP, VIEW_CLIP)
        self._dirty.clear()
        return self._view

    def occupied(self) -> np.ndarray:
        return self.view() >= OCC_THR

    def free(self) -> np.ndarray:
        return self.view() <= FREE_THR
