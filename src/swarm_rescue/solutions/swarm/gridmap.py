"""Carte d'occupation par tuiles, une couche par drone propriétaire.

Pourquoi une couche par origine plutôt qu'une carte unique :
  - les log-odds de deux drones sont des preuves INDÉPENDANTES (biais de pose différents) : la fusion exacte est leur
    SOMME (faite à la lecture, `view()`). Réémettre la somme compterait deux fois l'information quand elle circule en
    boucle (A->B->C->A). Chaque couche est la contribution déjà fusionnée de son drone (toutes ses scans, pondérés
    par l'incertitude de sa pose, cf. perception.py) ; on la relaie telle quelle et on ne fait que remplacer une tuile
    par une révision plus récente.
  - l'algo de fusion (somme + clip) est dans `view()`, remplaçable sans toucher au réseau.

Révisions : tampons d'un compteur par origine (`known[o]`), uniques et croissants (cf. schema.py). Une tuile de MA
couche n'est republiée que si son contenu a vraiment changé par rapport à la version publiée (écart >= PUB_DELTA sur
une cellule, ou changement de classe libre / inconnu / occupé) : une zone saturée (déjà vue 20 fois) ne génère plus
aucun trafic, même sous un drone immobile.

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
PUB_DELTA = 4    # écart (log-odds) à la version publiée qui justifie une nouvelle révision de tuile


def _cls(v: np.ndarray) -> np.ndarray:
    return (v >= OCC_THR).astype(np.int8) - (v <= FREE_THR).astype(np.int8)


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
        self._pub: dict[int, np.ndarray] = {}        # version publiée de mes couches (pour décider d'une révision)
        self.rev = np.zeros((max_origins, self.ncx, self.ncy), dtype=np.int64)   # tampon de chaque tuile
        self.known = np.zeros(max_origins, dtype=np.int64)                      # vecteur de versions
        self._view = np.zeros((self.W, self.H), dtype=np.int16)
        self._dirty: set[tuple[int, int]] = set()
        self.n_bumps = 0                             # stats : révisions publiées
        self.n_writes = 0                            # stats : tuiles écrites (publiées ou non)

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
        """Ajoute `delta` (scalaire ou tableau) aux cellules (ix, iy) de la couche `origin` (la mienne).
        Gère les doublons d'indices. Republie les tuiles dont le contenu a changé de façon significative."""
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
        self._commit(origin, uniq)

    def clamp_max(self, origin: int, ix, iy, vmax: int) -> None:
        """Ramène à <= vmax les cellules (ix, iy) de MA couche (effacement d'un obstacle qui n'en est pas un)."""
        ix = np.asarray(ix, dtype=np.int64).ravel()
        iy = np.asarray(iy, dtype=np.int64).ravel()
        if ix.size == 0 or origin not in self.layers:
            return
        uniq = np.unique(ix * self.H + iy)
        cells = self.layers[origin].reshape(-1)
        cells[uniq] = np.minimum(cells[uniq], vmax)
        self._commit(origin, uniq)

    def _commit(self, origin: int, uniq: np.ndarray) -> None:
        """Tuiles touchées : vue à recalculer ; nouvelle révision seulement si le contenu s'écarte de la version
        publiée (les voisins gardent sinon une version à moins de PUB_DELTA près : sans conséquence)."""
        cx, cy = (uniq // self.H) // self.C, (uniq % self.H) // self.C
        chunks = np.unique(cx * self.ncy + cy)
        tx, ty = chunks // self.ncy, chunks % self.ncy
        layer = self.layers[origin]
        pub = self._pub.get(origin)
        if pub is None:
            pub = self._pub[origin] = np.zeros_like(layer)
        self._dirty.update(zip(tx.tolist(), ty.tolist()))
        self.n_writes += len(chunks)
        # tuiles touchées extraites d'un coup : (k, C, C)
        C = self.C
        tl = layer.reshape(self.ncx, C, self.ncy, C).transpose(0, 2, 1, 3)[tx, ty]
        tp = pub.reshape(self.ncx, C, self.ncy, C).transpose(0, 2, 1, 3)[tx, ty]
        big = (np.abs(tl.astype(np.int16) - tp) >= PUB_DELTA) | (_cls(tl) != _cls(tp))
        bump = big.reshape(len(chunks), -1).any(axis=1)
        if not bump.any():
            return
        for x, y in zip(tx[bump].tolist(), ty[bump].tolist()):
            self.known[origin] += 1
            self.rev[origin, x, y] = self.known[origin]
            sl = self._sl(x, y)
            pub[sl] = layer[sl]
        self.n_bumps += int(bump.sum())

    # ---- synchronisation ----------------------------------------------
    def _sl(self, cx: int, cy: int):
        C = self.C
        return slice(cx * C, (cx + 1) * C), slice(cy * C, (cy + 1) * C)

    def export_patch(self, origin: int, cx: int, cy: int) -> MapPatch:
        # ma couche : on émet la version PUBLIÉE (celle de la révision), pas le contenu vivant : sinon deux voisins
        # recevraient des contenus différents sous la même révision et leurs cartes ne convergeraient pas exactement
        src = self._pub.get(origin)
        tile = (src if src is not None else self._layer(origin))[self._sl(cx, cy)]
        v0 = int(tile[0, 0])
        if (tile == v0).all():                          # tuile uniforme (vide, saturée libre) : 1 octet
            return MapPatch(origin, cx, cy, int(self.rev[origin, cx, cy]), None, v0)
        data = tile.copy()
        data.flags.writeable = False     # le simulateur passe les messages par référence
        return MapPatch(origin, cx, cy, int(self.rev[origin, cx, cy]), data)

    def apply_patch(self, p: MapPatch) -> bool:
        """Renvoie True si la tuile était plus récente que la nôtre (et l'a remplacée)."""
        if not (0 <= p.origin < self.max_origins and 0 <= p.cx < self.ncx and 0 <= p.cy < self.ncy):
            return False
        if p.rev <= self.rev[p.origin, p.cx, p.cy]:
            return False
        if p.data is not None and p.data.shape != (self.C, self.C):
            return False
        self._layer(p.origin)[self._sl(p.cx, p.cy)] = p.data if p.data is not None else p.fill
        self.rev[p.origin, p.cx, p.cy] = p.rev
        self._dirty.add((p.cx, p.cy))
        return True

    def tiles_after(self, origin: int, base: int) -> list[tuple[int, int, int]]:
        """Tuiles de `origin` de tampon > base, par tampon croissant : [(tampon, cx, cy)]."""
        r = self.rev[origin]
        xs, ys = np.nonzero(r > base)
        if xs.size == 0:
            return []
        stamps = r[xs, ys]
        o = np.argsort(stamps, kind="stable")
        return list(zip(stamps[o].tolist(), xs[o].tolist(), ys[o].tolist()))

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
