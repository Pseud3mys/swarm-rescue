"""Carte topologique : pièces et couloirs (nœuds), passages / portes (arêtes), organisés en arbre depuis la base.

Méthode "deux échelles" (d'après le prototype segproto.py), sur la carte d'occupation fusionnée d'un drone :
  1. DT optimiste : distance au mur CONNU le plus proche, l'inconnu compte comme libre (une pièce à moitié vue
     reste "large") ;
  2. noyaux de pièces = libre connu où DT >= r_room ; chaque composante connexe = une pièce ;
  3. extension GÉODÉSIQUE des noyaux de (r_room + 6 px) dans le libre connu (ne traverse pas un mur fin) ;
  4. reste du libre connu = couloirs / portes / éclats : un éclat ne touchant qu'une pièce (<= sliver cellules) est
     réabsorbé, une miette isolée est ignorée, le reste devient une région COULOIR ;
  5. arêtes = interfaces entre régions ; chaque tronçon continu d'interface est un PASSAGE (deux portes entre les
     mêmes pièces donnent deux passages), avec sa position et sa largeur ;
  6. arbre = parcours en largeur depuis la région de la base (parent, passage d'entrée, profondeur).
Une région est "ouverte" tant qu'elle contient des frontières d'exploration (attach_frontiers).

Convention : grilles indexées [ix, iy] (comme gridmap), coordonnées monde via ChunkedMap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

ROOM, CORRIDOR = "pièce", "couloir"
DOOR = "porte"          # (plus un type de région : les portes sont des arêtes ; gardé pour compatibilité)


@dataclass
class Region:
    rid: int
    kind: str                       # ROOM | CORRIDOR
    area: int                       # cellules libres connues
    center: np.ndarray              # barycentre des cellules, monde
    interior: np.ndarray            # cellule la plus dégagée (DT max), monde : toujours DANS la région
    width: float                    # px : 2 x DT max
    touches_unknown: bool = False
    parent: int = -1                # région parente dans l'arbre (-1 : racine ou non reliée)
    link: int = -1                  # passage vers le parent
    depth: int = -1
    frontiers: list = field(default_factory=list)   # indices de frontières (PlanGrid.frontiers) dans la région

    @property
    def open(self) -> bool:
        return bool(self.frontiers)

    @property
    def is_axis(self) -> bool:
        return self.kind == CORRIDOR


@dataclass
class Link:
    """Un passage (porte) : tronçon continu de l'interface entre deux régions."""
    lid: int
    a: int
    b: int
    pos: np.ndarray                 # milieu du passage, monde
    width: float                    # px


class TopoMap:
    def __init__(self, grid, root_xy=None, r_room: float = 48.0, sliver: int = 14, dt_clip: float = 100.0,
                 gap_cells: float = 2.5, prev: Optional["TopoMap"] = None):
        """`grid` : PlanGrid (on lit grid.free, grid.closed, grid.unknown, grid.arena, grid.m).
        r_room (px) : demi-largeur mini d'une pièce (DT des noyaux)."""
        self.m = m = grid.m
        res = m.res
        occ = (grid.closed > 0) | ~grid.arena
        free = grid.free & grid.arena & ~occ
        unknown = grid.unknown & ~occ
        # 1. DT optimiste (px)
        dt = np.minimum(cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3) * res, dt_clip)
        self.D = dt / res                                   # en cellules (pour les autres modules)
        # 2. noyaux
        core = ((dt >= r_room) & free).astype(np.uint8)
        n, lab_core = cv2.connectedComponents(core, connectivity=8)
        # 3. extension géodésique, confinée au libre connu
        rooms = lab_core.astype(np.float32)
        f8 = free.astype(np.float32)
        step = np.ones((3, 3), np.uint8)
        for _ in range(int(round((r_room + 6) / res))):
            rooms = np.where(rooms > 0, rooms, cv2.dilate(rooms, step) * f8)
        rooms = rooms.astype(np.int32)
        # 4. reste du libre connu : couloirs, portes, éclats
        mres, lab_left, stats, _ = cv2.connectedComponentsWithStats((free & (rooms == 0)).astype(np.uint8),
                                                                    connectivity=8)
        nb: dict[int, set] = {}
        for c, r in _adjacent_pairs(lab_left, rooms).tolist():
            nb.setdefault(c, set()).add(r)
        labels = rooms.copy()
        kinds: dict[int, str] = {r: ROOM for r in range(1, n)}
        for j in range(1, mres):
            area = int(stats[j, cv2.CC_STAT_AREA])
            N = nb.get(j, set())
            mask = lab_left == j
            if not N and area < 10:
                continue                                    # miette isolée
            if len(N) == 1 and area <= sliver:
                labels[mask] = next(iter(N))                # éclat de coin : réabsorbé par sa pièce
                continue
            rid = n + j
            labels[mask] = rid
            kinds[rid] = CORRIDOR
        self.labels = np.where(labels > 0, labels, -1).astype(np.int32)

        # régions
        unk_d = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.regions: dict[int, Region] = {}
        for rid, kind in kinds.items():
            mk = labels == rid
            if not mk.any():
                continue
            xs, ys = np.nonzero(mk)
            dm = np.where(mk, dt, -1.0)
            ix, iy = np.unravel_index(int(np.argmax(dm)), dt.shape)
            self.regions[rid] = Region(
                rid, kind, int(len(xs)),
                m.cell_to_world(np.array([xs.mean()]), np.array([ys.mean()]))[0],
                m.cell_to_world(np.array([ix]), np.array([iy]))[0],
                2.0 * float(dt[ix, iy]), bool((unk_d & mk).any()))

        # 5. passages : tronçons continus des interfaces entre régions
        pts_by_pair: dict[tuple, list] = {}
        W, H = labels.shape
        for a, b, px, py in ((labels[:-1, :], labels[1:, :], *np.meshgrid(np.arange(W - 1) + 0.5, np.arange(H), indexing="ij")),
                             (labels[:, :-1], labels[:, 1:], *np.meshgrid(np.arange(W), np.arange(H - 1) + 0.5, indexing="ij"))):
            mk = (a > 0) & (b > 0) & (a != b)
            if not mk.any():
                continue
            lo, hi = np.minimum(a[mk], b[mk]), np.maximum(a[mk], b[mk])
            for l, h_, xx, yy in zip(lo.tolist(), hi.tolist(), px[mk].tolist(), py[mk].tolist()):
                pts_by_pair.setdefault((l, h_), []).append((xx, yy))
        self.links: list[Link] = []
        for (a, b), pts in pts_by_pair.items():
            if a not in self.regions or b not in self.regions:
                continue
            P = np.array(pts)
            ax = 0 if np.ptp(P[:, 0]) >= np.ptp(P[:, 1]) else 1
            P = P[np.argsort(P[:, ax])]
            cut = np.nonzero(np.abs(np.diff(P[:, ax])) > gap_cells)[0] + 1
            for g in np.split(P, cut):
                if len(g) >= 2:
                    # coordonnées de cellule (x + 0.5 = interface) -> monde : cell_to_world ajoute déjà 0.5
                    pos = m.cell_to_world(np.array([g[:, 0].mean() - 0.5]), np.array([g[:, 1].mean() - 0.5]))[0]
                    self.links.append(Link(len(self.links), a, b, pos, float(len(g) * res)))

        # identifiants stables d'un calcul au suivant (recouvrement maximal avec le calcul précédent)
        self.next_id = max(self.regions, default=0) + 1
        if prev is not None and prev.labels.shape == self.labels.shape:
            self._inherit_ids(prev)

        # 6. arbre depuis la base
        self.root = -1
        if root_xy is not None:
            self.root = self.region_at(root_xy, search=10)
        if self.root < 0 and self.regions:
            self.root = max(self.regions.values(), key=lambda r: r.area).rid
        self._build_tree()

    def _inherit_ids(self, prev: "TopoMap") -> None:
        """Chaque région reprend l'identifiant de la région précédente qui la recouvre le plus (>= 50 % de sa
        surface, une seule fois) ; les autres reçoivent un identifiant neuf, jamais réutilisé."""
        mapping, taken = {}, set()
        order = sorted(self.regions.values(), key=lambda r: -r.area)
        for r in order:
            old = prev.labels[self.labels == r.rid]
            old = old[old >= 0]
            if old.size:
                ids, cnt = np.unique(old, return_counts=True)
                k = int(np.argmax(cnt))
                if cnt[k] >= 0.5 * r.area and int(ids[k]) not in taken:
                    mapping[r.rid] = int(ids[k])
                    taken.add(int(ids[k]))
        nxt = max(prev.next_id, max(taken, default=0) + 1)
        for r in order:
            if r.rid not in mapping:
                mapping[r.rid] = nxt
                nxt += 1
        self.next_id = nxt
        lut = np.full(max(self.regions) + 2, -1, np.int32)
        for a, b in mapping.items():
            lut[a] = b
        self.labels = np.where(self.labels >= 0, lut[np.clip(self.labels, 0, None)], -1).astype(np.int32)
        regions = {}
        for r in self.regions.values():
            r.rid = mapping[r.rid]
            regions[r.rid] = r
        self.regions = regions
        for l in self.links:
            l.a, l.b = mapping[l.a], mapping[l.b]

    # ------------------------------------------------------------------ requêtes
    def region_at(self, xy, search: int = 0) -> int:
        ix, iy = self.m.world_to_cell(np.asarray(xy, float))
        x, y = int(ix[0]), int(iy[0])
        r = int(self.labels[x, y])
        if r >= 0 or search <= 0:
            return r
        x0, x1 = max(x - search, 0), min(x + search + 1, self.labels.shape[0])
        y0, y1 = max(y - search, 0), min(y + search + 1, self.labels.shape[1])
        win = self.labels[x0:x1, y0:y1]
        if (win >= 0).any():
            gx, gy = np.nonzero(win >= 0)
            k = int(np.argmin((gx + x0 - x) ** 2 + (gy + y0 - y) ** 2))
            return int(win[gx[k], gy[k]])
        return -1

    def region_of_cell(self, cell) -> int:
        return int(self.labels[cell[0], cell[1]])

    def axis_mask(self, margin_cells: int = 2) -> np.ndarray:
        """Axes de passage : couloirs + passages (portes) élargis de `margin_cells`. On n'y stationne pas."""
        ax = np.zeros(self.labels.shape, np.uint8)
        for r in self.regions.values():
            if r.is_axis:
                ax[self.labels == r.rid] = 1
        if self.links:
            ix, iy = self.m.world_to_cell(np.array([l.pos for l in self.links]))
            ax[ix, iy] = 1
        k = 2 * margin_cells + 1
        return cv2.dilate(ax, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0

    def ancestors(self, rid: int) -> list[int]:
        out = []
        while rid >= 0 and rid in self.regions and len(out) < 64:
            out.append(rid)
            rid = self.regions[rid].parent
        return out

    def attach_frontiers(self, frontiers) -> None:
        for r in self.regions.values():
            r.frontiers = []
        for i, f in enumerate(frontiers):
            rid = self.region_of_cell(f.cell)
            if rid not in self.regions:                      # frontière au bord d'une région : la plus proche
                rid = self.region_at(self.m.cell_to_world(np.array([f.cell[0]]), np.array([f.cell[1]]))[0], 3)
            if rid in self.regions:
                self.regions[rid].frontiers.append(i)

    # ------------------------------------------------------------------ arbre
    def _build_tree(self) -> None:
        if self.root < 0:
            return
        adj: dict[int, list] = {}
        for l in self.links:
            adj.setdefault(l.a, []).append((l.b, l.lid))
            adj.setdefault(l.b, []).append((l.a, l.lid))
        self.regions[self.root].depth = 0
        queue = [self.root]
        while queue:
            r = queue.pop(0)
            for nbr, lid in sorted(adj.get(r, []), key=lambda x: x[0]):
                if self.regions[nbr].depth < 0:
                    self.regions[nbr].depth = self.regions[r].depth + 1
                    self.regions[nbr].parent = r
                    self.regions[nbr].link = lid
                    queue.append(nbr)


def _adjacent_pairs(la, lb):
    """Paires (a, b) d'étiquettes > 0 en contact 4-connexe entre deux images d'étiquettes."""
    out = []
    for x, y in ((la[:, :-1], lb[:, 1:]), (la[:, 1:], lb[:, :-1]), (la[:-1, :], lb[1:, :]), (la[1:, :], lb[:-1, :])):
        mk = (x > 0) & (y > 0)
        if mk.any():
            out.append(np.stack((x[mk], y[mk]), 1))
    return np.unique(np.concatenate(out), axis=0) if out else np.zeros((0, 2), int)
