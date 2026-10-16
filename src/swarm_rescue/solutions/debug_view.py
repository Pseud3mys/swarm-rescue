"""Fenêtre de debug : la mémoire interne (WorldModel) de chaque drone.

Une fenêtre pyqtgraph (Qt) séparée de la fenêtre du simulateur (celle-ci n'est pas modifiée) :
  - grand panneau "Essaim" : union des couches PROPRES de tous les drones (= ce que l'essaim saurait avec une
    radio parfaite), liens radio réels, routes et cibles, poses estimées reliées à la vraie position (trait rouge
    = erreur), vrais blessés (cercles noirs) et blessés tels que l'essaim les croit (fusion de toutes les pistes) ;
  - un petit panneau par drone : SA carte (sa couche + tuiles reçues), ses objets, sa route.
Couleur = mission (légende sous le grand panneau).

Performance : un seul morceau de dessin par pas de simulation (grand panneau un pas sur trois, petits panneaux
à tour de rôle), carte affichée via une table de couleurs (pas de conversion RVB), pinceaux mis en cache.
Utilise la vérité terrain uniquement pour l'affichage. Aucun effet sur le comportement des drones.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List, Optional

import numpy as np

from .swarm.gridmap import FREE_THR, OCC_THR, VIEW_CLIP
from .swarm.schema import MissionKind, ObjKind, ObjState
from .swarm.topo import CORRIDOR
from .swarm.world import fuse_tracks

# mission -> (lettre, couleur, libellé)
_ROLE = {
    MissionKind.EXPLORE: ("M", (30, 100, 235), "meneur / explore"),
    MissionKind.FOLLOW: ("S", (0, 175, 200), "suiveur"),
    MissionKind.RELAY: ("R", (150, 60, 200), "relais / base"),
    MissionKind.GOTO_WOUNDED: ("B", (225, 30, 30), "va chercher un blessé"),
    MissionKind.CARRY: ("P", (245, 140, 0), "porte un blessé"),
    MissionKind.RECEIVE: ("T", (0, 160, 60), "prend le relais"),
    MissionKind.RETURN: ("H", (120, 120, 120), "retour"),
    MissionKind.IDLE: ("I", (120, 120, 120), "sans tâche"),
    MissionKind.DROP: ("D", (200, 120, 40), "dépose un blessé à la porte"),
}
_DEAD = (60, 60, 60)
_W_COLOR = {ObjState.SEEN: (230, 30, 30), ObjState.CARRIED: (245, 140, 0)}


def _lut() -> np.ndarray:
    """Table de couleurs pour la vue log-odds décalée de VIEW_CLIP (index 0..2*VIEW_CLIP)."""
    v = np.arange(-VIEW_CLIP, VIEW_CLIP + 1, dtype=np.float32)
    g = np.full(v.shape, 150.0, np.float32)
    g[v < 0] = 150.0 + 105.0 * np.clip(v[v < 0] / FREE_THR, 0.0, 1.0)
    g[v > 0] = 150.0 - 150.0 * np.clip(v[v > 0] / OCC_THR, 0.0, 1.0)
    g = g.astype(np.uint8)
    return np.stack([g, g, g], axis=1)


def _topo_lut() -> np.ndarray:
    """0 transparent, 1..20 pièces (pastels), 21 porte (jaune), 22 couloir (orange)."""
    lut = np.zeros((256, 4), np.uint8)
    rng = np.random.RandomState(7)
    lut[1:21, :3] = rng.randint(90, 230, (20, 3))
    lut[1:21, 3] = 70
    lut[21] = (255, 215, 0, 150)
    lut[22] = (255, 120, 0, 120)
    return lut


def _index(view: np.ndarray) -> np.ndarray:
    return (np.clip(view, -VIEW_CLIP, VIEW_CLIP) + VIEW_CLIP).astype(np.uint8)


class _Panel:
    def __init__(self, plot, rect, shape, lut, big: bool = False):
        import pyqtgraph as pg
        self.plot = plot
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMenuEnabled(False)
        self.img = pg.ImageItem(np.full(shape, VIEW_CLIP, np.uint8), axisOrder="col-major")
        self.img.setLookupTable(lut)
        self.img.setLevels((0, 2 * VIEW_CLIP))
        self.img.setRect(rect)                    # repère monde (px), origine au centre de la carte
        plot.addItem(self.img)
        self.topo = None
        self.tree = None
        if big:                                   # surcouche topologique (pièces / portes / couloirs + arbre)
            self.topo = pg.ImageItem(np.zeros(shape, np.uint8), axisOrder="col-major")
            self.topo.setLookupTable(_topo_lut())
            self.topo.setLevels((0, 255))
            self.topo.setRect(rect)
            plot.addItem(self.topo)
            self.tree = plot.plot([], [], pen=pg.mkPen((0, 60, 220, 170), width=2), connect="pairs")
            self.doors = pg.ScatterPlotItem(size=9, pen=None, brush=pg.mkBrush(230, 50, 50))
            plot.addItem(self.doors)
        plot.setRange(rect, padding=0.0)          # cadrage fixe : le recadrage auto à chaque mise à jour coûte cher
        plot.disableAutoRange()
        w = 2 if big else 1
        self.links = plot.plot([], [], pen=pg.mkPen((0, 170, 0, 140), width=1), connect="pairs") if big else None
        self.routes = pg.PlotDataItem([], [], connect="finite", pen=pg.mkPen((60, 60, 60, 160), width=w))
        self.errors = plot.plot([], [], pen=pg.mkPen((220, 0, 0), width=2), connect="pairs") if big else None
        self.slots = pg.ScatterPlotItem(size=11, pen=pg.mkPen((150, 60, 200), width=2), brush=None) if big else None
        self.true_wounded = pg.ScatterPlotItem(size=18, symbol="o", pen=pg.mkPen((0, 0, 0), width=2), brush=None) \
            if big else None
        self.objects = pg.ScatterPlotItem(size=11 if big else 8, pen=pg.mkPen((0, 0, 0), width=1))
        self.targets = pg.ScatterPlotItem(size=14 if big else 10, symbol="x")
        self.truth = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(70, 70, 70, 200))
        self.est = pg.ScatterPlotItem(size=16 if big else 12, brush=None)
        plot.addItem(self.routes)
        for it in (self.true_wounded, self.slots, self.objects, self.truth, self.targets, self.est):
            if it is not None:
                plot.addItem(it)
        self.labels = []


class SwarmDebugView:
    """Singleton piloté par le premier drone (cf. SwarmDrone.display), appelé à chaque pas."""

    def __init__(self, period: int = 1):
        self.period = max(int(period), 1)
        self._key = None
        self._win = None
        self._app = None
        self._big: Optional[_Panel] = None
        self._panels: List[_Panel] = []
        self._job = 0
        self._pens = {}
        self._brushes = {}
        self._topo_t = None

    # ------------------------------------------------------------------ cache Qt
    def _pen(self, color, width=3):
        import pyqtgraph as pg
        k = (color, width)
        if k not in self._pens:
            self._pens[k] = pg.mkPen(color, width=width)
        return self._pens[k]

    def _brush(self, color):
        import pyqtgraph as pg
        if color not in self._brushes:
            self._brushes[color] = pg.mkBrush(*color)
        return self._brushes[color]

    # ------------------------------------------------------------------ construction
    def _build(self, drones) -> None:
        if sys.platform == "win32":
            # Sans ça, Qt rend le processus "DPI-aware" et la fenêtre du simulateur (pyglet), jusque-là agrandie
            # par la mise à l'échelle de Windows, rétrécit d'un coup. À fixer avant la création de l'application Qt.
            os.environ.setdefault("QT_QPA_PLATFORM", "windows:dpiawareness=0")
        import pyqtgraph as pg
        from PySide6.QtCore import QRectF
        from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QWidget

        self._app = QApplication.instance() or QApplication([])
        geom = None
        if self._win is not None:                 # nouveau round : on remplace la fenêtre au même endroit
            geom = self._win.geometry()
            self._win.close()
        # Un widget de tracé PAR panneau : Qt ne repeint que celui qui a changé (avec un seul
        # GraphicsLayoutWidget, toute la fenêtre était repeinte à chaque mise à jour : ~12 ms).
        self._win = QWidget()
        self._win.setWindowTitle("Swarm - cartes internes des drones")
        self._win.setStyleSheet("background: white;")
        grid = QGridLayout(self._win)
        grid.setSpacing(2)
        m = drones[0]._brain.world.map
        rect = QRectF(-m.half[0], -m.half[1], m.W * m.res, m.H * m.res)
        shape, lut = (m.W, m.H), _lut()
        ncol = min(5, len(drones))

        def plot_widget(title):
            pw = pg.PlotWidget(background="w", title=title)
            return pw, pw.getPlotItem()

        pw, item = plot_widget("Essaim")
        grid.addWidget(pw, 0, 0, 1, ncol)
        self._big = _Panel(item, rect, shape, lut, big=True)
        for d in drones:
            lab = pg.TextItem("", anchor=(0, 1))
            self._big.plot.addItem(lab)
            self._big.labels.append(lab)
        legend = "  ".join(f"<span style='color:rgb{c}'>&#9679; {l} {txt}</span>" for l, c, txt in
                           (_ROLE[k] for k in (MissionKind.EXPLORE, MissionKind.FOLLOW, MissionKind.RELAY,
                                               MissionKind.GOTO_WOUNDED, MissionKind.CARRY, MissionKind.RECEIVE)))
        legend += ("  &nbsp;|&nbsp; <b>O</b> vrai blessé &nbsp; <span style='color:rgb(230,30,30)'>&#9679;</span> blessé cru"
                   " &nbsp; &#9671; poste relais/base &nbsp; trait rouge = erreur de pose &nbsp; vert = lien radio"
                   " &nbsp;|&nbsp; <span style='color:rgb(230,50,50)'>&#9679;</span> passage/porte"
                   " <span style='background:rgb(255,160,60)'>couloir</span> pièces en pastel, arbre en bleu")
        lab = QLabel(legend)
        grid.addWidget(lab, 1, 0, 1, ncol)
        self._panels = []
        for i, d in enumerate(drones):
            pw, item = plot_widget(f"D{d.identifier}")
            grid.addWidget(pw, 2 + i // ncol, i % ncol)
            self._panels.append(_Panel(item, rect, shape, lut))
        nrows_small = 1 + (len(drones) - 1) // ncol
        grid.setRowStretch(0, 2 * nrows_small)
        for r in range(2, 2 + nrows_small):
            grid.setRowStretch(r, 1)
        if geom is not None:
            self._win.setGeometry(geom)
        else:
            self._win.resize(1400, 950)
        self._win.show()
        self._job = 0

    # ------------------------------------------------------------------ mise à jour
    def update(self, drones, t: int) -> None:
        drones = [d for d in drones if getattr(d, "_brain", None) is not None]
        if not drones:
            return
        key = tuple(id(d) for d in drones)
        if key != self._key:                      # nouveau round (nouvelles instances) : on reconstruit
            self._key = key
            self._build(drones)
        if t % self.period == 0 and self._win.isVisible():
            # un seul morceau de dessin par appel : grand panneau 1 fois sur 3, petits panneaux à tour de rôle
            if self._job % 3 == 0:
                self._draw_big(drones, t)
            else:
                k = (self._job // 3) * 2 + (self._job % 3) - 1
                i = k % len(drones)
                self._draw_drone(self._panels[i], drones[i])
            self._job += 1
        self._app.processEvents()                 # fenêtre réactive à chaque pas (pas seulement quand on dessine)

    @staticmethod
    def _role(d):
        if d.removed:
            return "X", _DEAD
        l, c, _ = _ROLE.get(d._brain.mission.kind, ("?", (120, 120, 120), ""))
        return l, c

    def _draw_big(self, drones, t: int) -> None:
        big = self._big
        alive = [d for d in drones if not d.removed]
        acc = None                                 # union : somme des couches propres
        for d in drones:
            own = d._brain.world.map.layers.get(d._brain.me)
            if own is not None:
                acc = own.astype(np.int16) if acc is None else acc + own
        if acc is not None:
            big.img.setImage(_index(acc), autoLevels=False)

        rx, ry, est, est_pens, err, tgts = [], [], [], [], [], []
        for d, lab in zip(drones, big.labels):
            b = d._brain
            if d.removed:
                lab.setText("")
                continue
            letter, col = self._role(d)
            p = b.ekf.pose().pos
            est.append(p)
            est_pens.append(self._pen(col))
            err += [p, d.true_position()]
            txt = f"{d.identifier}{letter}"
            if getattr(lab, "_txt", None) != txt:     # réécrire un texte Qt coûte cher : seulement s'il change
                lab.setText(txt, color=col)
                lab._txt = txt
            lab.setPos(float(p[0]) + 7, float(p[1]) + 7)
            if b.route is not None and len(b.route.pts):
                rx += list(b.route.pts[:, 0]) + [np.nan]
                ry += list(b.route.pts[:, 1]) + [np.nan]
            if b.mission.target is not None:
                tgts.append({"pos": b.mission.target, "pen": self._pen(col), "brush": None})
        big.routes.setData(rx, ry)
        big.est.setData(pos=est, pen=est_pens) if est else big.est.setData([])
        seg = np.array(err).reshape(-1, 2)
        big.errors.setData(seg[:, 0], seg[:, 1]) if len(seg) else big.errors.setData([], [])
        big.targets.setData(tgts)
        truth = np.array([d.true_position() for d in alive]).reshape(-1, 2)
        big.truth.setData(pos=truth)
        big.objects.setData(self._swarm_belief(drones))
        big.true_wounded.setData(pos=self._true_wounded(drones))

        seg = []
        for d in alive:
            if d.communicator_is_disabled():
                continue
            for c in d.communicator.comms_in_range:
                o = c.agent
                if id(d) < id(o) and not o.removed and not o.communicator_is_disabled():
                    seg += [d.true_position(), o.true_position()]
        seg = np.array(seg).reshape(-1, 2)
        big.links.setData(seg[:, 0], seg[:, 1]) if len(seg) else big.links.setData([], [])
        plan = alive[0]._brain.team.last if alive else None
        if plan is not None and plan.topo is not None and plan.t != self._topo_t:
            self._topo_t = plan.t                 # la topologie ne change qu'aux replanifications
            tp = plan.topo
            idx = np.zeros(tp.labels.shape, np.uint8)
            seg = []
            for r in tp.regions.values():
                idx[tp.labels == r.rid] = 22 if r.kind == CORRIDOR else 1 + r.rid % 20
                if r.parent >= 0:
                    lp = tp.links[r.link].pos
                    seg += [r.center, lp, lp, tp.regions[r.parent].center]
            big.topo.setImage(idx, autoLevels=False)
            big.doors.setData(pos=[l.pos for l in tp.links])           # passages (portes) = arêtes du graphe
            seg = np.array(seg).reshape(-1, 2)
            big.tree.setData(seg[:, 0], seg[:, 1]) if len(seg) else big.tree.setData([], [])
        big.slots.setData([{"pos": s.xy, "symbol": "d" if s.kind == "relay" else "star"}
                           for s in (plan.slots if plan else []) if s.kind in ("relay", "base")])

        errs = [float(np.linalg.norm(d._brain.ekf.pose().pos - d.true_position())) for d in alive]
        roles = Counter(self._role(d)[0] for d in alive)
        carriers = sum(1 for d in alive if d.grasped_wounded_persons())
        big.plot.setTitle(f"pas {t} | vivants {len(alive)}/{len(drones)} | porteurs {carriers} | "
                          f"err pose moy {np.mean(errs) if errs else 0:.1f}px max {max(errs, default=0):.1f}px | "
                          + " ".join(f"{k}:{v}" for k, v in sorted(roles.items())), size="10pt")

    def _swarm_belief(self, drones) -> list:
        """Blessés tels que l'essaim les croit : toutes les pistes de tous les drones (révision max par clé),
        regroupées et fusionnées comme le fait chaque drone. Les blessés sauvés ne sont pas affichés."""
        merged = {}
        for d in drones:
            for k, o in d._brain.world.objects.items():
                if o.kind == ObjKind.WOUNDED and (k not in merged or o.rev > merged[k].rev):
                    merged[k] = o
        spots = []
        for o in fuse_tracks(list(merged.values())):
            if o.state in _W_COLOR:
                spots.append({"pos": o.pos, "brush": self._brush(_W_COLOR[o.state]), "symbol": "o"})
        return spots

    @staticmethod
    def _true_wounded(drones):
        for d in drones:
            pg = d.playground
            if pg is not None:
                from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
                pts = [np.array(e.position) for e in pg.elements if isinstance(e, WoundedPerson) and not e.removed]
                return np.array(pts).reshape(-1, 2)
        return np.zeros((0, 2))

    def _draw_drone(self, panel: _Panel, d) -> None:
        b = d._brain
        w = b.world
        view = w.map.view()
        panel.img.setImage(_index(view), autoLevels=False)
        letter, col = self._role(d)
        pose = b.ekf.pose()
        panel.est.setData(pos=[pose.pos], pen=[self._pen(col)])
        panel.truth.setData(pos=[d.true_position()] if not d.removed else [])
        if b.route is not None and len(b.route.pts):
            panel.routes.setData(b.route.pts[:, 0], b.route.pts[:, 1])
        else:
            panel.routes.setData([], [])
        tgt = b.mission.target
        panel.targets.setData([{"pos": tgt, "pen": self._pen(col)}] if tgt is not None else [])
        spots = []
        for o in w.objects_view():
            if o.kind == ObjKind.WOUNDED and o.state in _W_COLOR:
                spots.append({"pos": o.pos, "brush": self._brush(_W_COLOR[o.state]), "symbol": "o"})
            elif o.kind == ObjKind.RESCUE_CENTER:
                spots.append({"pos": o.pos, "brush": self._brush((40, 90, 230)), "symbol": "s"})
        panel.objects.setData(spots)
        sub = view[:w.map.w0, :w.map.h0]
        known = float(np.mean((sub >= OCC_THR) | (sub <= FREE_THR)))
        err = float(np.linalg.norm(pose.pos - d.true_position())) if not d.removed else float("nan")
        name = "MORT" if d.removed else _ROLE.get(b.mission.kind, ("?", None, "?"))[2]
        panel.plot.setTitle(f"<span style='color:rgb{col}'>D{d.identifier} {letter}</span> {name} | "
                            f"{100 * known:.0f}% | err {err:.0f} | pv {d.drone_health}", size="8pt")
