"""Fenêtre de debug : la mémoire interne (WorldModel) de chaque drone.

Une fenêtre pyqtgraph (Qt) séparée de la fenêtre du simulateur (celle-ci n'est pas modifiée) :
  - grand panneau "Essaim" : union des couches PROPRES de tous les drones (= ce que l'essaim saurait avec une
    radio parfaite), routes et cibles de tous les drones, poses estimées reliées à la vraie position (trait = erreur),
    vrais blessés (cercles noirs) et blessés crus par les drones ;
  - un petit panneau par drone : SA carte fusionnée (sa couche + tuiles reçues), ses objets, sa route.

Légende : carte blanc = libre, noir = occupé, gris = inconnu.
  point gris = vraie position, cercle coloré = pose estimée, trait = route, croix = cible de mission,
  rouge/orange/vert = blessé vu/porté/sauvé, carré bleu = centre de secours, triangle violet = zone de retour.

Utilise la vérité terrain uniquement pour l'affichage. Aucun effet sur le comportement des drones.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List, Optional

import numpy as np

from .swarm.gridmap import CLIP, FREE_THR, OCC_THR
from .swarm.schema import MissionKind, ObjKind, ObjState

_ROLE = {MissionKind.EXPLORE: "M", MissionKind.FOLLOW: "S", MissionKind.RELAY: "R", MissionKind.GOTO_WOUNDED: "B",
         MissionKind.CARRY: "P", MissionKind.RETURN: "H", MissionKind.IDLE: "I", MissionKind.RECEIVE: "T"}
_STATE_COLOR = {ObjState.SEEN: (230, 40, 40), ObjState.CARRIED: (255, 150, 0), ObjState.RESCUED: (40, 180, 60)}
_SMALL_COLS = 5


def _palette(n: int):
    import pyqtgraph as pg
    return [pg.intColor(i, hues=max(n, 1), values=1, maxValue=220) for i in range(n)]


def _to_rgb(view: np.ndarray, thr_free: float = FREE_THR, thr_occ: float = OCC_THR) -> np.ndarray:
    """log-odds (W, H) -> image RGB uint8 (W, H, 3) : libre blanc, occupé noir, inconnu gris."""
    v = view.astype(np.float32)
    g = np.full(v.shape, 150.0, np.float32)
    free = v < 0
    occ = v > 0
    g[free] = 150.0 + 105.0 * np.clip(v[free] / thr_free, 0.0, 1.0)
    g[occ] = 150.0 - 150.0 * np.clip(v[occ] / thr_occ, 0.0, 1.0)
    return np.repeat(g.astype(np.uint8)[:, :, None], 3, axis=2)


def _object_spots(objects):
    import pyqtgraph as pg
    spots = []
    for o in objects:
        if o.kind == ObjKind.WOUNDED:
            spots.append({"pos": o.pos, "brush": pg.mkBrush(*_STATE_COLOR[o.state]), "symbol": "o"})
        elif o.kind == ObjKind.RESCUE_CENTER:
            spots.append({"pos": o.pos, "brush": pg.mkBrush(40, 90, 230), "symbol": "s", "size": 12})
        elif o.kind == ObjKind.RETURN_AREA:
            spots.append({"pos": o.pos, "brush": pg.mkBrush(150, 90, 230), "symbol": "t", "size": 12})
    return spots


class _Panel:
    def __init__(self, plot, rect, shape, color):
        import pyqtgraph as pg
        self.plot = plot
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMenuEnabled(False)
        self.img = pg.ImageItem(np.full((*shape, 3), 150, np.uint8), axisOrder="col-major", levels=(0, 255))
        self.img.setRect(rect)                    # repère monde (px), origine au centre de la carte
        plot.addItem(self.img)
        self.route = plot.plot([], [], pen=pg.mkPen(color, width=2))
        self.target = pg.ScatterPlotItem(size=14, symbol="x", pen=pg.mkPen(color, width=3), brush=None)
        self.objects = pg.ScatterPlotItem(size=9, pen=None)
        self.truth = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(90, 90, 90, 200))
        self.est = pg.ScatterPlotItem(size=14, pen=pg.mkPen(color, width=3), brush=None)
        for it in (self.objects, self.truth, self.target, self.est):
            plot.addItem(it)


class _BigPanel:
    def __init__(self, plot, rect, shape, colors):
        import pyqtgraph as pg
        self.plot = plot
        plot.setAspectLocked(True)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMenuEnabled(False)
        self.img = pg.ImageItem(np.full((*shape, 3), 150, np.uint8), axisOrder="col-major", levels=(0, 255))
        self.img.setRect(rect)
        plot.addItem(self.img)
        self.routes = [plot.plot([], [], pen=pg.mkPen(c, width=2)) for c in colors]
        self.links = plot.plot([], [], pen=pg.mkPen((0, 170, 0, 160), width=1), connect="pairs")
        self.errors = plot.plot([], [], pen=pg.mkPen((220, 0, 0), width=2), connect="pairs")
        self.slots = pg.ScatterPlotItem(size=13, pen=pg.mkPen((120, 0, 160), width=2), brush=None)
        self.true_wounded = pg.ScatterPlotItem(size=20, symbol="o", pen=pg.mkPen((0, 0, 0), width=2), brush=None)
        self.objects = pg.ScatterPlotItem(size=10, pen=None)
        self.targets = pg.ScatterPlotItem(size=16, symbol="x", pen=None)
        self.truth = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(90, 90, 90, 220))
        self.est = pg.ScatterPlotItem(size=16, brush=None)
        self.labels = []
        for it in (self.true_wounded, self.slots, self.objects, self.truth, self.targets, self.est):
            plot.addItem(it)
        for c in colors:
            lab = pg.TextItem("", color=c, anchor=(0, 1))
            plot.addItem(lab)
            self.labels.append(lab)


class SwarmDebugView:
    """Singleton piloté par le premier drone (cf. SwarmDrone.display)."""

    def __init__(self, period: int = 10):
        self.period = period
        self._key = None
        self._win = None
        self._panels: List[_Panel] = []
        self._big: Optional[_BigPanel] = None
        self._colors = []
        self._app = None

    # ------------------------------------------------------------------ construction
    def _build(self, drones) -> None:
        if sys.platform == "win32":
            # Sans ça, Qt rend le processus "DPI-aware" et la fenêtre du simulateur (pyglet), jusque-là agrandie
            # par la mise à l'échelle de Windows, rétrécit d'un coup. À fixer avant le premier import de Qt.
            os.environ.setdefault("QT_QPA_PLATFORM", "windows:dpiawareness=0")
        import pyqtgraph as pg
        from PySide6.QtCore import QRectF
        from PySide6.QtWidgets import QApplication

        self._app = QApplication.instance() or QApplication([])
        if self._win is None:
            self._win = pg.GraphicsLayoutWidget(title="Swarm - cartes internes des drones")
            self._win.resize(1400, 950)
            self._win.setBackground("w")
        self._win.clear()
        m = drones[0]._brain.world.map
        rect = QRectF(-m.half[0], -m.half[1], m.W * m.res, m.H * m.res)
        shape = (m.W, m.H)
        self._colors = _palette(len(drones))
        ncol = min(_SMALL_COLS, len(drones))
        lay = self._win.ci
        self._big = _BigPanel(lay.addPlot(row=0, col=0, colspan=ncol, title="Essaim"), rect, shape, self._colors)
        self._panels = []
        for i, d in enumerate(drones):
            plot = lay.addPlot(row=1 + i // ncol, col=i % ncol, title=f"D{d.identifier}")
            self._panels.append(_Panel(plot, rect, shape, self._colors[i]))
        nrows_small = 1 + (len(drones) - 1) // ncol
        lay.layout.setRowStretchFactor(0, 2 * nrows_small)     # le grand panneau prend ~2/3 de la hauteur
        for r in range(1, 1 + nrows_small):
            lay.layout.setRowStretchFactor(r, 1)
        self._win.show()

    # ------------------------------------------------------------------ mise à jour
    def update(self, drones, t: int) -> None:
        drones = [d for d in drones if getattr(d, "_brain", None) is not None]
        if not drones or t % self.period:
            return
        key = tuple(id(d) for d in drones)
        if key != self._key:                      # nouveau round (nouvelles instances) : on reconstruit
            self._key = key
            self._build(drones)
        if not self._win.isVisible():             # fenêtre fermée par l'utilisateur : on arrête de dessiner
            self._app.processEvents()
            return
        alive = [d for d in drones if not d.removed]
        truth = np.array([d.true_position() for d in alive]).reshape(-1, 2)
        self._draw_big(drones, alive, truth, t)
        for panel, d in zip(self._panels, drones):
            self._draw_drone(panel, d, truth)
        self._app.processEvents()

    def _draw_big(self, drones, alive, truth, t: int) -> None:
        import pyqtgraph as pg
        big = self._big
        acc = None                                 # union : somme des couches propres
        for d in drones:
            own = d._brain.world.map.layers.get(d._brain.me)
            if own is not None:
                acc = own.astype(np.int16) if acc is None else acc + own
        if acc is not None:
            big.img.setImage(_to_rgb(np.clip(acc, -3 * CLIP, 3 * CLIP)), autoLevels=False)

        est, est_pens, err_seg, tgts, objects = [], [], [], [], []
        missions = Counter()
        wounded_targets = Counter()
        for i, d in enumerate(drones):
            b, col = d._brain, self._colors[i]
            r = b.route
            if r is not None and len(r.pts) and not d.removed:
                big.routes[i].setData(r.pts[:, 0], r.pts[:, 1])
            else:
                big.routes[i].setData([], [])
            if d.removed:
                big.labels[i].setText("")
                continue
            p = b.ekf.pose().pos
            est.append(p)
            est_pens.append(pg.mkPen(col, width=3))
            err_seg += [p, d.true_position()]
            big.labels[i].setText(f"{d.identifier}{_ROLE.get(b.mission.kind, '?')}")
            big.labels[i].setPos(float(p[0]) + 6, float(p[1]) + 6)
            missions[b.mission.kind.name] += 1
            if b.mission.target is not None:
                tgts.append({"pos": b.mission.target, "pen": pg.mkPen(col, width=3)})
                if b.mission.kind in (MissionKind.GOTO_WOUNDED,):
                    wounded_targets[(round(b.mission.target[0] / 40), round(b.mission.target[1] / 40))] += 1
            objects += b.world.objects_view()
        big.est.setData(pos=est, pen=est_pens) if est else big.est.setData([])
        seg = np.array(err_seg).reshape(-1, 2)
        big.errors.setData(seg[:, 0], seg[:, 1]) if len(seg) else big.errors.setData([], [])
        big.targets.setData(tgts)
        big.truth.setData(pos=truth)
        big.objects.setData(_object_spots(objects))
        big.true_wounded.setData(pos=self._true_wounded(drones))

        # liens radio réels (vérité simulateur) et postes prévus par le plan d'équipe du premier drone vivant
        seg_links = []
        for d in alive:
            for c in d.communicator.comms_in_range:
                o = c.agent
                if id(d) < id(o) and not o.removed and not d.communicator_is_disabled()                         and not o.communicator_is_disabled():
                    seg_links += [d.true_position(), o.true_position()]
        seg_links = np.array(seg_links).reshape(-1, 2)
        big.links.setData(seg_links[:, 0], seg_links[:, 1]) if len(seg_links) else big.links.setData([], [])
        plan = alive[0]._brain.team.last if alive else None
        spots = []
        if plan is not None:
            for s in plan.slots:
                if s.kind in ("relay", "base"):
                    spots.append({"pos": s.xy, "symbol": "d" if s.kind == "relay" else "star"})
        big.slots.setData(spots)

        errs = [float(np.linalg.norm(d._brain.ekf.pose().pos - d.true_position())) for d in alive]
        carriers = sum(1 for d in alive if d.grasped_wounded_persons())
        dup = sum(n - 1 for n in wounded_targets.values() if n > 1)
        mis = "  ".join(f"{_ROLE.get(MissionKind[k], k)}:{v}" for k, v in sorted(missions.items()))
        big.plot.setTitle(f"pas {t} | vivants {len(alive)}/{len(drones)} | porteurs {carriers} | "
                          f"doublons de cible {dup} | err pose moy {np.mean(errs) if errs else 0:.1f}px "
                          f"max {max(errs, default=0):.1f}px | {mis}", size="10pt")

    @staticmethod
    def _true_wounded(drones):
        for d in drones:
            pg = d.playground
            if pg is not None:
                from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
                pts = [np.array(e.position) for e in pg.elements if isinstance(e, WoundedPerson) and not e.removed]
                return np.array(pts).reshape(-1, 2)
        return np.zeros((0, 2))

    def _draw_drone(self, panel: _Panel, d, truth) -> None:
        b = d._brain
        w = b.world
        view = w.map.view()
        panel.img.setImage(_to_rgb(view), autoLevels=False)
        pose = b.ekf.pose()
        panel.est.setData(pos=[pose.pos])
        panel.truth.setData(pos=truth)
        if b.route is not None and len(b.route.pts):
            panel.route.setData(b.route.pts[:, 0], b.route.pts[:, 1])
        else:
            panel.route.setData([], [])
        tgt = b.mission.target
        panel.target.setData(pos=[tgt] if tgt is not None else [])
        panel.objects.setData(_object_spots(w.objects_view()))
        sub = view[:w.map.w0, :w.map.h0]
        known = float(np.mean((sub >= OCC_THR) | (sub <= FREE_THR)))
        err = float(np.linalg.norm(pose.pos - d.true_position())) if not d.removed else float("nan")
        state = "MORT" if d.removed else b.mission.kind.name[:4]
        carry = " +blessé" if b._was_carrying else ""
        panel.plot.setTitle(f"D{d.identifier} {state}{carry} | {100 * known:.0f}% | err {err:.0f} | pv {d.drone_health}",
                            size="8pt")
