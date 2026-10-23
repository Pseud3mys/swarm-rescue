"""Debug : trace d'un run dans debug_export/ (écrasé à chaque run : toujours le DERNIER run).

Activé en mode debug (SwarmDrone.DEBUG_VIEW, désactivé par SWARM_DEBUG=0). Niveau de détail : SWARM_TRACE=0|1|2
(défaut 1) :
  0 : cartes finales seulement ;
  1 : + journal (changements de rôle, blessés découverts), positions tous les 10 pas, carte d'un drone à chaque
      changement de rôle (au plus une tous les 60 pas par drone), planche de tous les drones tous les 500 pas ;
  2 : positions tous les 2 pas, carte à chaque changement de rôle, planche tous les 100 pas, changements de cible.

debug_export/
  final/all.png, final/drone_<id>.png, final/drone_<id>.npz   cartes à la fin (appel de dump_maps par launcher.py)
  snapshots/t<pas>_d<id>_<ancien>-<nouveau>.png               carte d'un drone au moment où il change de rôle
  sheets/t<pas>.png                                           tous les drones à un instant
  timeline.txt                                                journal lisible
  events.jsonl                                                le même journal, une ligne JSON par évènement
  positions.csv                                               t, drone, x, y, cap, mission, origine, cible, porte

Image d'un drone (x vers la droite, y vers le haut) : blanc = libre, noir = mur, gris = inconnu, rouge = frontières,
vert plein = vrais blessés, cercle magenta = blessés connus du drone (pâle : pris / sauvés), bleu = le drone et un trait
vers sa cible, bleu clair = les autres drones (positions qu'il connaît).
"""
from __future__ import annotations

import json
import os
import shutil

import cv2
import numpy as np

from .swarm.pathing import PlanGrid
from .swarm.schema import ObjKind, ObjState

ROOT = "debug_export"
_LEVELS = {  # pos_period, snap_gap, sheet_period, cible
    0: (0, None, 0, False),
    1: (10, 60, 500, False),
    2: (2, 0, 100, True),
}


def _truth_from_playground(pg) -> list:
    out = []
    for e in getattr(pg, "elements", []) or []:
        if type(e).__name__ == "WoundedPerson":
            try:
                out.append(np.array(e.position, float)[:2])
            except Exception:
                pass
    return out


def _render(b, truth: list):
    """(image BGR, grilles brutes) de la carte interne du drone `b` (Brain)."""
    m = b.world.map
    pose = b.ekf.pose()
    G = PlanGrid(b.world, b.p, pose.pos)
    G.solve()
    fr = G.frontiers(b.p.frontier_min_cells)
    v = m.view()
    front = np.zeros(v.shape, bool)
    for f in fr:
        front[f.cells[0], f.cells[1]] = True

    img = np.full(v.shape + (3,), 128, np.uint8)
    img[v <= -3] = (255, 255, 255)
    img[v >= 5] = (0, 0, 0)
    img[front] = (0, 0, 255)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    def px(xy):
        ix, iy = m.world_to_cell(np.asarray(xy, float))
        return int(iy[0]) * 4 + 2, int(ix[0]) * 4 + 2          # (colonne, ligne) avant retournement

    for tp in truth:
        cv2.circle(img, px(tp), 6, (0, 200, 0), -1)
    known = list(b.world.objects_view(ObjKind.WOUNDED))
    for o in known:
        cv2.circle(img, px(o.pos), 9, (255, 0, 255) if o.state == ObjState.SEEN else (200, 150, 200), 2)
    for a in b.world.others():
        cv2.circle(img, px(a.pos), 4, (255, 200, 120), -1)
    if b.mission.target is not None:
        cv2.line(img, px(pose.pos), px(b.mission.target), (255, 0, 0), 1)
    cv2.circle(img, px(pose.pos), 6, (255, 0, 0), -1)
    img = np.ascontiguousarray(np.transpose(img, (1, 0, 2))[::-1])
    cv2.putText(img, f"drone {b.me}  {b.mission.kind.name} ({b.mission_src})", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    arrays = dict(view=v.copy(), frontiers=front,
                  pose=np.r_[pose.pos, pose.theta], res=m.res, half=m.half,
                  wounded_known=np.array([o.pos for o in known], float).reshape(-1, 2),
                  wounded_truth=np.array(truth, float).reshape(-1, 2))
    return img, arrays


def _sheet(ims: list):
    cols = min(5, len(ims))
    ims = ims + [np.zeros_like(ims[0])] * (-len(ims) % cols)
    return np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])


class Recorder:
    """Une instance par run (playground) : appelée par chaque drone après son pas (SwarmDrone.control)."""

    def __init__(self, pg, root: str = ROOT):
        self.pg = pg
        self.root = root
        lvl = int(os.environ.get("SWARM_TRACE", "1"))
        self.pos_period, self.snap_gap, self.sheet_period, self.track_target = _LEVELS.get(lvl, _LEVELS[1])
        shutil.rmtree(root, ignore_errors=True)
        for sub in ("final", "snapshots", "sheets"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        self._events = open(os.path.join(root, "events.jsonl"), "w", encoding="utf-8")
        self._timeline = open(os.path.join(root, "timeline.txt"), "w", encoding="utf-8")
        self._pos = open(os.path.join(root, "positions.csv"), "w", encoding="utf-8")
        self._pos.write("t,drone,x,y,theta,mission,src,tx,ty,carrying\n")
        self._role: dict = {}            # drone -> (kind, src)
        self._target: dict = {}
        self._last_snap: dict = {}
        self._known: dict = {}           # drone -> nb de blessés connus (libres)
        self._brains: dict = {}
        self._sheet_t = -1

    def _log(self, t, d, kind, msg, **data):
        self._events.write(json.dumps(dict(t=t, drone=d, event=kind, **data), default=float) + "\n")
        self._timeline.write(f"t={t:5d}  drone {d}  {msg}\n")

    def observe(self, b, t: int) -> None:
        self._brains[b.me] = b
        pose = b.ekf.pose()
        tgt = b.mission.target
        role = (b.mission.kind.name, b.mission_src)
        old = self._role.get(b.me)
        if old != role:
            self._role[b.me] = role
            where = f"({pose.x:.0f}, {pose.y:.0f})"
            cible = f" cible ({tgt[0]:.0f}, {tgt[1]:.0f})" if tgt is not None else ""
            self._log(t, b.me, "role", f"{old[0] + '/' + old[1] if old else '-'} -> {role[0]}/{role[1]} "
                      f"en {where}{cible}  [raison : {b.last_reason}]",
                      old=old, new=role, pos=list(pose.pos), target=tgt, reason=b.last_reason)
            if old is not None and self.snap_gap is not None \
                    and t - self._last_snap.get(b.me, -10 ** 9) >= self.snap_gap:
                self._last_snap[b.me] = t
                self._snapshot(b, t, f"{old[0]}-{role[0]}")
        elif self.track_target and tgt is not None:
            prev = self._target.get(b.me)
            if prev is None or np.hypot(tgt[0] - prev[0], tgt[1] - prev[1]) > 60.0:
                self._log(t, b.me, "target", f"{role[0]} nouvelle cible ({tgt[0]:.0f}, {tgt[1]:.0f})",
                          target=tgt, pos=list(pose.pos))
        self._target[b.me] = tgt
        if self.pos_period:
            n = sum(1 for o in b.world.objects_view(ObjKind.WOUNDED) if o.state == ObjState.SEEN)
            if n != self._known.get(b.me, 0):
                self._log(t, b.me, "wounded", f"blessés libres connus : {self._known.get(b.me, 0)} -> {n}", n=n)
                self._known[b.me] = n
            if t % self.pos_period == 0:
                self._pos.write(f"{t},{b.me},{pose.x:.1f},{pose.y:.1f},{pose.theta:.3f},{role[0]},{role[1]},"
                                f"{'' if tgt is None else f'{tgt[0]:.0f}'},{'' if tgt is None else f'{tgt[1]:.0f}'},"
                                f"{int(b._was_carrying)}\n")
        if self.sheet_period and t % self.sheet_period == 0 and t != self._sheet_t and b.me == min(self._brains):
            self._sheet_t = t
            self._write_sheet(os.path.join(self.root, "sheets", f"t{t:05d}.png"))
        if t % 50 == 0:
            for f in (self._events, self._timeline, self._pos):
                f.flush()

    def _snapshot(self, b, t, tag) -> None:
        try:
            img, _ = _render(b, _truth_from_playground(self.pg))
            cv2.imwrite(os.path.join(self.root, "snapshots", f"t{t:05d}_d{b.me}_{tag}.png"), img)
        except Exception as e:
            print(f"debug_export : instantané drone {b.me} : {e}")

    def _write_sheet(self, path) -> None:
        truth = _truth_from_playground(self.pg)
        ims = []
        for d in sorted(self._brains):
            try:
                ims.append(_render(self._brains[d], truth)[0])
            except Exception as e:
                print(f"debug_export : drone {d} : {e}")
        if ims:
            cv2.imwrite(path, _sheet(ims))

    def close(self) -> None:
        for f in (self._events, self._timeline, self._pos):
            f.close()


_REC: Recorder | None = None


def record(drone, brain, t: int) -> None:
    """Appelé par SwarmDrone.control après chaque pas (mode debug). Un nouveau playground = un nouveau run."""
    global _REC
    pg = getattr(drone, "playground", None)
    if pg is None:
        return
    if _REC is None or _REC.pg is not pg:
        if _REC is not None:
            _REC.close()
        _REC = Recorder(pg)
    try:
        _REC.observe(brain, t)
    except Exception as e:                                      # debug seulement : ne jamais casser la simulation
        print(f"debug_export : {e}")


def dump_maps(the_map, root: str = ROOT) -> str | None:
    """Fin de run (launcher.py) : une image + un .npz par drone et une planche dans <root>/final. Renvoie le dossier."""
    brains = [d._brain for d in the_map.drones if getattr(d, "_brain", None) is not None]
    if not brains:
        return None
    out = os.path.join(root, "final")
    if _REC is None or _REC.root != root:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    truth = _truth_from_playground(the_map.playground) \
        or [np.array(p, float) for p in getattr(the_map, "_wounded_persons_pos", [])]
    ims = []
    for b in brains:
        try:
            img, arrays = _render(b, truth)
        except Exception as e:
            print(f"dump_maps : drone {b.me} : {e}")
            continue
        cv2.imwrite(os.path.join(out, f"drone_{b.me}.png"), img)
        np.savez_compressed(os.path.join(out, f"drone_{b.me}.npz"), **arrays)
        ims.append(img)
    if ims:
        cv2.imwrite(os.path.join(out, "all.png"), _sheet(ims))
    if _REC is not None:
        _REC.close()
    print(f"debug_export : run enregistré dans {os.path.abspath(root)}")
    return out
