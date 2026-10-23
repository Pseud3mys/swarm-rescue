"""WorldModel : la mémoire PARTIELLE d'un drone (carte + agents + objets).

Aucun drone n'a la vue globale ; chacun possède un WorldModel qui se complète au fil des
rencontres (voir comms.py). Les algos (estimation, planification, contrôle) ne lisent que
l'API de lecture ci-dessous et n'écrivent que via les méthodes publiques.

Objets (blessés, centre de secours, amers) :
  - chaque drone tient UNE piste par objet physique : son estimation fusionnée (filtre de Kalman sur toutes ses
    observations, avec un plancher = incertitude de sa pose, commune à toutes ses observations) ;
  - il ne la publie (nouvelle révision) que si elle a changé de façon utile (déplacement, incertitude nettement
    réduite, changement d'état) : l'estimation interne continue de s'affiner entre deux publications ;
  - à la lecture, `fuse_tracks` associe les pistes des différents drones (porte de Mahalanobis, ordre déterministe :
    tous les drones qui ont les mêmes pistes obtiennent les mêmes objets) et les fusionne par somme d'informations.

Blessés :
  - bruit de processus (PROCESS_Q) : un blessé mobile reste associé à sa piste au lieu d'en créer une nouvelle ;
  - classe de mouvement par piste (`motion_class` : fixe / mobile / inconnue, d'après MES observations) ; sans GPS,
    un blessé fixe garde sa position (`observe_object(anchor_fixed=True)`) et sert d'amer à l'EKF (brain.py) ;
  - `mark_gone` : blessé vérifié absent (imaginaire, parti, déjà pris) -> mes pistes TAKEN + pierre tombale qui
    masque celles des autres drones ; une observation plus récente le fait réapparaître.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from . import fusion
from .gridmap import ChunkedMap
from .schema import (MAX_ORIGINS, AgentState, Digest, Mission, ObjKind, ObjState, ObjectTrack, Vec,
                     Waypoint)

# association (px) : (toujours le même objet en dessous de r_min, jamais au-delà de r_max, porte de Mahalanobis entre)
ASSOC = {
    ObjKind.WOUNDED: (20.0, 45.0),          # deux blessés (rayon 12) ne sont jamais à moins de ~24 px l'un de l'autre
    ObjKind.RESCUE_CENTER: (150.0, 250.0),
    ObjKind.RETURN_AREA: (150.0, 250.0),
    ObjKind.DANGER_HINT: (60.0, 60.0),
    ObjKind.NOCOM_HINT: (60.0, 60.0),
    ObjKind.NOGPS_HINT: (60.0, 60.0),
    ObjKind.CORNER: (4.0, 12.0),            # deux coins d'un mur fin peuvent être à ~8 px
}
GATE = 13.8          # Mahalanobis² (2 ddl, 99,9 %)
CLUSTER_RADIUS = {k: v[1] for k, v in ASSOC.items()}   # compatibilité (rayon maximal)

# poids de la nouvelle mesure dans la position d'une piste sans covariance (1 = on remplace)
SMOOTH = {ObjKind.WOUNDED: 0.5, ObjKind.RESCUE_CENTER: 0.1, ObjKind.RETURN_AREA: 0.05,
          ObjKind.CORNER: 0.2}
# republication périodique d'une piste inchangée (pas) : seulement pour les objets dont l'état peut changer
REFRESH = {ObjKind.WOUNDED: 30}
# bruit de processus (px² par pas, marche aléatoire) : certains blessés se déplacent (~2 px/pas). Sans lui, la
# covariance d'une piste tombe à quelques px² et un blessé qui a bougé de 20 px sort de la porte : nouvelle piste,
# l'ancienne reste « vue » à un endroit vide (blessé imaginaire)
PROCESS_Q = {ObjKind.WOUNDED: 4.0}
# blessé fixe / mobile (d'après mes observations) : fixe = estimation filtrée restée à moins de WOUNDED_FIXED_DEV px de
# son ancre (estimation après 5 observations) pendant WOUNDED_FIXED_T pas (bruit de mesure et de pose compris) ;
# mobile dès WOUNDED_MOVING_DEV px (définitif)
WOUNDED_FIXED_T = 60
WOUNDED_FIXED_DEV = 10.0
WOUNDED_MOVING_DEV = 20.0
PROCESS_DT_MAX = 100


@dataclass(frozen=True)
class FusedObject:
    """Vue dédupliquée (non transmise) : ce que le planner voit."""
    kind: ObjKind
    pos: Vec
    t_seen: int
    state: ObjState
    sources: tuple[tuple[int, int], ...]
    t_carried: int = -1     # dernier instant où un drone le portait (-1 : jamais)
    cov: tuple = (400.0, 0.0, 400.0)   # covariance de la position fusionnée (toutes les origines)
    others_pos: Optional[Vec] = None   # fusion de toutes les origines SAUF le lecteur (None : lui seul l'a vu)
    others_cov: Optional[tuple] = None
    mine: Optional[tuple] = None       # clé de la piste du lecteur dans ce groupe

    @property
    def sigma(self) -> float:
        return math.sqrt(max(0.5 * (self.cov[0] + self.cov[2]), 0.0))


def _inv2(a: float, b: float, c: float) -> tuple:
    """Inverse d'une matrice symétrique 2x2 [[a, b], [b, c]] (forme fermée : appelée des milliers de fois)."""
    det = max(a * c - b * b, 1e-12)
    return c / det, -b / det, a / det


def _same(kind, pa, ca, pb, cb) -> tuple[bool, float]:
    r_min, r_max = ASSOC[kind]
    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
    d = math.hypot(dx, dy)
    if d > r_max:
        return False, float("inf")
    if d <= r_min:
        return True, 0.0
    ia, ib, ic = _inv2(ca[0] + cb[0], ca[1] + cb[1], ca[2] + cb[2])
    m2 = ia * dx * dx + 2.0 * ib * dx * dy + ic * dy * dy
    return m2 < GATE, m2


def _fuse_group(tracks: list, exclude: int = -1):
    """Fusion des pistes d'un groupe : une estimation par origine (la plus précise, CARRIED exclues), sommées en
    forme information (sources indépendantes). Renvoie ((pos, cov) ou None)."""
    per_origin = {}
    for x in tracks:
        if x.state == ObjState.CARRIED or x.origin == exclude:
            continue
        cur = per_origin.get(x.origin)
        if cur is None or x.cov[0] + x.cov[2] < cur.cov[0] + cur.cov[2]:
            per_origin[x.origin] = x
    if not per_origin:
        return None
    if len(per_origin) == 1:
        x = next(iter(per_origin.values()))
        return x.pos, x.cov
    Ya = Yb = Yc = ya = yb = 0.0
    for x in per_origin.values():
        ia, ib, ic = _inv2(*x.cov)
        Ya, Yb, Yc = Ya + ia, Yb + ib, Yc + ic
        ya += ia * x.pos[0] + ib * x.pos[1]
        yb += ib * x.pos[0] + ic * x.pos[1]
    Pa, Pb, Pc = _inv2(Ya, Yb, Yc)
    return (Pa * ya + Pb * yb, Pb * ya + Pc * yb), (Pa, Pb, Pc)


def fuse_tracks(tracks, me: int = -1) -> list[FusedObject]:
    """Association + fusion des pistes de tous les drones (d'une même nature ou non).
    Ordre déterministe (précision puis clé) : deux drones qui ont les mêmes pistes obtiennent les mêmes groupes."""
    order = sorted(tracks, key=lambda o: (int(o.kind), round(o.cov[0] + o.cov[2], 3), o.origin, o.local_id))
    groups: list[dict] = []
    buckets: dict = {}
    for tr in order:
        r = ASSOC[tr.kind][1]
        bx, by = int(math.floor(tr.pos[0] / r)), int(math.floor(tr.pos[1] / r))
        best, bm = None, float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for g in buckets.get((tr.kind, bx + dx, by + dy), ()):
                    ok, m2 = _same(tr.kind, g["pos"], g["cov"], tr.pos, tr.cov)
                    if ok and m2 < bm:
                        best, bm = g, m2
        if best is None:
            best = {"kind": tr.kind, "tracks": [], "pos": tr.pos, "cov": tr.cov, "b": (tr.kind, bx, by)}
            groups.append(best)
            buckets.setdefault(best["b"], []).append(best)
        best["tracks"].append(tr)
        if tr.state != ObjState.CARRIED:
            f = _fuse_group(best["tracks"])
            if f is not None:
                best["pos"], best["cov"] = f
    out = []
    for g in groups:
        c = g["tracks"]
        latest = max(c, key=lambda x: (x.t_seen, x.origin, x.local_id))
        if any(x.state == ObjState.RESCUED for x in c):
            st = ObjState.RESCUED
        else:
            st = latest.state                                # l'observation la plus récente décide
        if latest.state == ObjState.CARRIED:                 # porté : il est là où est son porteur
            pos, cov = latest.pos, latest.cov
        else:
            f = _fuse_group(c)
            pos, cov = f if f is not None else (latest.pos, latest.cov)
        oth = _fuse_group(c, exclude=me) if me >= 0 else None
        mine = next((x.key for x in c if x.origin == me), None)
        out.append(FusedObject(g["kind"], pos, latest.t_seen, st, tuple(sorted(x.key for x in c)),
                               max([x.t_seen for x in c if x.state == ObjState.CARRIED], default=-1), cov,
                               oth[0] if oth else None, oth[1] if oth else None, mine))
    return out


class WorldModel:
    def __init__(self, me: int, size_area, resolution: float = 8.0, chunk: int = 16):
        self.me = me
        self.map = ChunkedMap(size_area, resolution, chunk)
        self.near = None     # (W, H) bool : passé à portée sémantique d'un drone (local, écrit par brain._mark_near)
        self.agents: dict[int, AgentState] = {}
        self.objects: dict[tuple[int, int], ObjectTrack] = {}
        self.obj_known = np.zeros(MAX_ORIGINS, dtype=np.int64)   # vecteur de versions des objets (cf. schema.py)
        self._own: dict[tuple[int, int], tuple] = {}             # estimation interne de mes pistes : (pos, P, t_seen)
        self._motion: dict = {}          # mes pistes de blessés : [ancre (estimation filtrée), instant, écart max, nb d'obs.]
        self._pending: dict = {}                                 # plages reçues hors séquence, par (flux, origine)
        self._agent_seq = 0
        self._next_local = 0
        self.version = 0     # incrémenté à chaque changement de la table d'objets
        self._kver: dict = {}            # version par nature d'objet (caches de vue indépendants)
        self._views: dict = {}

    # ---- interne ---------------------------------------------------------
    def _touch(self, kind) -> None:
        self.version += 1
        self._kver[kind] = self._kver.get(kind, 0) + 1

    def _stamp(self) -> int:
        self.obj_known[self.me] += 1
        return int(self.obj_known[self.me])

    def _publish(self, tr: ObjectTrack) -> ObjectTrack:
        tr = replace(tr, rev=self._stamp())
        self.objects[tr.key] = tr
        self._touch(tr.kind)
        return tr

    # ---- écritures locales (je suis propriétaire) -----------------------
    def publish_self(self, t: int, pos: Vec, vel: Vec, health: float,
                     mission: Mission, route: tuple[Waypoint, ...] = (), heard=(), pos_sigma: float = 0.0) -> AgentState:
        self._agent_seq += 1
        a = AgentState(self.me, self._agent_seq, t, (float(pos[0]), float(pos[1])),
                       (float(vel[0]), float(vel[1])), float(health), mission, tuple(route),
                       tuple(sorted(int(h) for h in heard)), float(pos_sigma))
        self.agents[self.me] = a
        return a

    def observe_object(self, kind: ObjKind, pos, t: int,
                       state: ObjState = ObjState.SEEN, cov=None, floor_var: float = 0.0,
                       anchor_fixed: bool = False) -> ObjectTrack:
        """Crée ou met à jour MA piste de même nature qui correspond à la mesure (porte de Mahalanobis si `cov`).
        Avec `cov` (covariance 2x2 de la mesure, monde) : filtre de Kalman sur la position, avec un plancher
        `floor_var` (incertitude de MA pose, commune à toutes mes observations : elle ne se moyenne pas)."""
        pos = (float(pos[0]), float(pos[1]))
        R = None if cov is None else np.asarray(cov, float)
        r0 = ASSOC[kind][0]
        mc = fusion.cov_pack(R) if R is not None else (r0 * r0, 0.0, r0 * r0)
        best, bm = None, float("inf")
        q = PROCESS_Q.get(kind, 0.0)
        for tr in self.objects.values():
            if tr.origin == self.me and tr.kind == kind and tr.state not in (ObjState.TAKEN,):
                own = self._own.get(tr.key)
                p0, c0 = (own[0], fusion.cov_pack(own[1])) if own else (tr.pos, tr.cov)
                if q > 0.0:
                    dq = q * min(max(t - (own[2] if own else tr.t_seen), 0), PROCESS_DT_MAX)
                    c0 = (c0[0] + dq, c0[1], c0[2] + dq)
                ok, m2 = _same(kind, p0, c0, pos, mc)
                if ok and m2 < bm:
                    best, bm = tr, m2
        if best is None:
            P0 = R if R is not None else fusion.cov_unpack(ObjectTrack.cov)
            if R is not None and floor_var > 0:
                P0 = P0 + max(floor_var - float(np.linalg.eigvalsh(P0).min()), 0.0) * np.eye(2)
            tr = ObjectTrack(self.me, self._next_local, 0, kind, pos, t, state, fusion.cov_pack(P0))
            self._next_local += 1
            self._own[tr.key] = (np.array(pos), P0, t)
            if kind == ObjKind.WOUNDED:
                self._motion[tr.key] = [None, t, 0.0, 1]
            return self._publish(tr)
        own = self._own.get(best.key) or (np.array(best.pos), fusion.cov_unpack(best.cov), best.t_seen)
        new_state = ObjState.RESCUED if best.state == ObjState.RESCUED else state
        mu, P = own[0], own[1]
        mot = self._motion.get(best.key) if kind == ObjKind.WOUNDED else None
        if anchor_fixed and state == ObjState.SEEN and self.motion_class(best.key) == "fixed":
            # blessé fixe servant d'amer sans GPS : sa position n'est plus réestimée avec une pose qui dérive
            self._own[best.key] = (own[0], own[1], t)
            stale = kind in REFRESH and t - best.t_seen >= REFRESH[kind]
            return self._publish(replace(best, t_seen=t)) if stale else best
        if q > 0.0 and state != ObjState.CARRIED:        # le blessé a pu bouger depuis la dernière mesure
            P = P + q * min(max(t - own[2], 0), PROCESS_DT_MAX) * np.eye(2)
        if state == ObjState.CARRIED:                    # un blessé porté ne déplace pas sa piste ici
            pass
        elif R is not None:
            mu, P = fusion.kalman_update(mu, P, pos, R, floor_var)
        else:
            a = SMOOTH.get(kind, 1.0)
            mu = mu + a * (np.array(pos) - mu)
        self._own[best.key] = (np.array(mu, float), P, t)
        if mot is not None and state == ObjState.SEEN:
            # écart de l'estimation FILTRÉE à une ancre prise après 5 observations (une mesure brute peut être fausse
            # de 13 px : relèvement sémantique discrétisé à ~10°)
            mot[3] += 1
            if mot[0] is None:
                if mot[3] >= 5:
                    mot[0], mot[1] = np.array(mu, float), t
            else:
                mot[2] = max(mot[2], math.hypot(mu[0] - mot[0][0], mu[1] - mot[0][1]))
        # publication seulement si utile : état, déplacement, incertitude nettement réduite, ou rafraîchissement
        moved = math.hypot(mu[0] - best.pos[0], mu[1] - best.pos[1])
        shrunk = P[0, 0] + P[1, 1] < 0.8 * (best.cov[0] + best.cov[2])
        stale = kind in REFRESH and t - best.t_seen >= REFRESH[kind]
        if new_state == best.state and moved < 1.5 and not shrunk and not stale:
            return best
        return self._publish(replace(best, pos=(float(mu[0]), float(mu[1])), t_seen=t, state=new_state,
                                     cov=fusion.cov_pack(P)))

    def motion_class(self, key) -> str:
        """Ma piste de blessé : "fixed" (vue >= WOUNDED_FIXED_T pas sans bouger de plus de WOUNDED_FIXED_DEV px),
        "moving" (a bougé de plus de WOUNDED_MOVING_DEV px : définitif), sinon "unknown"."""
        mot = self._motion.get(key)
        if mot is None:
            return "unknown"
        if mot[2] > WOUNDED_MOVING_DEV:
            return "moving"
        if mot[0] is not None and mot[2] < WOUNDED_FIXED_DEV \
                and self._own.get(key, (None, None, mot[1]))[2] - mot[1] >= WOUNDED_FIXED_T:
            return "fixed"
        return "unknown"

    def wounded_class(self, obj: "FusedObject") -> str:
        """Classe d'un blessé fusionné d'après MES pistes du groupe (mobile si l'une a bougé)."""
        cls = [self.motion_class(k) for k in obj.sources if k[0] == self.me]
        if "moving" in cls:
            return "moving"
        return "fixed" if "fixed" in cls else "unknown"

    def set_state(self, key, state: ObjState, t: int) -> Optional[ObjectTrack]:
        """Change l'état d'une de MES pistes (ex. la piste d'un blessé que je viens de déposer)."""
        tr = self.objects.get(key)
        if tr is None or tr.origin != self.me:
            return None
        if tr.state == ObjState.RESCUED:
            return tr
        return self._publish(replace(tr, t_seen=t, state=state))

    def mark_gone(self, obj: FusedObject, t: int) -> None:
        """Objet fusionné vérifié absent de sa position (blessé imaginaire, blessé mobile parti, blessé déjà pris) :
        mes pistes du groupe passent à TAKEN, et une pierre tombale TAKEN plus récente masque celles des autres drones
        (que je ne peux pas modifier). Une observation plus récente le fait réapparaître."""
        others = False
        for key in obj.sources:
            if key[0] == self.me:
                self.set_state(key, ObjState.TAKEN, t)
            else:
                others = True
        if others:
            self.add_track(obj.kind, obj.pos, t, ObjState.TAKEN, obj.cov)

    def add_track(self, kind: ObjKind, pos, t: int, state: ObjState, cov=None) -> ObjectTrack:
        """Nouvelle piste à MOI, sans association (ex. pierre tombale "pris ici")."""
        tr = ObjectTrack(self.me, self._next_local, 0, kind, (float(pos[0]), float(pos[1])), t, state,
                         tuple(cov) if cov is not None else ObjectTrack.cov)
        self._next_local += 1
        return self._publish(tr)

    def move_track(self, key, pos, t: int) -> Optional[ObjectTrack]:
        """Déplace une de MES pistes (blessé porté : il est là où je suis)."""
        tr = self.objects.get(key)
        if tr is None or tr.origin != self.me or tr.state == ObjState.RESCUED:
            return tr
        self._own[key] = (np.array(pos, float), fusion.cov_unpack(tr.cov), t)
        return self._publish(replace(tr, pos=(float(pos[0]), float(pos[1])), t_seen=t))

    def nearest_own(self, kind: ObjKind, pos, radius: float) -> Optional[ObjectTrack]:
        best, bd = None, radius
        for tr in self.objects.values():
            if tr.origin == self.me and tr.kind == kind:
                d = math.hypot(tr.pos[0] - pos[0], tr.pos[1] - pos[1])
                if d <= bd:
                    best, bd = tr, d
        return best

    # ---- fusions distantes (je ne suis PAS propriétaire) -----------------
    def merge_agent(self, a: AgentState) -> bool:
        if a.origin == self.me:   # écho de ma propre ligne : la mienne fait foi
            return False
        cur = self.agents.get(a.origin)
        if cur is None or a.seq > cur.seq:
            self.agents[a.origin] = a
            return True
        return False

    def merge_object(self, o: ObjectTrack) -> bool:
        if o.origin == self.me:
            return False
        cur = self.objects.get(o.key)
        if cur is None or o.rev > cur.rev:
            self.objects[o.key] = o
            self._touch(o.kind)
            return True
        return False

    def advance_known(self, ranges, map_ranges) -> None:
        """Plages (origin, lo, hi] reçues COMPLÈTES : le vecteur de versions n'avance que si elles prolongent ce que
        j'ai déjà (sinon un trou resterait invisible). Une plage reçue avant celle qui la précède (paquets de
        plusieurs voisins dans le désordre) est gardée en attente et intégrée dès que le trou est comblé : ses
        données ont déjà été appliquées, et toute version plus récente a un tampon plus grand."""
        for stream, known, rr in (("obj", self.obj_known, ranges), ("map", self.map.known, map_ranges)):
            for o, lo, hi in rr:
                if o == self.me or not 0 <= o < MAX_ORIGINS or hi <= known[o]:
                    continue
                pend = self._pending.setdefault((stream, o), [])
                pend.append((lo, hi))
                changed = True
                while changed:
                    changed = False
                    for i, (a, b) in enumerate(pend):
                        if a <= known[o] < b:
                            known[o] = b
                            changed = True
                    pend[:] = [(a, b) for a, b in pend if b > known[o]]
                if len(pend) > 32:
                    del pend[:-32]

    def tracks_after(self, origin: int, base: int) -> list[ObjectTrack]:
        return sorted((o for o in self.objects.values() if o.origin == origin and o.rev > base), key=lambda o: o.rev)

    # ---- lecture ---------------------------------------------------------
    def objects_view(self, kind: Optional[ObjKind] = None) -> list[FusedObject]:
        """Vue fusionnée par nature, mise en cache jusqu'au prochain changement des pistes de cette nature."""
        if kind is None:
            return [o for k in sorted({int(x.kind) for x in self.objects.values()})
                    for o in self.objects_view(ObjKind(k))]
        ver = self._kver.get(kind, 0)
        c = self._views.get(kind)
        if c is None or c[0] != ver:
            c = (ver, fuse_tracks([o for o in self.objects.values() if o.kind == kind], self.me))
            self._views[kind] = c
        return c[1]

    def _objects_view(self, kind: Optional[ObjKind]) -> list[FusedObject]:
        return fuse_tracks([o for o in self.objects.values() if kind is None or o.kind == kind], self.me)

    def others(self) -> list[AgentState]:
        return [a for o, a in self.agents.items() if o != self.me]

    def digest(self, t: int) -> Digest:
        return Digest(self.me, t, tuple(int(v) for v in self.map.known), tuple(int(v) for v in self.obj_known),
                      {o: a.seq for o, a in self.agents.items()})
