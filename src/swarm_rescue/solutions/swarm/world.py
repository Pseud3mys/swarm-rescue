"""WorldModel : la mémoire PARTIELLE d'un drone (carte + agents + objets).

Aucun drone n'a la vue globale ; chacun possède un WorldModel qui se complète au fil des
rencontres (voir comms.py). Les algos (estimation, planification, contrôle) ne lisent que
l'API de lecture ci-dessous et n'écrivent que via les méthodes publiques.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from .gridmap import ChunkedMap
from .schema import (AgentState, Digest, Mission, ObjKind, ObjState, ObjectTrack, Vec,
                     Waypoint)

# rayon de fusion à la lecture (px) : deux observations plus proches = même objet
CLUSTER_RADIUS = {
    ObjKind.WOUNDED: 30.0,
    ObjKind.RESCUE_CENTER: 150.0,
    ObjKind.RETURN_AREA: 150.0,
    ObjKind.DANGER_HINT: 60.0,
    ObjKind.NOCOM_HINT: 60.0,
    ObjKind.NOGPS_HINT: 60.0,
    ObjKind.CORNER: 12.0,
}

# poids de la nouvelle mesure dans la position d'une piste (1 = on remplace)
SMOOTH = {ObjKind.WOUNDED: 0.5, ObjKind.RESCUE_CENTER: 0.1, ObjKind.RETURN_AREA: 0.05,
          ObjKind.CORNER: 0.2}


@dataclass(frozen=True)
class FusedObject:
    """Vue dédupliquée (non transmise) : ce que le planner voit."""
    kind: ObjKind
    pos: Vec
    t_seen: int
    state: ObjState
    sources: tuple[tuple[int, int], ...]
    t_carried: int = -1     # dernier instant où un drone le portait (-1 : jamais)


class WorldModel:
    def __init__(self, me: int, size_area, resolution: float = 8.0, chunk: int = 16):
        self.me = me
        self.map = ChunkedMap(size_area, resolution, chunk)
        self.agents: dict[int, AgentState] = {}
        self.objects: dict[tuple[int, int], ObjectTrack] = {}
        self._agent_seq = 0
        self._next_local = 0
        self.version = 0     # incrémenté à chaque changement de la table d'objets (invalide les caches)

    # ---- écritures locales (je suis propriétaire) -----------------------
    def publish_self(self, t: int, pos: Vec, vel: Vec, health: float,
                     mission: Mission, route: tuple[Waypoint, ...] = ()) -> AgentState:
        self._agent_seq += 1
        a = AgentState(self.me, self._agent_seq, t, (float(pos[0]), float(pos[1])),
                       (float(vel[0]), float(vel[1])), float(health), mission, tuple(route))
        self.agents[self.me] = a
        return a

    def observe_object(self, kind: ObjKind, pos, t: int,
                       state: ObjState = ObjState.SEEN) -> ObjectTrack:
        """Crée ou met à jour MA piste la plus proche de même nature."""
        pos = (float(pos[0]), float(pos[1]))
        best, best_d = None, CLUSTER_RADIUS[kind]
        for tr in self.objects.values():
            if tr.origin == self.me and tr.kind == kind:
                d = math.hypot(tr.pos[0] - pos[0], tr.pos[1] - pos[1])
                if d <= best_d:
                    best, best_d = tr, d
        if best is None:
            tr = ObjectTrack(self.me, self._next_local, 1, kind, pos, t, state)
            self._next_local += 1
        else:
            a = 0.0 if state == ObjState.CARRIED else SMOOTH.get(kind, 1.0)   # un blessé porté ne déplace pas sa piste
            new_state = ObjState.RESCUED if best.state == ObjState.RESCUED else state
            new_pos = (best.pos[0] + a * (pos[0] - best.pos[0]), best.pos[1] + a * (pos[1] - best.pos[1]))
            moved = math.hypot(new_pos[0] - best.pos[0], new_pos[1] - best.pos[1])
            if new_state == best.state and moved < 1.5 and t - best.t_seen < 30:
                return best      # rien de neuf : pas de nouvelle révision (évite d'inonder le gossip)
            tr = replace(best, rev=best.rev + 1, pos=new_pos, t_seen=t, state=new_state)
        self.objects[tr.key] = tr
        self.version += 1
        return tr

    def set_state(self, key, state: ObjState, t: int) -> Optional[ObjectTrack]:
        """Change l'état d'une de MES pistes (ex. la piste d'un blessé que je viens de déposer)."""
        tr = self.objects.get(key)
        if tr is None or tr.origin != self.me:
            return None
        if tr.state == ObjState.RESCUED:
            return tr
        tr = replace(tr, rev=tr.rev + 1, t_seen=t, state=state)
        self.objects[key] = tr
        self.version += 1
        return tr

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
            self.version += 1
            return True
        return False

    # ---- lecture ---------------------------------------------------------
    def objects_view(self, kind: Optional[ObjKind] = None) -> list[FusedObject]:
        clusters: list[list[ObjectTrack]] = []
        for tr in sorted(self.objects.values(), key=lambda o: -o.t_seen):
            if kind is not None and tr.kind != kind:
                continue
            r = CLUSTER_RADIUS[tr.kind]
            for c in clusters:
                head = c[0]
                if head.kind == tr.kind and math.hypot(head.pos[0] - tr.pos[0],
                                                       head.pos[1] - tr.pos[1]) <= r:
                    c.append(tr)
                    break
            else:
                clusters.append([tr])
        out = []
        for c in clusters:
            if any(x.state == ObjState.RESCUED for x in c):
                st = ObjState.RESCUED
            else:
                st = max(c, key=lambda x: x.t_seen).state          # l'observation la plus récente décide
            out.append(FusedObject(c[0].kind, c[0].pos, c[0].t_seen, st, tuple(x.key for x in c),
                                   max([x.t_seen for x in c if x.state == ObjState.CARRIED], default=-1)))
        return out

    def others(self) -> list[AgentState]:
        return [a for o, a in self.agents.items() if o != self.me]

    def digest(self, t: int) -> Digest:
        rev = self.map.rev.copy()
        rev.flags.writeable = False
        return Digest(self.me, t, rev,
                      {o: a.seq for o, a in self.agents.items()},
                      {k: o.rev for k, o in self.objects.items()})
