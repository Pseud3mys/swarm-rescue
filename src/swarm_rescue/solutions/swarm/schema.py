"""Format des données partagées entre drones (le "contrat").

Règles de conception (à ne pas casser) :
  1. Chaque donnée a UN SEUL propriétaire (`origin` = identifiant du drone qui l'a produite)
     et une révision croissante (`rev` / `seq`). Fusion = "la révision la plus haute gagne".
     => la fusion est commutative, associative et idempotente : ordre, doublons et
        pertes de messages n'ont aucune importance (c'est un CRDT par propriétaire).
  2. Un drone ne modifie JAMAIS une donnée dont il n'est pas propriétaire : il la relaie.
  3. Tout est immuable (frozen) et les tableaux numpy sont copiés à l'émission,
     pour que deux drones ne partagent jamais la même mémoire dans le simulateur.
  4. Le temps est le pas de simulation (commun à tous les drones, pas d'horloge à synchroniser).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

MAX_ORIGINS = 16  # identifiants de drones 0..15 (10 en pratique)

Vec = tuple[float, float]


class MissionKind(IntEnum):
    IDLE = 0
    EXPLORE = 1
    GOTO_WOUNDED = 2
    CARRY = 3
    RETURN = 4
    RELAY = 5  # rester en position centrale pour servir de messager
    FOLLOW = 6  # suivre un meneur d'exploration à distance radio (binôme)
    RECEIVE = 7  # aller prendre le blessé d'un porteur (passage de relais), peer = porteur


class ObjKind(IntEnum):
    WOUNDED = 1
    RESCUE_CENTER = 2
    RETURN_AREA = 3
    DANGER_HINT = 4   # un drone a disparu ici (kill zone probable)
    NOCOM_HINT = 5    # perte de contact ici (no-com zone probable)
    NOGPS_HINT = 6    # GPS/compas incohérents ici
    CORNER = 7        # amer géométrique (coin de mur) pour le recalage de pose


class ObjState(IntEnum):
    """Ordre = treillis : la fusion prend le max (un blessé sauvé ne redevient pas 'vu')."""
    SEEN = 0
    CARRIED = 1
    RESCUED = 2


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    eta: int  # pas de simulation absolu prévu de passage


@dataclass(frozen=True)
class Mission:
    kind: MissionKind = MissionKind.IDLE
    target: Optional[Vec] = None
    started: int = 0
    peer: int = -1      # FOLLOW : meneur suivi ; CARRY : receveur demandé (passage de relais) ; RECEIVE : porteur


@dataclass(frozen=True)
class AgentState:
    """Ligne propriétaire = `origin`. Sert à la coordination (qui va où, quand)."""
    origin: int
    seq: int
    t: int
    pos: Vec
    vel: Vec
    health: float
    mission: Mission
    route: tuple[Waypoint, ...] = ()
    heard: tuple[int, ...] = ()     # voisins dont j'ai reçu un paquet à ce pas (liens radio réels)


@dataclass(frozen=True)
class ObjectTrack:
    """Observation d'un objet par son propriétaire. La déduplication entre drones
    (même blessé vu par deux drones) est faite à la lecture, pas à l'écriture."""
    origin: int
    local_id: int
    rev: int
    kind: ObjKind
    pos: Vec
    t_seen: int
    state: ObjState

    @property
    def key(self) -> tuple[int, int]:
        return (self.origin, self.local_id)


@dataclass(frozen=True)
class MapPatch:
    """Une tuile (chunk) de la couche d'occupation d'UN drone propriétaire."""
    origin: int
    cx: int
    cy: int
    rev: int
    data: np.ndarray  # int8 (chunk, chunk), log-odds de l'origine


@dataclass(frozen=True)
class Digest:
    """Résumé de ce que je sais : permet au voisin de m'envoyer seulement ce qui me manque."""
    sender: int
    t: int
    map_rev: np.ndarray                 # (MAX_ORIGINS, ncx, ncy) uint32
    agent_seq: dict[int, int]
    obj_rev: dict[tuple[int, int], int]


@dataclass(frozen=True)
class Packet:
    """Ce qu'un drone diffuse à chaque pas (broadcast dans les 250 px)."""
    sender: int
    t: int
    digest: Digest
    agents: tuple[AgentState, ...] = ()
    objects: tuple[ObjectTrack, ...] = ()
    patches: tuple[MapPatch, ...] = ()

    def approx_bytes(self) -> int:
        n = self.digest.map_rev.nbytes + 16 * (len(self.digest.agent_seq) + len(self.digest.obj_rev))
        n += sum(64 + 24 * len(a.route) for a in self.agents)
        n += 48 * len(self.objects)
        n += sum(24 + p.data.nbytes for p in self.patches)
        return n


# ---------------------------------------------------------------------------
# Types d'interface entre algorithmes (non transmis par radio)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pose:
    """Sortie de l'estimateur. Repère monde (origine au centre de la carte), px et rad."""
    x: float
    y: float
    theta: float
    vx: float = 0.0           # px / pas
    vy: float = 0.0
    cov: np.ndarray = field(default_factory=lambda: np.eye(3))   # covariance (x, y, theta)
    gps_ok: bool = True

    @property
    def pos(self) -> np.ndarray:
        return np.array((self.x, self.y))

    @property
    def vel(self) -> np.ndarray:
        return np.array((self.vx, self.vy))

    @property
    def sigma(self) -> float:
        """Incertitude de position (px, écart-type moyen)."""
        return float(math.sqrt(max(0.5 * (self.cov[0, 0] + self.cov[1, 1]), 0.0)))


@dataclass(frozen=True)
class Route:
    """Trajectoire grossière : polyligne monde + heure de passage prévue (pas de simulation)."""
    pts: np.ndarray            # (N, 2)
    eta: np.ndarray            # (N,) int
    goal: Optional[Vec]
    t0: int

    def __len__(self) -> int:
        return len(self.pts)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum()) if len(self.pts) > 1 else 0.0

    def waypoints(self, n: int = 8) -> tuple:
        """Version compacte pour la diffusion aux autres drones."""
        if len(self.pts) == 0:
            return ()
        idx = np.unique(np.linspace(0, len(self.pts) - 1, min(n, len(self.pts))).astype(int))
        return tuple(Waypoint(float(self.pts[i, 0]), float(self.pts[i, 1]), int(self.eta[i])) for i in idx)


@dataclass(frozen=True)
class Command:
    forward: float = 0.0
    lateral: float = 0.0
    rotation: float = 0.0
    grasper: int = 0

    def as_dict(self) -> dict:
        return {"forward": float(self.forward), "lateral": float(self.lateral),
                "rotation": float(self.rotation), "grasper": int(self.grasper)}


# ---------------------------------------------------------------------------
# Entrée du cerveau : un instantané des capteurs (le seul format que drone.py doit produire)
# ---------------------------------------------------------------------------
from typing import Any, NamedTuple


class Det(NamedTuple):
    """Détection sémantique. kind in {'wounded', 'rescue', 'drone'}."""
    kind: str
    distance: float
    angle: float          # relatif au cap du drone
    grasped: bool = False


@dataclass
class SensorFrame:
    t: int
    gps: Optional[np.ndarray]          # None si capteur coupé
    compass: Optional[float]
    odom: Optional[np.ndarray]         # (dist, alpha, dtheta)
    lidar: Optional[np.ndarray]
    angles: Optional[np.ndarray]
    semantic: list                     # list[Det]
    carrying: bool
    health: float
    in_return_area: bool
    inbox: list                        # messages reçus tels quels (paquets ou (émetteur, paquet))
