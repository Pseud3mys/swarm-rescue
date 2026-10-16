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

Ce qui circule = la CONTRIBUTION de chaque drone, déjà fusionnée par lui : pour un amer ou un blessé, sa meilleure
estimation (moyenne + covariance) qui résume TOUTES ses observations ; pour la carte, sa couche de log-odds. Le
receveur combine les contributions des différents drones par somme d'informations (sources indépendantes : biais de
pose différents) : tous les drones obtiennent la MÊME estimation globale, et l'incertitude décroît vraiment avec le
nombre de drones. On ne réémet jamais l'estimation globale elle-même : en revenant par un autre chemin radio, elle
serait comptée deux fois (le filtre deviendrait sûr de lui sans information nouvelle).

Révisions = tampons d'un compteur PAR PROPRIÉTAIRE (un pour la carte, un pour les objets), uniques et croissants.
Résumé = vecteur de versions : `known[o]` = "j'ai la dernière version de chaque donnée de `o` dont le tampon est
<= known[o]". L'émetteur envoie les données de `o` par tampons croissants et déclare la plage couverte (lo, hi] ;
le receveur n'avance `known[o]` que si la plage prolonge ce qu'il a déjà (lo <= known[o]) : rien ne se perd.
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
    TAKEN = 3      # ramassé ICI (n'y est plus) : pierre tombale à l'endroit de la saisie, remplacée par une
                   # observation plus récente si le blessé y est reposé


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
    pos_sigma: float = 0.0          # incertitude de position (px, écart-type) de l'estimateur du drone


@dataclass(frozen=True)
class ObjectTrack:
    """Contribution d'un drone sur un objet : SON estimation fusionnée (toutes ses observations) avec covariance.
    L'association entre drones (même blessé vu par deux drones) et la fusion globale sont faites à la lecture
    (world.fuse_tracks : porte de Mahalanobis, ordre déterministe)."""
    origin: int
    local_id: int
    rev: int                           # tampon du compteur d'objets du propriétaire (unique, croissant)
    kind: ObjKind
    pos: Vec
    t_seen: int
    state: ObjState
    cov: tuple = (400.0, 0.0, 400.0)   # covariance de position (xx, xy, yy) de l'estimation FUSIONNÉE du propriétaire

    @property
    def key(self) -> tuple[int, int]:
        return (self.origin, self.local_id)


@dataclass(frozen=True)
class MapPatch:
    """Une tuile (chunk) de la couche d'occupation d'UN drone propriétaire."""
    origin: int
    cx: int
    cy: int
    rev: int                          # tampon du compteur de carte du propriétaire (unique, croissant)
    data: Optional[np.ndarray]        # int8 (chunk, chunk), log-odds de l'origine ; None si tuile uniforme ...
    fill: int = 0                     # ... de valeur `fill` (tuile saturée libre, vide : fréquent, 1 octet)


@dataclass(frozen=True)
class Digest:
    """Résumé de ce que je sais (vecteurs de versions, taille fixe) : le voisin m'envoie seulement ce qui me manque."""
    sender: int
    t: int
    map_known: tuple                    # (MAX_ORIGINS,) : tuiles de l'origine o à jour jusqu'au tampon map_known[o]
    obj_known: tuple                    # (MAX_ORIGINS,) : idem pour les objets
    agent_seq: dict[int, int]


@dataclass(frozen=True)
class Packet:
    """Ce qu'un drone diffuse à chaque pas (broadcast dans les 250 px)."""
    sender: int
    t: int
    digest: Digest
    agents: tuple[AgentState, ...] = ()
    objects: tuple[ObjectTrack, ...] = ()
    patches: tuple[MapPatch, ...] = ()
    obj_ranges: tuple = ()              # ((origin, lo, hi), ...) : TOUS les objets de origin de tampon dans (lo, hi]
    map_ranges: tuple = ()              # idem pour les tuiles

    def approx_bytes(self) -> int:
        n = 8 * 2 * MAX_ORIGINS + 8 * len(self.digest.agent_seq) + 12 * (len(self.obj_ranges) + len(self.map_ranges))
        n += sum(72 + 24 * len(a.route) for a in self.agents)
        n += 64 * len(self.objects)
        n += sum(16 + (p.data.nbytes if p.data is not None else 1) for p in self.patches)
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
