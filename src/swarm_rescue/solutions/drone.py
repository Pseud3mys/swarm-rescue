"""Seul fichier qui touche au simulateur : adapte DroneAbstract <-> Brain.

À placer dans src/swarm_rescue/solutions/ (avec le paquet `swarm/` à côté, ou copié dedans), puis dans
my_drone_eval.py :   from swarm_rescue.solutions.drone import SwarmDrone
                     class MyDroneEval(SwarmDrone): pass
Les points à vérifier sur le vrai simulateur sont marqués ADAPT.
"""
from __future__ import annotations

import os

import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract

from .swarm.brain import Brain
from .swarm.config import Params
from .swarm.schema import Det, MAX_ORIGINS, SensorFrame


def _kind(entity_type) -> str:
    """ADAPT : nom du type d'entité du capteur sémantique (WoundedPerson / RescueCenter / Drone)."""
    name = str(getattr(entity_type, "name", entity_type)).upper()
    if "WOUND" in name:
        return "wounded"
    if "RESCUE" in name:
        return "rescue"
    if "DRONE" in name:
        return "drone"
    return "other"


_DEBUG_VIEW = None   # fenêtre unique partagée par tous les drones (cf. debug_view.py)


class SwarmDrone(DroneAbstract):
    # Debug (à couper pour la soumission, ou via la variable d'environnement SWARM_DEBUG=0) :
    DEBUG_VIEW = os.environ.get("SWARM_DEBUG", "1") != "0"   # fenêtre Qt avec la carte interne de chaque drone
    DEBUG_PERIOD = 1                                         # la vue répartit elle-même son travail sur les pas

    def __init__(self, identifier=None, misc_data=None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, **kwargs)
        self._brain = None
        self._t = 0
        self._origin = int(self.identifier) if identifier is not None and 0 <= int(identifier) < MAX_ORIGINS else None
        self._debug_group = None

    def _ensure_brain(self):
        if self._brain is None:
            origin = self._origin if self._origin is not None else (abs(hash(self.identifier)) % MAX_ORIGINS)
            self._brain = Brain(origin, self.size_area, Params())

    def define_message_for_all(self):
        # Les messages sont diffusés avant control() : on envoie l'état construit au pas précédent.
        return self._brain.outgoing() if self._brain is not None else None

    def control(self) -> CommandsDict:
        self._ensure_brain()
        sem = []
        for d in (self.semantic_values() or []):
            k = _kind(d.entity_type)
            if k in ("wounded", "rescue", "drone") and d.distance > 0:     # distance 0 = mur : à ignorer
                sem.append(Det(k, float(d.distance), float(d.angle), bool(d.grasped)))
        frame = SensorFrame(
            t=self._t,
            gps=self.measured_gps_position(),
            compass=self.measured_compass_angle(),
            odom=None if self.odometer_is_disabled() else np.array(self.odometer_values(), float),
            lidar=None if self.lidar_is_disabled() else np.array(self.lidar_values(), float),
            angles=np.array(self.lidar_rays_angles(), float),
            semantic=sem,
            carrying=len(self.grasped_wounded_persons()) > 0,
            health=float(self.drone_health),
            in_return_area=bool(self.is_inside_return_area),
            inbox=[msg for (_, msg) in self.communicator.received_messages if msg is not None],
        )
        self._t += 1
        return self._brain.step(frame).as_dict()

    # ------------------------------------------------------------------ debug (affichage seulement)
    def display(self) -> None:
        """Appelée à chaque pas par le GUI, sur le premier drone seulement : pilote la fenêtre de debug."""
        super().display()
        if not self.DEBUG_VIEW:
            return
        if self._debug_group is None:
            pg = self.playground
            if pg is None or getattr(pg.window, "headless", False):
                return
            self._debug_group = [a for a in pg.agents if isinstance(a, SwarmDrone)]
        global _DEBUG_VIEW
        if _DEBUG_VIEW is None:
            from .debug_view import SwarmDebugView
            _DEBUG_VIEW = SwarmDebugView(self.DEBUG_PERIOD)
        _DEBUG_VIEW.update(self._debug_group, self._t)
