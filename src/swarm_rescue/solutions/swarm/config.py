"""Tous les paramètres réglables au même endroit. Les valeurs "simulateur" viennent des fichiers
du simulateur (drone_base.py, definitions.py, constants.py, drone_sensors.py) : à revérifier
si la version du simulateur change."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Params:
    # ---- dynamique du drone (unités : px, pas de simulation) ------------------------------
    dt: float = 1 / 30                 # constants.FRAME_RATE
    accel_per_cmd: float = 6.0 / 900   # (LINEAR_FORCE*LINEAR_SPEED_RATIO / masse) * dt^2 = 6 px/s^2 -> px/pas^2
    drag: float = 0.95 ** (1 / 30)     # SPACE_DAMPING^dt : perte de vitesse par pas
    ang_per_cmd: float = 0.3 * 0.6 / 30  # rad/pas pour rotation = 1 (très lent : on ne tourne presque jamais)
    robot_radius: float = 10.0

    # ---- capteurs (drone_sensors.py) ---------------------------------------------------------
    gps_phi: float = 0.98
    gps_sigma: float = 5.0
    compass_phi: float = 0.98
    compass_sigma: float = math.radians(4.0)
    odom_sigma_dist: float = 0.2
    odom_sigma_alpha: float = math.radians(8.0)
    odom_sigma_theta: float = math.radians(1.0)
    lidar_sigma: float = 2.0           # bruit gaussien supposé (non vu dans les fichiers fournis)
    lidar_max_range: float = 300.0
    semantic_max_range: float = 200.0
    comm_range: float = 250.0

    # ---- estimateur -----------------------------------------------------------------------------
    gps_meas_sigma: float = 0.7        # bruit blanc résiduel sur le GPS (tout le gros du bruit est dans le biais)
    compass_meas_sigma: float = math.radians(0.7)
    landmark_sigma_r: float = 2.5
    landmark_sigma_b: float = math.radians(1.5)
    gate_chi2: float = 11.0
    accel_model_trust: float = 0.003   # écart-type d'accélération non modélisée (px/pas^2)

    # ---- carte ------------------------------------------------------------------------------------
    resolution: float = 8.0
    chunk: int = 16
    ev_free: int = -1
    ev_hit: int = 3
    max_pose_sigma_for_mapping: float = 14.0

    # ---- planification --------------------------------------------------------------------------
    inflate_cells: int = 2             # gonflage des obstacles (cellules)
    inflate_cells_tight: int = 1       # repli si le but est injoignable
    unknown_cost: float = 2.5          # coût d'un pas en terrain inconnu (planification optimiste)
    frontier_min_cells: int = 3
    replan_period: int = 45
    switch_gain: float = 0.75          # une nouvelle cible doit coûter < 75 % de l'actuelle pour la remplacer

    # ---- contrôle ---------------------------------------------------------------------------------
    v_cruise: float = 1.6              # px/pas (≈ 48 px/s)
    v_carry: float = 1.0
    brake_frac: float = 0.7            # part de l'accélération max réservée au freinage
    lookahead: float = 45.0
    wall_margin: float = 14.0
    repulse_dist: float = 28.0
    vel_time_const: float = 40.0       # pas : constante de temps de l'asservissement de vitesse

    # ---- missions ----------------------------------------------------------------------------------
    grasp_dist: float = 32.0
    release_dist: float = 10.0         # distance au bord du centre de secours pour lâcher (contact ; à vérifier sur le simulateur)
    drop_check_steps: int = 20         # après une dépose, un blessé encore visible n'est PAS sauvé
