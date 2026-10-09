"""Tous les paramètres réglables au même endroit. Les valeurs "simulateur" viennent des fichiers
du simulateur (drone_base.py, definitions.py, constants.py, drone_sensors.py) : à revérifier
si la version du simulateur change."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Params:
    # ---- dynamique du drone (unités : px, pas de simulation) ------------------------------
    # Un pas de simulation = 10 micro-pas pymunk de 0,1 s (playground.step), soit 1 s de physique ;
    # la force n'est appliquée que pendant le 1er micro-pas (pymunk la remet à zéro). Valeurs mesurées
    # en boucle ouverte sur le vrai simulateur (forward = 1, rotation = 1).
    dt: float = 1.0                    # s de physique par pas
    accel_per_cmd: float = 0.55        # px/pas^2 pour forward = 1 (300 N / 50 kg * 0.1 s ~ 0.6, moins le frottement)
    accel_per_cmd_carry: float = 0.18  # en portant : blessé de 100 kg accroché au drone de 50 kg -> masse x3
    drag: float = 0.95                 # SPACE_DAMPING par seconde de physique = par pas
    ang_per_cmd: float = 0.176         # rad/pas pour rotation = 1 (ANGULAR_VELOCITY * ANGULAR_SPEED_RATIO * 1 s)
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
    accel_model_trust: float = 0.15    # écart-type d'accélération non modélisée (px/pas^2)

    # ---- carte ------------------------------------------------------------------------------------
    resolution: float = 8.0
    chunk: int = 16
    ev_free: int = -1
    ev_hit: int = 3
    max_pose_sigma_for_mapping: float = 14.0
    # capteur sémantique (drone_semantic_sensor.py) : distance + N(2.5, 1) (biais +2,5 px, écart-type 1 :
    # np.random.normal(2.5) prend 2.5 comme MOYENNE), angle exact mais discrétisé (35 rayons, 10,3°) ;
    # la distance va jusqu'à la SURFACE du blessé (rayon 12)
    sem_dist_bias: float = 2.5
    sem_sigma_r: float = 2.0
    sem_sigma_b: float = 0.05          # rad (~3° : écart-type d'une erreur uniforme de ±5°)
    wounded_radius: float = 12.0

    # ---- planification --------------------------------------------------------------------------
    inflate_cells: int = 2             # gonflage des obstacles (cellules) : contrainte dure (16 px + rayon du drone)
    wall_cost_cells: float = 6.0       # coût de proximité des murs jusqu'à 6 cellules (48 px) : reste au milieu des couloirs
    wall_cost: float = 6.0             # surcoût max (collé au mur)
    los_wall_cells: float = 3.0        # les raccourcis en ligne droite ne passent pas à moins de 3 cellules d'un mur
    inflate_cells_tight: int = 1       # repli si le but est injoignable
    unknown_cost: float = 2.5          # coût d'un pas en terrain inconnu (planification optimiste)
    frontier_min_cells: int = 3
    replan_period: int = 45
    switch_gain: float = 0.75          # une nouvelle cible doit coûter < 75 % de l'actuelle pour la remplacer

    # ---- contrôle ---------------------------------------------------------------------------------
    v_cruise: float = 4.0              # px/pas (vitesse max théorique ~11 px/pas)
    v_carry: float = 2.5
    brake_frac: float = 0.7            # part de l'accélération max réservée au freinage
    lookahead: float = 45.0
    wall_margin: float = 14.0
    repulse_dist: float = 28.0
    vel_time_const: float = 6.0        # pas : constante de temps de l'asservissement de vitesse

    # ---- équipe / réseau (teamplan.py) ----------------------------------------------------------
    # unités de coût : longueur de chemin en cellules (8 px) pondérée par le coût du terrain
    link_range: float = 220.0          # px : écart max entre deux maillons du réseau (radio 250 px, marge)
    team_fresh: int = 100              # pas : un drone non entendu depuis plus longtemps est hors de mon réseau
    team_min_for_base: int = 4         # à partir de N drones libres, l'un reste posté à la base
    team_max_frontiers: int = 4        # binômes d'exploration reliés au réseau, au plus
    team_max_excursions: int = 2       # binômes (ou solos) hors réseau, au plus : le réseau se scinde en 3 morceaux max
    frontier_spacing: float = 150.0    # px : deux binômes ne visent pas des frontières plus proches que ça
    frontier_size_gain: float = 2.0    # une grande frontière passe devant une petite un peu plus proche
    standby_ring: float = 110.0        # px : base et réserve en anneau autour du centre, pas sur son point d'accès
    wounded_bonus: float = 80.0        # un blessé passe devant une frontière (en coût de chemin)
    sticky_radius: float = 60.0        # px : même tâche si la cible a bougé de moins que ça
    sticky_gain: float = 0.7           # hystérésis : coût de la tâche déjà annoncée x 0.7 ...
    sticky_bonus: float = 5.0          # ... moins ce bonus
    follow_dist: float = 70.0          # px : le suiveur se tient à cette distance derrière le meneur
    pair_max: float = 160.0            # px : au-delà, le meneur attend son suiveur
    follow_replan: int = 12            # pas : rafraîchissement de la route du suiveur

    # ---- missions ----------------------------------------------------------------------------------
    grasp_dist: float = 32.0
    rescue_approach_dist: float = 120.0  # centre de secours vu plus près : on fonce dessus à vue (le blessé est sauvé au contact)
    rescue_approach_speed: float = 1.2
    rescue_spin_dist: float = 30.0     # centre plus près que ça ...
    rescue_spin_after: int = 60        # ... depuis N pas sans sauvetage : on tourne sur place (secours si le cap ne suffit pas)
    rot_gain: float = 3.0              # rotation = gain x erreur de cap (0,176 rad/pas pour 1 : gain 3 -> ~50 % par pas)
    carry_track_period: int = 15       # pas : la piste du blessé porté suit le porteur
    # ---- saisie, priorités de passage, passage de relais du blessé ------------------------------
    grasp_close_dist: float = 60.0     # px : pince fermée dès cette distance (la saisie se fait au contact)
    approach_min_speed: float = 1.0    # px/pas : vitesse mini de l'approche finale (heurter un blessé ne coûte rien)
    approach_mask_deg: float = 30.0    # rayons lidar ignorés autour de la cible (sinon la répulsion nous en écarte)
    inflate_cells_carry: int = 3       # porteur : le blessé dépasse, on garde plus de marge (repli 2 puis 1)
    yield_radius: float = 75.0         # px : en dessous, le drone le moins prioritaire s'écarte
    yield_path_dist: float = 40.0      # px : ... si la route annoncée de l'autre passe à moins de ça de lui
    yield_speed: float = 1.5
    handoff_enabled: bool = False      # désactivé : le donneur reste collé au receveur et ils se bloquent (à reprendre)
    queue_max_wait: int = 40           # pas : attente max derrière un drone qui va dans le même sens, puis on passe
    handoff_gain: float = 150.0        # px : un receveur doit être au moins ça plus près du centre que le porteur
    handoff_range: float = 220.0       # px : et à portée radio du porteur
    handoff_timeout: int = 200         # pas : passage de relais abandonné au-delà
    recover_steps: int = 30            # durée de la manœuvre de dégagement après un blocage
    drop_check_steps: int = 20         # après une dépose, un blessé encore visible n'est PAS sauvé
