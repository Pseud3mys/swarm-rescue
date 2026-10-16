# Essaim de drones : architecture et code

## Installation
Copier `solutions/` dans `src/swarm_rescue/solutions/` (le dossier `swarm/` et `drone.py` côte à côte), puis dans `my_drone_eval.py` :

    from swarm_rescue.solutions.drone import SwarmDrone
    class MyDroneEval(SwarmDrone): pass

Tests (sans le vrai simulateur) : `cd tests && PYTHONPATH=../solutions:. python3 test_pipeline.py 3 3000`

## Qui fait quoi (on change un fichier sans toucher aux autres)

| Fichier | Rôle | Lit | Écrit |
|---|---|---|---|
| `schema.py` | formats de données (le contrat) | | |
| `config.py` | tous les paramètres (dynamique, capteurs, réglages) | | |
| `estimator.py` | EKF 6 états (pose + biais GPS/compas AR(1)), filtre de vitesse, `update_landmark` générique | capteurs | `Pose` |
| `landmarks.py` | coins de mur : extraction lidar, association Mahalanobis, recalage de l'EKF sur l'estimation de l'essaim moins la sienne, contribution avec covariance | scan, EKF | objets `CORNER` |
| `perception.py` | lidar → carte (preuves pondérées par l'incertitude de pose, corps mobiles masqués), sémantique → objets | `Pose`, capteurs | `WorldModel` |
| `gridmap.py` / `world.py` | carte par tuiles (une couche par drone, révision seulement si le contenu change), agents, objets (une contribution fusionnée par drone, association + fusion à la lecture) | | |
| `comms.py` | gossip à vecteurs de versions (plages complètes, résumé de taille fixe) : ne dépend que de `WorldModel` | paquets | paquet sortant |
| `pathing.py` | grille gonflée, Dijkstra scipy (coût de tous les buts en une passe), frontières | `WorldModel` | `Route` |
| `missions.py` | QUOI faire : porter, blessé libre, frontière, retour ; coordination par missions annoncées | coûts de `PlanGrid` | `Mission` |
| `controller.py` | asservissement en vitesse, freinage, évitement | `Pose`, `Route`, lidar | `Command` |
| `brain.py` | séquencement et événements (replanification, blocage, saisie/dépose) | tout | `Command`, `Packet` |
| `drone.py` | seul fichier qui touche au simulateur | `DroneAbstract` | |
| `network.py` | image du réseau radio (liens entendus / prévus, composantes) | `WorldModel` | |
| `teamplan.py` | allocation par réseau connexe : dépose, meneurs (le plus proche, en profondeur), relais derrière eux, suiveurs, blessés/base (hongrois) | `PlanGrid`, `WorldModel` | `Mission` par drone |
| `topo.py` | carte topologique : pièces / portes / couloirs (ouverture morphologique), arbre depuis la base, axes de passage | `PlanGrid` | `TopoMap` |
| `fusion.py` | gaussiennes 2D : forme information, fusion indépendante, Kalman à plancher, Mahalanobis | | |
| `../debug_view.py` | fenêtre Qt de debug (carte de chaque drone + vue essaim), `SWARM_DEBUG=0` pour la couper | | |

Un pas : radio → EKF (+ amers) → carte/objets → mission+chemin (sur événement seulement) → vitesse → publication de l'état aux voisins.

## Ce que les fichiers du simulateur ont changé
- Drone commandé en **force** (accélération max ≈ 0,0067 px/pas²), frottement quasi nul : freiner de 1,5 px/pas demande > 150 px. Rotation ≈ 0,006 rad/pas. D'où : pilotage en vitesse holonome, **cap maintenu**, vitesse plafonnée par la distance de freinage (fin de route, virages, lidar).
- Bruit GPS/compas AR(1) (φ=0,98), **gelé** quand le capteur est coupé : le biais est un état de l'EKF et n'évolue que si le capteur marche.
- Le bruit de θ de l'odomètre (1°/pas) dépasse la rotation réelle : le cap est prédit par le modèle, l'odomètre n'est utilisé qu'en cas de vraie rotation.

## Résultats sur le mini-simulateur (`tests/tools/toysim.py`, mêmes modèles de bruit)
- Position : EKF 3,5 px RMS contre 6,6 px pour le GPS brut ; cap 2,2° contre 3,9° ; 5,3 px après 500 pas sans GPS.
- Recalage sur amers : 15 px d'erreur → ~2 px en 400 pas, GPS coupé (test unitaire `test_landmarks.py`).
- 3 drones, 3000 pas : 4 blessés sur 6 sauvés, 0 collision, ~2 ms de calcul par pas et par drone.
- Radio coupée 1500 pas puis rétablie : les cartes (42-54 %) convergent vers l'union (~75 %).
- Murs de la carte : aucune cellule à plus d'une cellule d'un vrai mur.

## À vérifier sur le vrai simulateur (je n'ai pas pu l'exécuter)
1. `drone.py::_kind` : noms des types d'entité du capteur sémantique.
2. ~~Format de `received_messages`~~ : vérifié, `[(Communicator, message)]`, objets passés par référence (d'où l'immutabilité).
3. Géométrie du centre de secours et condition de sauvetage : `release_dist`, `grasp_dist` dans `config.py`. Après une dépose, un blessé encore visible est considéré non sauvé.
4. Identifiant du drone : supposé dans 0..15 (`MAX_ORIGINS`).
5. Gain d'accélération (`accel_per_cmd`) : issu de force/masse ; un essai en boucle ouverte (forward=1 pendant 100 pas, comparer l'odomètre) le confirmerait.
6. ~~Bruit du lidar~~ : vérifié, σ = 2,5 px sur TOUS les rayons, y compris sans impact (300 + bruit) : seuil d'impact à 300 − 3σ.

## Limites connues
- Non codé : zones de destruction / sans radio (les indices `DANGER_HINT`, `NOCOM_HINT` existent dans le schéma ; `extra_cost` de `PlanGrid` permet de les éviter ; `communicator_is_disabled()` n'est pas lu), mission `RELAY` (drone central messager), blessés mobiles.
- Un drone mort ou muet reste compté dans l'équipe 100 pas (allocation), 400 (porteur), 600 (surcoût de route).

## Partage de perceptions fusionnées
Principe : chaque drone publie SA contribution déjà fusionnée (toutes ses observations, avec covariance), le receveur
combine les contributions des différents drones par somme d'informations. On ne réémet jamais l'estimation globale :
revenue par un autre chemin radio, elle serait comptée deux fois (filtre sûr de lui sans information nouvelle).
Tous les drones qui ont reçu les mêmes contributions ont donc exactement la même estimation globale.

- Objets (blessés, centre, coins) : une piste par drone et par objet = Kalman sur ses observations, plancher de
  covariance = incertitude de sa pose (biais commun à ses observations). Publication seulement si utile (déplacement
  > 1,5 px, covariance réduite de 20 %, changement d'état ; blessés : rafraîchis tous les 30 pas). L'estimation interne
  s'affine entre deux publications. Lecture (`world.fuse_tracks`) : association par porte de Mahalanobis (99,9 %)
  bornée par deux rayons par nature, ordre déterministe, fusion par somme d'informations, plus l'estimation
  « tous sauf moi » (`others_pos/others_cov`).
- Coins (`landmarks.py`) : créés avec la covariance de mesure propagée depuis la pose ; recalage de l'EKF sur
  « global moins ma contribution » avec la covariance 2D complète (`update_landmark(cov_l=...)`) ; avec GPS aussi, sur
  les coins vus par d'autres et convergés (σ < 3 px), délai 40 pas par coin ; plus de nouvelle contribution sur un coin
  déjà convergé (σ < 2 px) : volume borné.
- Carte : une couche par drone (preuves indépendantes, somme exacte). Une tuile n'est republiée que si son contenu a
  changé (écart >= 4 log-odds ou changement de classe) ; on émet la version publiée (copies identiques partout) ;
  tuiles uniformes envoyées sur 1 octet. Preuves pondérées par l'incertitude de pose (impact 3 / 2 / 1 selon σ).
  Corps mobiles masqués du lidar : blessés et drones vus par le sémantique, plus les drones (position annoncée
  extrapolée, porteurs élargis) et blessés connus jusqu'à 330 px ; impacts lidar déjà gravés sur un blessé connu
  effacés de sa propre couche. Seuil d'impact à 300 − 3σ (le bruit s'applique aussi aux rayons sans impact).
- Radio (`comms.py`) : vecteurs de versions par origine (2 x 16 entiers au lieu d'un tableau de révisions par tuile
  et d'un dictionnaire par objet) ; plages (lo, hi] déclarées complètes, plafonnées à ce que l'émetteur connaît sans
  trou ; plages reçues dans le désordre gardées en attente jusqu'à ce que le trou soit comblé ; objets envoyés dès
  qu'ils changent (chaque pas), tuiles partagées équitablement entre origines. Anti-redondance : relais désigné par
  voisin en retard et par origine (le mieux informé parmi ceux qu'il entend, puis le plus petit identifiant ; tout le
  monde reprend après 8 pas sans progrès), plages entendues chez un voisin non renvoyées pendant le délai d'accusé
  (5 pas : paquet construit à k, lu à k+2, résumé du voisin lu à k+4).
  Vérifié hors simulateur : convergence exacte (6 drones, 30 % de pertes, topologie mobile).

À faire :
1. Localisation coopérative : détection sémantique des autres drones -> mesure de position pour eux
   (intersection de covariances, surtout sans GPS ; `AgentState.pos_sigma` est déjà publié).
2. Comms : priorité des tuiles selon leur utilité (frontières, routes), résumé d'une pièce explorée.
3. Sous-réseaux : 2-3 groupes entiers, chacun connexe (aujourd'hui : excursions de binômes hors réseau).
4. Routes de rapatriement partagées, puis retour du passage de relais (`handoff_enabled`).

## Carte topologique (topo.py) et planification
Méthode « deux échelles » du prototype segproto.py :
- distance aux murs CONNUS (l'inconnu compte comme libre : une pièce à moitié vue garde son noyau) ;
- noyaux de pièces = libre à plus de `topo_room_core` d'un mur, étendus géodésiquement (sans traverser un mur fin) ;
- le reste du libre = couloirs (éclats de coin réabsorbés) ; arêtes = passages (tronçons continus d'interface,
  position + largeur) ; arbre depuis la base ; identifiants stables par recouvrement maximal d'un calcul au suivant.
- Seuil : 36 px dans le prototype (portes de 40-80 px) ; les cartes du dépôt ont des ouvertures de 70-170 px,
  d'où 48 px, validé sur la vérité terrain de MapMedium02 (murs rastérisés) et sur la carte réelle d'un drone
  (pas 300 : régions ouvertes ; pas 1000 : quasi identique à la vérité terrain). 56 px fusionne le hall.
- Recalcul tous les `topo_period` = 40 pas (~8 ms).
- Créneau d'une frontière (teamplan) : sa cellule la plus proche de la BASE, pas du drone qui calcule. Mesuré
  (MapMedium02, plans de deux drones à moins de 3 pas d'écart) : 87 % des désaccords d'allocation venaient de créneaux
  différents, dont près de la moitié de cette seule position ; les cartes, elles, concordent à > 99 % là où les deux
  drones savent. Le reste vient de connaissances différentes (frontières vues par l'un seulement, drones hors portée).
- Validation sur la vérité terrain de toutes les cartes (scratch `topo_check.py`) : correcte sur MapMedium01 et les
  cartes à murs droits ; défauts : un couloir est découpé et absorbé par les pièces voisines à travers les portes
  (2023_24_03), fragmentation près des murs diagonaux (2023_24_02, 2024_25_02). L'intérieur des murs doubles forme
  des régions hors arbre, sans conséquence. Partager les passages : peu utile (topologie recalculée au même pas par tous).

## Planification d'équipe (teamplan.py) et exploration
Causes mesurées des défauts de l'ancienne allocation (MapMedium02 et finale 2024_25_02) : chaque frontière exigeait une
chaîne de relais complète depuis la base (2,5 à 3 relais + 1 à 1,8 drones garés par plan, 46-49 % du temps en relais) ;
la méthode hongroise sur la somme des coûts échangeait le drone de tête avec un plus lointain (30 % des créneaux meneur
avaient un relais plus proche : « tête devenue relais », relais devant l'explorateur) ; frontières prises par distance à
la base (exploration en largeur) ; grille, liste noire et topologie propres à chaque drone (plans divergents).

Nouvelle allocation, par réseau connexe (`component` : liens entendus ou distance < 240 px entre positions extrapolées ;
positions dans le réseau : 3 pas d'âge médian, 7-10 au 90e centile ; hors réseau : 45-150 pas) :
1. dépose (`DROP`) : un meneur qui QUITTE une pièce finie contenant un blessé le pose devant la porte, côté pièce
   parente, hors de l'axe ; un autre drone (relais, base, libre) le ramène à la base (`brain._refresh_explore`) ;
2. meneurs : glouton « le plus proche d'abord » sur le coût de chemin, bonus pour finir sa pièce et garder sa voie,
   frontières réservées par les drones d'autres réseaux pénalisées ; `lane_share` = 50 % des drones libres ;
3. relais : chaîne base -> meneur tous les 200 px, JAMAIS au-delà du meneur, décalés jusqu'à 64 px hors des passages
   et des routes (la radio traverse les murs) ;
4. suiveurs : le drone libre le plus proche de chaque meneur ; drones en trop -> meneurs supplémentaires ;
5. blessés, relais, base : hongrois sur les drones restants.
Grille d'équipe commune (depuis la base, sans surcoût autour des drones), pas de liste noire dans le plan (un drone
qui a échoué sur une cible y renonce lui-même), topologie recalculée à `t // topo_period`.
Exploration : un meneur recible toutes les 10 pas le bord ouvert le plus proche de lui (il suit l'ouvert au lieu de viser
un point fixe) ; routes au milieu des couloirs et des portes (un raccourci ne s'approche pas plus des murs que ses
extrémités). Lidar : un ou deux rayons qui « traversent » un mur entre deux impacts distants de < 28 px sont corrigés
en impact (`perception.fill_lidar_holes`) : sinon un trou gravé dans le mur attirait la route le long du mur.
Mesures (1 essai chacune) : MapMedium02 1000 pas : désaccord des créneaux meneur 24 -> 8 %, meneur le plus proche
47 -> 75 %, temps d'exploration x2 ; finale 2024_25_02 1500 pas : exploration 38 -> 68 %, relais sur une route 22 -> 0 %.
Contrepartie : réseau moins connexe (5 drones par réseau au lieu de 8) ; `lane_share` règle le compromis.
Avec la dépose à la porte (déclenchée quand le meneur quitte la pièce) : finale 1500 pas : exploration 49 % (68 % sans
dépose, 38 % ancien code), trajets vers des blessés x6 (blessés déposés + 1/3 des drones libres réservés aux blessés),
toujours 4/19 sauvés. Un seul essai par configuration : à confirmer sur plusieurs graines et sur une partie complète.
