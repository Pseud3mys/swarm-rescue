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
| `landmarks.py` | coins de mur : extraction lidar, association, recalage de l'EKF | scan, EKF | objets `CORNER` |
| `perception.py` | lidar → carte (preuves libre/occupé), sémantique → objets | `Pose`, capteurs | `WorldModel` |
| `gridmap.py` / `world.py` | carte par tuiles (une couche par drone), agents, objets | | |
| `comms.py` | gossip : ne dépend que de `WorldModel` | paquets | paquet sortant |
| `pathing.py` | grille gonflée, Dijkstra scipy (coût de tous les buts en une passe), frontières | `WorldModel` | `Route` |
| `missions.py` | QUOI faire : porter, blessé libre, frontière, retour ; coordination par missions annoncées | coûts de `PlanGrid` | `Mission` |
| `controller.py` | asservissement en vitesse, freinage, évitement | `Pose`, `Route`, lidar | `Command` |
| `brain.py` | séquencement et événements (replanification, blocage, saisie/dépose) | tout | `Command`, `Packet` |
| `drone.py` | seul fichier qui touche au simulateur | `DroneAbstract` | |
| `network.py` | image du réseau radio (liens entendus / prévus, composantes) | `WorldModel` | |
| `teamplan.py` | allocation globale (base, binômes meneur/suiveur, relais, blessés) par méthode hongroise | `PlanGrid`, `WorldModel` | `Mission` par drone |
| `fusion.py` | gaussiennes 2D : forme information, fusion indépendante, retrait de sa contribution, intersection de covariances | | |
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
2. Format de `received_messages` (supposé `[(Communicator, message)]`, d'après `communicator.py`) et que les objets sont passés par référence (les paquets sont immuables).
3. Géométrie du centre de secours et condition de sauvetage : `release_dist`, `grasp_dist` dans `config.py`. Après une dépose, un blessé encore visible est considéré non sauvé.
4. Identifiant du drone : supposé dans 0..15 (`MAX_ORIGINS`).
5. Gain d'accélération (`accel_per_cmd`) : issu de force/masse ; un essai en boucle ouverte (forward=1 pendant 100 pas, comparer l'odomètre) le confirmerait.
6. Bruit du lidar (supposé σ = 2 px), valeur renvoyée sans impact (supposée ≥ 0,985 × portée).

## Limites connues
- Le recalage sur coins n'est actif que sans GPS (ou pose très incertaine) : avec un bon GPS il rendait le filtre trop confiant. Il dépend de la densité de coins ; dans mon arène de test les coins sont rares.
- Non codé : zones de destruction / sans radio (les indices `DANGER_HINT`, `NOCOM_HINT` existent dans le schéma ; `extra_cost` de `PlanGrid` permet de les éviter), mission `RELAY` (drone central messager), évitement entre drones (seulement via lidar et surcoût autour des routes annoncées), blessés mobiles.
- Aucun chiffre ici ne valide le comportement sur les vraies cartes.

## Partage de perceptions fusionnées (en cours)
Fait : chaque piste d'objet porte la covariance de l'estimation fusionnée de son propriétaire (Kalman avec
plancher = incertitude de sa pose, commune à ses observations). La vue fusionnée combine les propriétaires
par somme d'informations (biais de pose indépendants entre drones). Le modèle du capteur sémantique corrige
son biais (+2,5 px) et vise le centre du blessé (surface + 12 px).

À faire, dans l'ordre :
1. Association par distance de Mahalanobis (au lieu du rayon fixe) et identifiant commun déterministe.
2. Amers (coins) partagés avec covariance ; recalage de l'EKF sur « global moins ma contribution ».
3. Localisation coopérative : détection sémantique des autres drones -> mesure de position pour eux
   (intersection de covariances, surtout sans GPS).
4. Comms : résumé compressé, diffusion poussée à durée de vie limitée des infos urgentes, priorité vers la base.
   Le gossip actuel reste la garantie de convergence.
5. Sous-réseaux : 2-3 groupes entiers, chacun connexe (aujourd'hui : excursions de binômes hors réseau).
6. Routes de rapatriement partagées, puis retour du passage de relais (`handoff_enabled`).
