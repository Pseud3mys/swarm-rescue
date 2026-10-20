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
| `config.py` | tous les paramètres (dynamique, capteurs, réglages), commentés avec leur justification | | |
| `estimator.py` | EKF 8 états (pose, vitesse, biais GPS / boussole AR(1)) avec modèle de mouvement mesuré ; filtre de vitesse séparé pour le contrôle ; mises à jour GPS, boussole, amer, recalage de carte | capteurs | `Pose` |
| `landmarks.py` | coins de mur : extraction lidar, association Mahalanobis, recalage de l'EKF sur l'estimation de l'essaim moins la sienne, contribution avec covariance | scan, EKF | objets `CORNER` |
| `scanmatch.py` | recalage scan -> carte (champ d'occupation lissé, Gauss-Newton) : pose sur les murs des autres, cap sur ma carte | lidar, carte | mesure pour l'EKF |
| `perception.py` | lidar -> MA couche de carte (règles anti-effacement / anti-épaississement, corps mobiles masqués) ; sémantique -> pistes d'objets avec covariance | `Pose`, capteurs | `WorldModel` |
| `gridmap.py` | carte par tuiles, une couche de log-odds par drone, somme à la lecture, révision seulement si le contenu change | | |
| `world.py` | agents et objets : une piste fusionnée par drone et par objet, bruit de processus et classe fixe / mobile des blessés, pierres tombales, association + fusion à la lecture | | |
| `fusion.py` | gaussiennes 2D : forme information, fusion indépendante, Kalman à plancher, Mahalanobis | | |
| `comms.py` | gossip à vecteurs de versions (plages complètes, résumé de taille fixe) : ne dépend que de `WorldModel` | paquets | paquet sortant |
| `pathing.py` | grille gonflée, Dijkstra scipy (coût de tous les buts en une passe), frontières | `WorldModel` | `Route` |
| `missions.py` | QUOI faire : porter, blessé libre, frontière, retour ; coordination par missions annoncées | coûts de `PlanGrid` | `Mission` |
| `teamplan.py` | allocation par réseau connexe : dépose, meneurs (le plus proche, en profondeur), relais derrière eux, suiveurs, blessés/base (hongrois) | `PlanGrid`, `WorldModel` | `Mission` par drone |
| `topo.py` | carte topologique : pièces / portes / couloirs (ouverture morphologique), arbre depuis la base, axes de passage | `PlanGrid` | `TopoMap` |
| `network.py` | image du réseau radio (liens entendus / prévus, composantes) | `WorldModel` | |
| `controller.py` | asservissement en vitesse, freinage, évitement | `Pose`, `Route`, lidar | `Command` |
| `brain.py` | séquencement et événements : ordre des recalages, blessés fixes comme amers, contrôle négatif des blessés, saisie / dépose, replanification, blocage, retour en fin de partie | tout | `Command`, `Packet` |
| `monitoring.py` | télémétrie JSONL et détection d'informations radio périmées, changements de rôle et oscillations | `Brain` | journal JSONL |
| `drone.py` | seul fichier qui touche au simulateur | `DroneAbstract` | |
| `../debug_view.py` | fenêtre Qt de debug (carte de chaque drone + vue essaim, classes de blessés), `SWARM_DEBUG=0` pour la couper | | |

## Un pas de `Brain.step`

1. **Radio** : paquets reçus -> `WorldModel` (cartes, objets, agents des autres).
2. **Prédiction EKF** (modèle de mouvement + odomètre), puis GPS et boussole s'ils marchent.
3. **Coins** (`landmarks.py`) : tous les 4 pas avec GPS, tous les 2 pas sans.
4. **Lidar** (un scan sur deux) : masque des corps mobiles -> **recalage scan -> carte** (`scanmatch.py`) AVANT
   d'écrire le scan -> **intégration** dans ma couche (`perception.py`).
5. **Sémantique** : blessés / centre de secours -> pistes ; sans GPS, **blessés fixes = amers** pour l'EKF ;
   **contrôle négatif** des blessés connus (un pas sur deux) ; effacement des impacts lidar gravés sur les blessés.
6. Saisie / dépose (pistes du blessé porté), mission + chemin (sur événement), asservissement.
7. Publication de mon état (position, σ de pose, mission, route) et construction du paquet sortant.

## Faits mesurés sur le simulateur (à la base des choix)
- Un pas = 1 s de physique ; vitesse : `v_k = 0,95 v_{k-1} + 0,55 x commande` px/pas (0,18 en portant : masse x3) ;
  rotation = commande x 0,176 rad/pas. Le contrôle pilote en vitesse holonome, cap maintenu, vitesse plafonnée par
  la distance de freinage (cf. `controller.py`).
- GPS / boussole : biais AR(1) (φ = 0,98, σ = 5 px / 4°), **gelé** quand le capteur est coupé.
- Odomètre : distance σ 0,2 px, direction σ 8°, **cap σ 1°/pas** (marche aléatoire de ~30° en 1000 pas).
- Lidar : σ = 2,5 px sur TOUS les rayons, y compris sans impact (300 + bruit) ; portée 300 px.
- Sémantique : distance jusqu'à la SURFACE du blessé (rayon 12) avec un biais de +2,5 px (σ 1) ; angle exact mais
  discrétisé (35 rayons, ~10°) ; portée 200 px.
- `received_messages` : `[(Communicator, message)]`, objets passés par référence (d'où l'immutabilité des messages).

## Estimation de pose (`estimator.py`, `landmarks.py`, `scanmatch.py`, `brain.py`)

### Filtre
EKF à 8 états `[x, y, θ, vx, vy, bx, by, bθ]` : pose, déplacement du dernier pas, biais GPS et boussole. Les biais
sont des états AR(1) qui n'évoluent que si leur capteur fonctionne. Modèle de mouvement mesuré sur le vrai simulateur
(MapIntermediate01) :

- **Cap** : l'incrément est la fusion modèle (commande x 0,176, résidu 0,07°/pas) + odomètre (1°/pas). L'odomètre
  seul est pris en cas de **choc** (écart > 4°) et **en portant** : le couple du blessé écarte la rotation du modèle de
  façon systématique (fusionner biaisait le cap de 30° sans boussole).
- **Position** : odomètre dominant. Le modèle de vitesse est 3 fois plus précis que l'odomètre sur la vitesse, mais
  lui faire confiance dégradait la position (petit biais intégré ; en portant, le blessé traîné tire le drone de
  côté) : `vel_model_sigma` est volontairement large. Un écart odomètre / modèle au-delà de la porte χ² est traité comme
  un choc (incertitude de vitesse gonflée).
- **Vitesse du contrôle** : filtre séparé (modèle d'accélération + odomètre), sans effet sur la position.
- GPS : porte χ² 99,99 %, réaccepté après 20 rejets (filtre perdu). Boussole : porte sur l'innovation.

### Sources de recalage
| Source | Avec GPS | Sans GPS | Référence utilisée |
|---|---|---|---|
| Coins de mur (`landmarks.py`) | seulement sur des coins vus par d'autres et convergés (σ < 3 px), au plus tous les 40 pas par coin | dès qu'un coin connu est revu (tous les 12 pas par coin) | estimation de l'essaim **moins ma contribution** ; la mienne seule sans GPS |
| Murs des autres (`scanmatch.py`) | pose (x, y, θ) tous les 6 pas | idem | couches des AUTRES drones |
| Ma carte (`scanmatch.py`) | jamais | cap seul, quand σθ > 1,5° (typiquement en portant) | carte complète ; plancher 1,5° si la boussole marche |
| Blessés fixes (`brain._wounded_landmarks`) | jamais | distance + relèvement, au plus tous les 10 pas par blessé | position des autres drones si dispo, sinon la mienne figée |

Pourquoi ces règles :
- **Jamais de recalage sur ses propres observations quand la pose est bonne** : elles partagent l'erreur de ma pose
  (auto-confirmation). D'où « essaim moins moi » pour les coins et les murs des autres pour la position.
- **Ma carte seulement pour le cap, et seulement s'il est vraiment incertain** : le modèle de mouvement tient le cap à
  ~0,1°/pas ; recaler sur ma carte (dont l'orientation garde le biais de boussole de mon passé) le dégradait (mesuré).
  En portant (odomètre seul, ~1°/pas), c'est au contraire la seule chose qui tient le cap.
- **Avec GPS, recalage seulement sur de l'information indépendante et convergée** : elle aide l'EKF à estimer le
  biais GPS ; un amer peu sûr ne ferait que transférer son erreur.
- Mesure de recalage de carte : covariance = variance des résidus x H⁻¹, gonflée (points corrélés), avec planchers ;
  une direction dégénérée (couloir) reçoit une grande variance et l'EKF n'y corrige presque rien. Portes χ² 99,9 %.

### Coins de mur
Extraits du scan (deux segments droits quasi perpendiculaires, contigus, 30-120 px). Un coin inconnu n'est publié
qu'à sa 2e observation (≥ 6 pas plus tard, autre point de vue). Contribution seulement avec une pose sûre (GPS,
σ < 4,5 px), observations espacées de 30 pas (bruit GPS corrélé sur ~50 pas), arrêt quand le coin a convergé
(σ < 2 px) : volume radio borné. Association Mahalanobis avec test de rapport (ambigu -> ignoré).

### Résultats (rejeu hors ligne de l'EKF sur des parties enregistrées, carte 1, 3 parties par config)
| Mode | erreur p50 / p95 / max (px) | cap |
|---|---|---|
| Avec GPS | 4,2 / 8,3 / 13,7 | p50 0,9° |
| Phase sans GPS | 8,4 / 17,0 / 19,3 | p95 2,9° |

Référence (recalage seul, sans modèle de mouvement) : 3,8 / 8,9 / 11,1 et 1,8° avec GPS ; 12,1 / 27,2 / 31,3 sans GPS.
Le recalage sur blessés fixes n'est pas encore validé en partie réelle sans GPS (aucun drone dans la zone sur
MapMedium01, 1500 pas).

## Carte d'occupation (`perception.py`, `gridmap.py`)

### Structure
Une couche de log-odds par drone (tuiles de 16 x 16 cellules de 8 px, saturée à ±20). Les couches de deux drones sont
des preuves indépendantes : la vue fusionnée est leur **somme** (faite à la lecture, saturée à ±60 ; occupé ≥ 5,
libre ≤ −3). On ne réémet jamais la somme (comptée deux fois en circulant en boucle) : chaque drone relaie les couches
telles quelles. Une tuile de MA couche n'est republiée que si son contenu a changé (écart ≥ 4 log-odds ou changement
de classe) : une zone saturée ne génère plus de trafic.

### Intégration d'un scan (`ScanIntegrator.integrate`)
La cause principale des murs effacés ou épaissis est le **cap** (biais boussole ~4°, odomètre 1°/pas : à 250 px,
5° = 22 px). Le recalage scan -> carte le corrige en amont ; l'intégration est en plus robuste à ce qui reste :
- **Poids selon ma pose** : impact 3 / 2 / 1 selon σ (< 6, < 10 px), rien d'écrit au-delà de 14 px ; libre −1.
  Ma couche porte ainsi l'incertitude de ma pose.
- **Impact** seulement à plus de 3σ sous la portée max. Un rayon sans impact n'écrit du libre que jusqu'à 16 px avant
  cette limite (un mur vers 290-300 px alterne impact / portée max).
- **Marge avant l'impact** : le libre s'arrête à 1 cellule / sin(incidence) avant l'impact (max 60 px). Un rayon rasant
  traverse la partie vide de la cellule du mur : un drone garé effacerait petit à petit un mur vu en rasant.
- **Cellules certaines** (|log-odds| ≥ 12 dans ma couche) : contredites seulement par des **rayons fiables** (impact
  continu avec ses voisins, < 200 px, et incidence franche pour le libre). Un rayon peut « glisser » entre deux pixels
  de mur en incidence rasante : il n'efface plus un mur établi.
- **Impacts lointains** (> 250 px) : poids 1 (décalage latéral dû au cap).
- **Corps mobiles masqués** (ni impact, ni libre au-delà) : blessés et drones vus par le sémantique, et corps connus
  jusqu'à 330 px (portée lidar > portée sémantique) : autres drones à leur position annoncée extrapolée (porteurs
  élargis de leur blessé), blessés connus. Les impacts gravés sur un blessé avant qu'il soit reconnu sont effacés de ma
  couche (`brain._clear_phantoms`).

### Résultats (carte 1, 1 drone, rejeu des scans ; cœurs de mur effacés / faux murs)
Avec la vraie pose, presque aucun mur effacé. Avec la pose estimée : avec GPS 11 / 39 (référence sans ces règles :
29 / 95) ; sans GPS 1 / 18 (référence : 9 / 52). Coût : le recalage tourne un pas sur deux (à surveiller sur les
grandes cartes : limite de temps réel).

## Objets et blessés (`world.py`, `perception.observe_semantic`, `brain.py`)

### Pistes et fusion
- Chaque drone tient **une piste par objet** (blessé, centre de secours, coin, zone de retour) : Kalman sur toutes ses
  observations, plancher de covariance = incertitude de sa pose (biais commun, il ne se moyenne pas).
- Mesure sémantique : rayons voisins regroupés à 25 px, centre = surface + rayon − biais, covariance propagée depuis la
  pose (relèvement : moyenne circulaire, σ / √n). Centre de secours : objet étendu, incertitude large.
- **Publication seulement si utile** : déplacement > 1,5 px, covariance réduite de 20 %, changement d'état ; blessés
  republiés tous les 30 pas. L'estimation interne s'affine entre deux publications.
- **Lecture** (`world.fuse_tracks`) : association par porte de Mahalanobis (99,9 %) bornée par deux rayons par nature
  (blessé : toujours le même sous 20 px, jamais au-delà de 45), ordre déterministe, fusion par somme d'informations,
  plus l'estimation « tous sauf moi » (`others_pos / others_cov`) pour les recalages. Tous les drones qui ont reçu
  les mêmes pistes ont exactement les mêmes objets.

### Blessés mobiles et classe fixe / mobile
- **Bruit de processus** sur les pistes de blessés (4 px² par pas, borné à 100 pas) : certains blessés se déplacent
  (~2 px/pas). Sans lui, une piste devenue très sûre ne s'associe plus à un blessé qui a bougé de 20 px : nouvelle
  piste, et l'ancienne reste « vue » à un endroit vide.
- **Classe** (`world.motion_class`, par piste, d'après MES observations) : ancre = estimation filtrée après 5
  observations (une mesure brute peut être fausse de 13 px) ; **fixe** si l'estimation reste à moins de 10 px de
  l'ancre pendant 60 pas ; **mobile** dès 20 px (définitif) ; sinon non classé. Pour un objet fusionné
  (`wounded_class`) : mobile si l'une de mes pistes du groupe l'est.
- **Blessé fixe = amer sans GPS** : sa position n'est plus réestimée (elle suivrait la dérive de ma pose) et l'EKF se
  recale dessus (cf. tableau des sources).
- Vue debug : carré bordeaux = fixe, triangle magenta = mobile, rond rouge = non classé, orange = porté.

### Blessés imaginaires, partis ou déjà pris
- **Contrôle négatif** (`brain._check_vanished`, un pas sur deux) : un blessé connu à moins de 110 px, que le
  sémantique ne voit pas, et dont tous les rayons lidar couvrant le disque (incertitudes comprises) passent plus loin
  que lui. S'il était là, son corps arrêterait au moins un rayon (un drone ou un mur devant l'arrête aussi : pas de
  conclusion). Après 4 constats consécutifs : `world.mark_gone`.
- **`mark_gone`** : mes pistes du groupe passent à TAKEN ; une **pierre tombale** TAKEN plus récente masque celles des
  autres drones (que je ne peux pas modifier). Une observation plus récente fait réapparaître le blessé.
- **Saisie** : la piste est prise au point de saisie (25 px dans la direction du blessé), une pierre tombale « pris
  ici » efface sa position, la piste passe à CARRIED et suit le porteur (tous les 15 pas). Les autres blessés connus à
  moins de 80 px et non vus sont des doublons du blessé saisi (piste d'un blessé mobile, positions décalées d'autres
  drones) : retirés.
- **Dépose** : verdict après 20 pas ; un blessé encore visible n'est pas sauvé. Dépose volontaire devant une porte :
  la piste est replacée là et redevient libre.
- **Missions** : un GOTO dont la cible n'a plus de blessé à moins de 50 px est réaffecté ; arrivé sur place sans rien
  voir pendant 25 pas : blessés proches oubliés et cible bannie 600 pas ; fin de route loin de la cible (injoignable) :
  bannie 300 pas sans l'effacer. Une DROP dont le blessé à prendre n'existe plus est abandonnée.

## Radio (`comms.py`)
Vecteurs de versions par origine (2 x 16 entiers au lieu d'un tableau de révisions par tuile et d'un dictionnaire par
objet) ; plages (lo, hi] déclarées complètes, plafonnées à ce que l'émetteur connaît sans trou ; plages reçues dans le
désordre gardées en attente jusqu'à ce que le trou soit comblé ; objets envoyés dès qu'ils changent (chaque pas),
tuiles partagées équitablement entre origines ; tuiles uniformes sur 1 octet. Anti-redondance : relais désigné par
voisin en retard et par origine (le mieux informé parmi ceux qu'il entend, puis le plus petit identifiant ; tout le
monde reprend après 8 pas sans progrès), plages entendues chez un voisin non renvoyées pendant le délai d'accusé
(5 pas : paquet construit à k, lu à k+2, résumé du voisin lu à k+4).
Vérifié hors simulateur : convergence exacte (6 drones, 30 % de pertes, topologie mobile).

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
un point fixe), avec hystérésis pour éviter les allers-retours dus au retard de fusion des cartes ; routes au milieu des
couloirs et des portes (un raccourci ne s'approche pas plus des murs que ses extrémités). Lidar : les retours proches
de la portée maximale sont rejetés à moins d'être au moins à 3σ sous la portée maximale ; le filtre de reconstruction
des « faux trous » a été supprimé.
Mesures (1 essai chacune) : MapMedium02 1000 pas : désaccord des créneaux meneur 24 -> 8 %, meneur le plus proche
47 -> 75 %, temps d'exploration x2 ; finale 2024_25_02 1500 pas : exploration 38 -> 68 %, relais sur une route 22 -> 0 %.
Contrepartie : réseau moins connexe (5 drones par réseau au lieu de 8) ; `lane_share` règle le compromis.
Avec la dépose à la porte (déclenchée quand le meneur quitte la pièce) : finale 1500 pas : exploration 49 % (68 % sans
dépose, 38 % ancien code), trajets vers des blessés x6 (blessés déposés + 1/3 des drones libres réservés aux blessés),
toujours 4/19 sauvés. Un seul essai par configuration : à confirmer sur plusieurs graines et sur une partie complète.

## Monitoring d'une exécution

Le monitoring est désactivé par défaut. Pour enregistrer chaque drone sans modifier le
launcher, définir `SWARM_MONITOR` et utiliser son mode headless existant :

    $env:SWARM_DEBUG = "0"
    $env:SWARM_MONITOR = "monitor\map_medium_02.jsonl"
    & .\.venv\Scripts\python.exe .\src\swarm_rescue\launcher.py --config .\config\my_eval_plan.yml --headless

Le fichier JSONL contient, par drone et par pas, la mission, sa source, l'âge des
informations de blessés, les voisins réellement entendus, les compteurs gossip, l'état
de l'exploration et les anomalies détectées (`rescuer-on-stale-info`,
`stale-agent-info`, `role-oscillation`). Pour afficher la trace ou l'exporter en CSV :

    & .\.venv\Scripts\python.exe -m swarm_rescue.solutions.swarm.monitoring monitor\map_medium_02.jsonl --tail 40
    & .\.venv\Scripts\python.exe -m swarm_rescue.solutions.swarm.monitoring monitor\map_medium_02.jsonl --csv monitor\map_medium_02.csv


## Résultats sur le mini-simulateur (`tests/tools/toysim.py`, mêmes modèles de bruit)
- 3 drones, 3000 pas : 4 blessés sur 6 sauvés, 0 collision, ~2 ms de calcul par pas et par drone.
- Radio coupée 1500 pas puis rétablie : les cartes (42-54 %) convergent vers l'union (~75 %).
- Murs de la carte : aucune cellule à plus d'une cellule d'un vrai mur.
- Recalage sur coins : 15 px d'erreur -> ~2 px en 400 pas, GPS coupé (test unitaire `test_landmarks.py`).
Les chiffres de précision de pose à jour sont ceux du rejeu sur le vrai simulateur (section Estimation de pose).

## À vérifier sur le vrai simulateur
1. `drone.py::_kind` : noms des types d'entité du capteur sémantique.
2. Géométrie du centre de secours et condition de sauvetage : `release_dist`, `grasp_dist` dans `config.py`.
3. Identifiant du drone : supposé dans 0..15 (`MAX_ORIGINS`).
4. Recalage sur blessés fixes sans GPS : pas encore observé en partie réelle.

## Limites connues
- Non codé : zones de destruction / sans radio (les indices `DANGER_HINT`, `NOCOM_HINT` existent dans le schéma ;
  `extra_cost` de `PlanGrid` permet de les éviter ; `communicator_is_disabled()` n'est pas lu), mission `RELAY`
  (drone central messager).
- La porte d'association ENTRE drones (`fuse_tracks`) ne vieillit pas les pistes de blessés (pas d'instant courant à
  la lecture) : le bruit de processus ne s'applique qu'à mes propres pistes.
- La classe fixe / mobile d'un blessé est locale (non transmise) : chaque drone la construit sur ses observations.
- Un drone mort ou muet reste compté dans l'équipe 100 pas (allocation), 400 (porteur), 600 (surcoût de route).

## À faire
1. Localisation coopérative : détection sémantique des autres drones -> mesure de position pour eux
   (intersection de covariances, surtout sans GPS ; `AgentState.pos_sigma` est déjà publié).
2. Comms : priorité des tuiles selon leur utilité (frontières, routes), résumé d'une pièce explorée.
3. Sous-réseaux : 2-3 groupes entiers, chacun connexe (aujourd'hui : excursions de binômes hors réseau).
4. Routes de rapatriement partagées, puis retour du passage de relais (`handoff_enabled`).
