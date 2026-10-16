"""Protocole de gossip (anti-entropie) pour un canal BROADCAST à portée limitée.

Principe : chaque pas, je diffuse (1) mon résumé `Digest` (vecteurs de versions : 2 x 16 entiers, taille fixe),
(2) ma ligne `AgentState`, (3) ce que mes voisins récents n'ont pas d'après LEUR dernier résumé :
  - pour chaque origine o en retard chez au moins un voisin, ses données de tampon > (plus petit known[o] des voisins),
    par tampons croissants, avec la plage (lo, hi] déclarée COMPLÈTE : le receveur avance known[o] sans trou possible ;
  - en attendant l'accusé (résumé du voisin, 1 à 2 pas), je continue la plage là où je l'ai laissée au lieu de
    renvoyer la même chose (`cooldown`) ; si le voisin a raté le paquet, son résumé n'avance pas et je reprends.
Objets (blessés, états, amers) : envoyés dès qu'ils changent (chaque pas) : ce sont les infos importantes et mobiles.
Tuiles : budget par pas partagé équitablement entre origines (la mienne d'abord, puis celles qui manquent au plus de
voisins). Tout drone relaie ce qu'il a, quelle que soit l'origine => un drone immobile devient un messager.

Ce fichier ne connaît ni la carte ni la planification : il ne manipule que l'API de WorldModel / ChunkedMap.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schema import MAX_ORIGINS, Packet
from .world import WorldModel


@dataclass
class GossipConfig:
    max_patches: int = 40       # tuiles de carte par pas (~256 octets, 1 octet si uniforme). Le simulateur n'impose
                                # aucune limite ; c'est surtout un plafond de calcul.
    max_objects: int = 150      # pistes d'objets par pas
    peer_ttl: int = 5           # un voisin non entendu depuis N pas n'est plus pris en compte
    cooldown: int = 5           # délai d'accusé : on continue une plage envoyée au lieu de la renvoyer. Aller-retour
                                # mesuré : paquet construit au pas k -> lu par le voisin à k+2 -> son résumé lu à k+4
    meta_period: int = 3        # états des AUTRES drones relayés tous les N pas (mon propre état : chaque pas)
    stall: int = 8              # pas sans progrès d'un voisin en retard avant que tout le monde le serve


def extract_packets(received: Iterable) -> list[Packet]:
    """Adaptateur : accepte un Packet nu ou un tuple/liste qui en contient un."""
    out = []
    for item in received or []:
        if isinstance(item, Packet):
            out.append(item)
        elif isinstance(item, (tuple, list)):
            out.extend(x for x in item if isinstance(x, Packet))
    return out


class Gossip:
    def __init__(self, world: WorldModel, cfg: GossipConfig | None = None):
        self.world = world
        self.cfg = cfg or GossipConfig()
        self._peers: dict = {}   # sender -> (t_entendu, Digest)
        self._cursor: dict = {}  # (flux, origine) -> (dernier tampon envoyé, pas)
        self._overheard: dict = {}  # (flux, origine) -> (lo, hi, pas) : dernière plage diffusée par un voisin
        self._progress: dict = {}   # (flux, voisin) -> [son vecteur connu, dernier pas de progrès par origine]
        self.stats: Counter = Counter()
        self.heard: set = set()  # émetteurs entendus au dernier ingest (= liens radio réels à ce pas)

    def ingest(self, packets: Iterable[Packet], t: int) -> None:
        w = self.world
        self.heard = set()
        for p in packets:
            if p.sender == w.me:
                continue
            self.heard.add(p.sender)
            self._peers[p.sender] = (t, p.digest)
            self.stats["agents"] += sum(w.merge_agent(a) for a in p.agents)
            self.stats["objects"] += sum(w.merge_object(o) for o in p.objects)
            for x in p.patches:
                self.stats["patches"] += w.map.apply_patch(x)
            w.advance_known(p.obj_ranges, p.map_ranges)
            # plages diffusées par un voisin : nos voisins communs les ont sans doute reçues ; pendant le délai
            # d'accusé on ne les renvoie pas (si un voisin les a ratées, son résumé n'avance pas et on reprend)
            for stream, rr in (("obj", p.obj_ranges), ("map", p.map_ranges)):
                for o, lo, hi in rr:
                    cur = self._overheard.get((stream, o))
                    if cur is None or cur[2] < t or hi > cur[1]:
                        self._overheard[(stream, o)] = (lo, hi, t)

    # ------------------------------------------------------------------ émission
    def _lagging(self, stream: str, mine, peers: dict, t: int) -> list[tuple[int, int, int]]:
        """[(priorité, origine, base)] : origines pour lesquelles un voisin est en retard ET dont je suis le relais
        désigné vers lui ; base = d'où repartir.
        Relais désigné (évite que tous les drones qui entendent un voisin en retard lui envoient la même chose) :
        parmi moi et mes voisins que LUI entend (son `heard` publié), celui qui en sait le plus sur l'origine, à égalité
        le plus petit identifiant. Si le voisin ne progresse pas pendant `stall` pas, tout le monde reprend : rien ne
        peut se perdre, au pire ça va moins vite."""
        cfg, w = self.cfg, self.world
        me = w.me
        mine = np.asarray(mine, dtype=np.int64)
        ids = list(peers)
        K = np.array([peers[i] for i in ids], dtype=np.int64).reshape(len(ids), MAX_ORIGINS)
        if not (K < mine).any():                               # personne en retard (cas courant) : tous à jour
            for r, P in enumerate(ids):
                prog = self._progress.get((stream, P))
                if prog is None:
                    self._progress[(stream, P)] = [K[r].copy(), np.full(MAX_ORIGINS, t)]
                else:
                    prog[0] = np.maximum(prog[0], K[r])
                    prog[1][:] = t
            return []
        # clé d'élection (connaissance, puis plus petit identifiant) : un seul max par origine
        ida = np.array(ids, dtype=np.int64)
        key = K * 64 + (63 - ida)[:, None]
        key_me = mine * 64 + (63 - me)
        lagging: dict[int, list] = {}
        for r, P in enumerate(ids):
            kP = K[r]
            behind = kP < mine
            prog = self._progress.setdefault((stream, P), [kP.copy(), np.full(MAX_ORIGINS, t)])
            moved = (kP > prog[0]) | ~behind                    # à jour = pas bloqué
            prog[1][moved] = t
            prog[0] = np.maximum(prog[0], kP)
            if not behind.any():
                continue
            a = w.agents.get(P)
            hp = a.heard if a is not None and t - a.t <= 2 else ()
            cand = np.isin(ida, hp) & (ida != P)
            best = np.maximum(key_me, key[cand].max(axis=0)) if cand.any() else key_me
            best_id = np.where(best == key_me, me, -1)
            stalled = t - prog[1] > cfg.stall
            for o in np.nonzero(behind & ((best_id == me) | stalled))[0].tolist():
                lagging.setdefault(o, []).append(int(kP[o]))
        out = []
        for o, lag in lagging.items():
            m = int(mine[o])
            base = min(lag)
            # continuer une plage déjà partie (par moi, ou entendue chez un voisin) au lieu de la renvoyer
            cont = base
            cur = self._cursor.get((stream, o))
            if cur is not None and t - cur[1] < cfg.cooldown:
                cont = max(cont, cur[0])
            oh = self._overheard.get((stream, o))
            if oh is not None and t - oh[2] < cfg.cooldown and oh[0] <= base:
                cont = max(cont, oh[1])
            if cont > base:
                if cont >= m:
                    continue                                   # tout est parti : on attend l'accusé
                base = cont
            out.append((0 if o == me else 1, -len(lag), o, base))
        out.sort()
        return [(p, o, b) for _, p, o, b in out]

    def build_packet(self, t: int) -> Packet:
        w, cfg = self.world, self.cfg
        fresh = {s: d for s, (th, d) in self._peers.items() if t - th <= cfg.peer_ttl}
        peers = list(fresh.values())
        agents = [w.agents[w.me]] if w.me in w.agents else []   # heartbeat toujours émis
        objects, patches, obj_ranges, map_ranges = [], [], [], []
        if peers:
            if t % cfg.meta_period == 0:
                agents += [a for a in w.others() if any(p.agent_seq.get(a.origin, -1) < a.seq for p in peers)]
            # ---- objets : tout ce qui manque, par origine et tampons croissants
            budget = cfg.max_objects
            for _, o, base in self._lagging("obj", w.obj_known, {s: d.obj_known for s, d in fresh.items()}, t):
                if budget <= 0:
                    break
                tr = w.tracks_after(o, base)[:budget]
                if not tr:
                    continue
                objects += tr
                budget -= len(tr)
                # plage déclarée complète : jamais au-delà de ce que je connais moi-même sans trou (je peux détenir
                # une donnée plus récente reçue hors séquence : je l'envoie, mais je ne garantis pas l'intervalle)
                hi = min(tr[-1].rev, int(w.obj_known[o]))
                if hi > base:
                    obj_ranges.append((o, base, hi))
                    self._cursor[("obj", o)] = (hi, t)
            # ---- tuiles : budget partagé équitablement entre origines (remplissage par niveau)
            lag = self._lagging("map", w.map.known, {s: d.map_known for s, d in fresh.items()}, t)
            cands = [(o, base, w.map.tiles_after(o, base)) for _, o, base in lag]
            cands = [c for c in cands if c[2]]
            quota = self._fair_share([len(c[2]) for c in cands], cfg.max_patches)
            for (o, base, tiles), q in zip(cands, quota):
                if q <= 0:
                    continue
                sel = tiles[:q]
                patches += [w.map.export_patch(o, cx, cy) for _, cx, cy in sel]
                hi = min(sel[-1][0], int(w.map.known[o]))
                if hi > base:
                    map_ranges.append((o, base, hi))
                    self._cursor[("map", o)] = (hi, t)
        self.stats["sent_patches"] += len(patches)
        self.stats["sent_objects"] += len(objects)
        return Packet(w.me, t, w.digest(t), tuple(agents), tuple(objects), tuple(patches),
                      tuple(obj_ranges), tuple(map_ranges))

    @staticmethod
    def _fair_share(needs: list[int], budget: int) -> list[int]:
        """Partage max-min équitable de `budget` entre des demandes (dans l'ordre de priorité pour le reste)."""
        q = [0] * len(needs)
        left, active = budget, [i for i, n in enumerate(needs) if n > 0]
        while left > 0 and active:
            share = max(left // len(active), 1)
            for i in list(active):
                give = min(share, needs[i] - q[i], left)
                q[i] += give
                left -= give
                if q[i] >= needs[i]:
                    active.remove(i)
                if left <= 0:
                    break
        return q
