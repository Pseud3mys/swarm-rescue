"""Protocole de gossip (anti-entropie) pour un canal BROADCAST à portée limitée.

Principe : chaque pas, je diffuse (1) mon résumé `Digest`, (2) ma ligne `AgentState`,
(3) ce que mes voisins récents n'ont pas d'après LEUR dernier résumé, dans la limite d'un budget.
Tout drone relaie ce qu'il a, quelle que soit l'origine => un drone immobile au centre
devient naturellement un messager. Zéro état partagé, zéro accusé de réception nécessaire :
les pertes (no-com zone, sortie de portée) sont rattrapées à la rencontre suivante.

Ce fichier ne connaît ni la carte ni la planification : il ne manipule que l'API de WorldModel.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schema import Packet
from .world import WorldModel


@dataclass
class GossipConfig:
    max_patches: int = 40       # tuiles de carte par pas (~256 octets chacune). Le simulateur n'impose aucune limite ;
                                # à 6, le plafond était atteint à chaque pas et les cartes prenaient 20-140 tuiles de retard.
    peer_ttl: int = 5           # un voisin non entendu depuis N pas n'est plus pris en compte
    resend_cooldown: int = 3    # évite de réémettre la même tuile avant l'arrivée de l'accusé
    meta_period: int = 3        # états d'agents relayés et objets : tous les N pas (mon propre état : chaque pas)


def extract_packets(received: Iterable) -> list[Packet]:
    """Adaptateur : à ajuster une fois vu le format réel de `received_messages` du simulateur.
    Accepte un Packet nu ou un tuple/liste qui en contient un."""
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
        # dernière émission (par moi OU entendue chez un voisin) de chaque tuile : (révision, pas)
        rev = world.map.rev
        self._sent_rev = np.zeros(rev.shape, np.int64)
        self._sent_t = np.full(rev.shape, -10 ** 9, np.int64)
        self.stats: Counter = Counter()
        self.heard: set = set()  # émetteurs entendus au dernier ingest (= liens radio réels à ce pas)

    def ingest(self, packets: Iterable[Packet], t: int) -> None:
        w = self.world
        m = w.map
        self.heard = set()
        for p in packets:
            if p.sender == w.me:
                continue
            self.heard.add(p.sender)
            self._peers[p.sender] = (t, p.digest)
            self.stats["agents"] += sum(w.merge_agent(a) for a in p.agents)
            self.stats["objects"] += sum(w.merge_object(o) for o in p.objects)
            for x in p.patches:
                if not (0 <= x.origin < m.max_origins and 0 <= x.cx < m.ncx and 0 <= x.cy < m.ncy):
                    continue
                # suppression des doublons : une tuile qu'un voisin vient de diffuser, nos voisins communs l'ont
                # sans doute reçue -> on ne la renvoie pas tout de suite (le délai de garde reste court : si elle
                # manque encore, le résumé du voisin le dira et elle repartira)
                if x.rev >= self._sent_rev[x.origin, x.cx, x.cy]:
                    self._sent_rev[x.origin, x.cx, x.cy] = x.rev
                    self._sent_t[x.origin, x.cx, x.cy] = t
                if x.rev > m.rev[x.origin, x.cx, x.cy]:
                    self.stats["patches"] += m.apply_patch(x)

    def build_packet(self, t: int) -> Packet:
        w, cfg = self.world, self.cfg
        peers = [d for th, d in self._peers.values() if t - th <= cfg.peer_ttl]
        agents = [w.agents[w.me]] if w.me in w.agents else []   # heartbeat toujours émis
        objects, patches = [], []
        if peers:
            if t % cfg.meta_period == 0:       # états des autres drones et objets : pas besoin de chaque pas
                agents += [a for a in w.others()
                           if any(p.agent_seq.get(a.origin, -1) < a.seq for p in peers)]
                objects = [o for o in w.objects.values()
                           if any(p.obj_rev.get(o.key, 0) < o.rev for p in peers)]
            patches = self._select_patches(peers, t)
        return Packet(w.me, t, w.digest(t), tuple(agents), tuple(objects), tuple(patches))

    def _select_patches(self, peers, t: int):
        """Tuiles plus récentes chez moi que chez au moins un voisin ; d'abord celles qui manquent au plus de
        voisins, puis les plus en retard. Vectorisé sur tout le tableau des révisions."""
        w, cfg = self.world, self.cfg
        rev = w.map.rev.astype(np.int64)
        need = np.zeros(rev.shape, np.int32)
        lag = np.zeros(rev.shape, np.int64)
        for p in peers:
            if p.map_rev.shape != rev.shape:
                continue
            d = rev - p.map_rev
            need += d > 0
            np.maximum(lag, d, out=lag)
        need[(self._sent_rev >= rev) & (t - self._sent_t < cfg.resend_cooldown)] = 0
        idx = np.flatnonzero(need)
        if idx.size == 0:
            return []
        order = np.lexsort((-lag.ravel()[idx], -need.ravel()[idx]))[:cfg.max_patches]
        out = []
        for i in idx[order]:
            o, cx, cy = (int(v) for v in np.unravel_index(i, rev.shape))
            out.append(w.map.export_patch(o, cx, cy))
            self._sent_rev[o, cx, cy] = rev[o, cx, cy]
            self._sent_t[o, cx, cy] = t
        return out
