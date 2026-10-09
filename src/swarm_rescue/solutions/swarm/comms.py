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

from .schema import Packet
from .world import WorldModel


@dataclass
class GossipConfig:
    max_patches: int = 6        # tuiles de carte par pas (~256 octets chacune)
    peer_ttl: int = 5           # un voisin non entendu depuis N pas n'est plus pris en compte
    resend_cooldown: int = 3    # évite de réémettre la même tuile avant l'arrivée de l'accusé


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
        self._sent: dict = {}    # (origin, cx, cy) -> (rev, t)
        self.stats: Counter = Counter()

    def ingest(self, packets: Iterable[Packet], t: int) -> None:
        w = self.world
        for p in packets:
            if p.sender == w.me:
                continue
            self._peers[p.sender] = (t, p.digest)
            self.stats["agents"] += sum(w.merge_agent(a) for a in p.agents)
            self.stats["objects"] += sum(w.merge_object(o) for o in p.objects)
            self.stats["patches"] += sum(w.map.apply_patch(x) for x in p.patches)

    def build_packet(self, t: int) -> Packet:
        w, cfg = self.world, self.cfg
        peers = [d for th, d in self._peers.values() if t - th <= cfg.peer_ttl]
        agents = [w.agents[w.me]] if w.me in w.agents else []   # heartbeat toujours émis
        objects, patches = [], []
        if peers:
            agents += [a for a in w.others()
                       if any(p.agent_seq.get(a.origin, -1) < a.seq for p in peers)]
            objects = [o for o in w.objects.values()
                       if any(p.obj_rev.get(o.key, 0) < o.rev for p in peers)]
            patches = self._select_patches(peers, t)
        return Packet(w.me, t, w.digest(t), tuple(agents), tuple(objects), tuple(patches))

    def _select_patches(self, peers, t: int):
        w, cfg = self.world, self.cfg
        score: dict = {}
        for p in peers:
            for o, cx, cy, lag in w.map.newer_than(p.map_rev):
                n, l = score.get((o, cx, cy), (0, 0))
                score[(o, cx, cy)] = (n + 1, max(l, lag))
        cands = []
        for key, (n, lag) in score.items():
            rev = int(w.map.rev[key])
            prev = self._sent.get(key)
            if prev and prev[0] == rev and t - prev[1] < cfg.resend_cooldown:
                continue
            cands.append((-n, -lag, key))   # d'abord ce qui manque à le plus de voisins
        cands.sort()
        out = []
        for _, _, key in cands[:cfg.max_patches]:
            out.append(w.map.export_patch(*key))
            self._sent[key] = (int(w.map.rev[key]), t)
        return out
