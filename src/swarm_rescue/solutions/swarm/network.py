"""Image du réseau radio telle qu'un drone se la représente (à partir de SON WorldModel).

Nœuds : les drones dont l'état est assez récent. Arêtes :
  - "entendues" : un des deux drones a déclaré avoir reçu un paquet de l'autre (AgentState.heard) -> lien réel ;
  - "prévues"   : distance < link_range entre positions annoncées -> lien probable (pas encore confirmé).
La radio du simulateur ne dépend que de la distance (pas des murs) : portée 250 px, on garde une marge.

Aucune écriture : c'est une vue de lecture, recalculée quand on en a besoin (planification, debug).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .world import WorldModel


@dataclass
class NetworkView:
    nodes: dict = field(default_factory=dict)        # origin -> (x, y)
    heard: set = field(default_factory=set)          # {(a, b)} a < b, liens réels
    predicted: set = field(default_factory=set)      # {(a, b)} a < b, liens probables non confirmés

    def neighbors(self, n: int) -> set:
        out = set()
        for a, b in self.heard | self.predicted:
            if a == n:
                out.add(b)
            elif b == n:
                out.add(a)
        return out

    def components(self) -> list[set]:
        seen, comps = set(), []
        for n in sorted(self.nodes):
            if n in seen:
                continue
            comp, stack = set(), [n]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(self.neighbors(x) - comp)
            seen |= comp
            comps.append(comp)
        return comps


def build_network(world: WorldModel, t: int, fresh: int = 60, link_range: float = 220.0) -> NetworkView:
    net = NetworkView()
    agents = {o: a for o, a in world.agents.items() if t - a.t <= fresh}
    net.nodes = {o: a.pos for o, a in agents.items()}
    for o, a in agents.items():
        for h in a.heard:
            if h in agents and h != o:
                net.heard.add((min(o, h), max(o, h)))
    ids = sorted(agents)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if (a, b) in net.heard:
                continue
            pa, pb = net.nodes[a], net.nodes[b]
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) < link_range:
                net.predicted.add((a, b))
    return net
