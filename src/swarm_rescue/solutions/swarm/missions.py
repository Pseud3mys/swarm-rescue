"""Choix de la mission (QUOI faire). Ne calcule aucun chemin : lit les coûts de PlanGrid et la mémoire du monde.

Priorités : porter un blessé > aller chercher un blessé libre > explorer une frontière > rentrer.
Coordination sans serveur : chaque drone lit les missions annoncées par les autres (AgentState) et
cède un blessé / une frontière à celui qui y est manifestement mieux placé ; l'égalité est départagée
par l'identifiant (tous les drones arrivent donc à la même conclusion sans se parler davantage).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .pathing import PlanGrid
from .schema import Mission, MissionKind, ObjKind, ObjState, Pose
from .world import WorldModel


class MissionPlanner:
    def __init__(self, world: WorldModel, params: Params):
        self.w, self.p = world, params
        self.blacklist: list[tuple[np.ndarray, int]] = []     # (position, expire_t)

    def ban(self, pos, until: int) -> None:
        self.blacklist.append((np.asarray(pos, float), until))

    def _banned(self, pos, t: int) -> bool:
        self.blacklist = [(q, e) for q, e in self.blacklist if e > t]
        return any(np.linalg.norm(q - pos) < 40.0 for q, _ in self.blacklist)

    # ---------------------------------------------------------------------------
    def home(self) -> Optional[np.ndarray]:
        h = self.w.objects_view(ObjKind.RETURN_AREA)
        return np.array(h[0].pos) if h else None

    def rescue_point(self) -> Optional[np.ndarray]:
        r = self.w.objects_view(ObjKind.RESCUE_CENTER)
        return np.array(r[0].pos) if r else self.home()

    def candidates(self, t: int, pose: Pose, grid: PlanGrid, carrying: bool,
                   current: Optional[Mission]) -> list[Mission]:
        """Missions possibles par ordre de préférence (l'appelant prend la première qui a un chemin)."""
        p, w = self.p, self.w
        out: list[Mission] = []
        if carrying:
            tgt = self.rescue_point()
            if tgt is not None:
                return [Mission(MissionKind.CARRY, (float(tgt[0]), float(tgt[1])), t)]
            return [Mission(MissionKind.EXPLORE, None, t)]           # centre inconnu : on le cherche

        # ---- blessés libres
        others = [a for a in w.others() if t - a.t < 400]
        # un blessé "porté" n'est vraiment pris que si un drone est en mission de portage (sinon : lâché, disponible)
        carriers = [a for a in others if a.mission.kind == MissionKind.CARRY]
        scored = []
        for o in w.objects_view(ObjKind.WOUNDED):
            if o.state in (ObjState.RESCUED, ObjState.TAKEN):
                continue
            if o.state == ObjState.CARRIED and carriers:
                continue
            pos = np.array(o.pos)
            if self._banned(pos, t):
                continue
            mine = grid.cost_to(pos)
            if not math.isfinite(mine):
                continue
            lost = False
            for a in others:
                if a.mission.kind == MissionKind.GOTO_WOUNDED and a.mission.target is not None \
                        and np.linalg.norm(np.array(a.mission.target) - pos) < 50.0:
                    theirs = 1.3 * np.linalg.norm(np.array(a.pos) - pos)
                    if theirs < 0.9 * mine or (abs(theirs - mine) <= 0.1 * mine and a.origin < w.me):
                        lost = True
                        break
            if lost:
                continue
            sticky = 0.7 if (current and current.kind == MissionKind.GOTO_WOUNDED and current.target
                             and np.linalg.norm(np.array(current.target) - pos) < 50.0) else 1.0
            scored.append((mine * sticky, o.pos))
        scored.sort(key=lambda s: s[0])
        out += [Mission(MissionKind.GOTO_WOUNDED, (float(q[0]), float(q[1])), t) for _, q in scored[:2]]
        if out:
            return out

        # ---- exploration
        fr = grid.frontiers(p.frontier_min_cells)
        explore_targets = [np.array(a.mission.target) for a in others
                           if a.mission.kind == MissionKind.EXPLORE and a.mission.target is not None]
        cand = []
        for f in fr:
            pos = f.pos
            c = grid.m.cell_to_world(np.array([f.cell[0]]), np.array([f.cell[1]]))[0]
            if self._banned(c, t):
                continue
            eff = f.cost
            if any(np.linalg.norm(c - q) < 160.0 for q in explore_targets):
                eff += 250.0                                         # un autre drone y va déjà
            if current and current.kind == MissionKind.EXPLORE and current.target \
                    and np.linalg.norm(np.array(current.target) - c) < 70.0:
                eff *= p.switch_gain                                 # hystérésis
            cand.append((eff, c))
        cand.sort(key=lambda s: s[0])
        out += [Mission(MissionKind.EXPLORE, (float(c[0]), float(c[1])), t) for _, c in cand[:3]]
        if out:
            return out

        h = self.home()
        if h is not None and np.linalg.norm(h - pose.pos) > 60.0:
            return [Mission(MissionKind.RETURN, (float(h[0]), float(h[1])), t)]
        return [Mission(MissionKind.IDLE, None, t)]
