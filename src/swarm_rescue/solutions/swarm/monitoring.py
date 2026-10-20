"""Mission telemetry and consistency checks for swarm runs.

The monitor is deliberately independent of the simulator.  A :class:`Brain`
records one JSON object per simulation step when ``SWARM_MONITOR`` is set; the
CLI can then inspect or convert that stream without opening a GUI.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .schema import MissionKind, ObjKind, ObjState


class MissionMonitor:
    """Record per-drone state and emit actionable anomaly events."""

    def __init__(self, path: str | None = None, stale_after: int = 20, keep: int = 200):
        self.path = Path(path) if path else None
        self.stale_after = stale_after
        self._roles: dict[int, MissionKind] = {}
        self._history: dict[int, deque[MissionKind]] = defaultdict(lambda: deque(maxlen=8))
        self.records: deque[dict[str, Any]] = deque(maxlen=keep)   # derniers enregistrements seulement (mémoire bornée)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, brain, t: int) -> dict[str, Any] | None:
        if self.path is None:          # désactivé : aucun coût par pas (carte, objets) ni mémoire
            return None
        w, mission = brain.world, brain.mission
        role = mission.kind
        previous = self._roles.get(brain.me)
        self._roles[brain.me] = role
        hist = self._history[brain.me]
        hist.append(role)
        events: list[str] = []
        if previous is not None and previous != role:
            events.append(f"role-change:{previous.name}->{role.name}")

        target_age = None
        if mission.target is not None:
            candidates = [o for o in w.objects_view(ObjKind.WOUNDED)
                          if o.state in (ObjState.SEEN, ObjState.CARRIED)]
            if candidates:
                target_age = max(0, t - min(candidates, key=lambda o:
                                             (o.pos[0] - mission.target[0]) ** 2 +
                                             (o.pos[1] - mission.target[1]) ** 2).t_seen)
                if role in (MissionKind.GOTO_WOUNDED, MissionKind.CARRY) and target_age > self.stale_after:
                    events.append("rescuer-on-stale-info")
        # Repeated role changes are an oscillation, not just a role transition.
        if len(hist) >= 6 and sum(a != b for a, b in zip(hist, list(hist)[1:])) >= 5:
            events.append("role-oscillation")
        peers = {str(k): max(0, t - a.t) for k, a in w.agents.items() if k != brain.me}
        stale_peers = [k for k, age in peers.items() if age > self.stale_after]
        if stale_peers:
            events.append("stale-agent-info")
        view = w.map.view()
        rec = {
            "t": int(t), "drone": int(brain.me), "mission": role.name,
            "mission_source": str(getattr(brain, "mission_src", "")),
            "target_age": target_age, "heard": sorted(int(x) for x in brain.gossip.heard),
            "gossip": dict(brain.gossip.stats), "peer_age": peers,
            "stale_peers": stale_peers, "role_changed": previous != role if previous is not None else False,
            "exploration": {"known_cells": int(np.count_nonzero(view)),
                            "occupied_cells": int(np.count_nonzero(view > 0)),
                            "last_explore": int(getattr(brain, "_last_explore", -1))},
            "events": events,
        }
        self.records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rec, separators=(",", ":")) + "\n")
        return rec


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect swarm mission telemetry (JSONL).")
    parser.add_argument("input", type=Path, help="JSONL file produced with SWARM_MONITOR")
    parser.add_argument("--csv", type=Path, help="Export a flattened CSV")
    parser.add_argument("--tail", type=int, default=20, help="Number of records to display")
    args = parser.parse_args(argv)
    rows = _read(args.input)
    if args.csv:
        fields = ["t", "drone", "mission", "mission_source", "target_age", "heard", "events"]
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            out = csv.DictWriter(stream, fieldnames=fields)
            out.writeheader()
            for row in rows:
                out.writerow({k: json.dumps(row[k]) if isinstance(row[k], (list, dict)) else row.get(k)
                              for k in fields})
    for row in rows[-max(args.tail, 0):]:
        print(f"t={row['t']:>4} drone={row['drone']:>2} {row['mission']:<12} "
              f"heard={row['heard']} age={row['target_age']} events={','.join(row['events'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
