"""Validate v3 configs and optionally solve GT from measured state exports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomind.episodes import ContractError, EpisodeResolver, Snapshot, solve
from geomind.episodes.resolver import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_id")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, help="Backend export: current snapshot and history map")
    args = parser.parse_args()
    try:
        resolved = EpisodeResolver(args.root).resolve(args.episode_id)
        if args.state:
            state = load_json(args.state)
            if (not isinstance(state, dict) or not {"current", "history"} <= state.keys()
                    or state.keys() - {"current", "history", "backend", "physical_clock"}
                    or not isinstance(state["history"], dict)):
                raise ContractError("State export requires current and history; optional backend/physical_clock are allowed")
            result = solve(resolved, Snapshot.from_dict(state["current"]), history={
                key: Snapshot.from_dict(value) for key, value in state["history"].items()
            })
            result["evidence_backend"] = state.get("backend", "external")
        else:
            result = {
                "configuration_valid": True,
                "ground_truth_computed": False,
                "contract_hash": resolved.contract_hash,
                "source_hashes": resolved.source_hashes,
                "episode": resolved.episode,
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ContractError, OSError, ValueError) as exc:
        parser.exit(2, f"Episode error: {exc}\n")


if __name__ == "__main__":
    main()
