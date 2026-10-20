"""Snapshot a controlled team-diversity experiment and generate disjoint pools."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import shlex
import sys

from .generalization import checksum, party_identity, read_json, team_files


def setup(root, source_run, output, *, counts=(3000, 500, 500), team_size=3,
          seed=26092801, episodes=1000, timesteps=500000, learning_rate=.0001,
          environments=64, exclude_dirs=(), exclude_suites=()):
    from .baseline_setup import _suite
    from .evaluation import digest, validate_suite
    from .linked_reward import LinkedRewardConfig
    from .periodic_evaluation import suite_opponent_path
    from .team_generation import generate, load_catalog

    root, source_run, output = [Path(p).expanduser().resolve() for p in (root, source_run, output)]
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if any(output.is_relative_to(Path(p).expanduser().resolve()) for p in (source_run, *exclude_dirs)):
        raise ValueError("output must be outside source/exclusion directories")
    config = read_json(source_run / "config.json")
    if config.get("backend") != "showdown":
        raise ValueError("source must be a Showdown training run")
    saved = config.get("saved_args", config)
    suite_path = source_run / "inputs/evaluation/suite.json"
    original_suite = validate_suite(read_json(suite_path))
    if original_suite["backend"] != "showdown":
        raise ValueError("source evaluation must use Showdown")
    pool = original_suite.get("opponents", {})
    if {e["group"] for e in pool.values()} != {"random", "dumb", "smart"}:
        raise ValueError("source suite must contain Random, Dumb and Smart opponents")
    reward = asdict(LinkedRewardConfig.load(source_run / "inputs/reward.json"))
    learner = (source_run / "checkpoints/best.zip").read_bytes()
    best = read_json(source_run / "checkpoints/best.json")
    if checksum(learner) != best["checkpoint_sha256"]:
        raise ValueError("source best.zip does not match best.json")
    frozen_info = config.get("frozen_opponent")
    if not frozen_info:
        raise ValueError("source must have a frozen training opponent")
    frozen_path = source_run / "inputs/frozen-opponent.zip"
    frozen = frozen_path.read_bytes()
    if checksum(frozen) != frozen_info["sha256"]:
        raise ValueError("frozen training opponent checksum mismatch")
    opponent_bytes = {}
    for key, entry in pool.items():
        if entry["kind"] == "checkpoint":
            data = suite_opponent_path(suite_path, entry).read_bytes()
            if checksum(data) != entry["sha256"]:
                raise ValueError(f"evaluation opponent checksum mismatch: {key}")
            opponent_bytes[key] = data

    known, exclusions = set(), []
    entries = config.get("teams", {}).get("teams", [])
    if not entries:
        raise ValueError("source has no recorded training team inventory")
    for entry in entries:
        data = Path(entry["path"]).read_bytes()
        if checksum(data) != entry["sha256"]:
            raise ValueError(f"recorded training team changed: {entry['path']}")
        known.add(party_identity(json.loads(data)))
        exclusions.append(entry)
    for path in [suite_path, *map(Path, exclude_suites)]:
        suite = validate_suite(read_json(path))
        if suite["backend"] != "showdown":
            raise ValueError("exclusion suites must use Showdown")
        known.update(party_identity(a["team"]) for a in suite["assets"].values())
        exclusions.append(dict(path=str(path.resolve()), suite_sha256=digest(suite)))
    for directory in exclude_dirs:
        for path in team_files(directory):
            data = path.read_bytes()
            known.add(party_identity(json.loads(data)))
            exclusions.append(dict(path=str(path), sha256=checksum(data)))

    catalog, templates = load_catalog(root / "config/generation")
    options = dict(team_size=team_size, level=50, competitive_rate=.6, chaos_rate=.2,
                   competitive_item_rate=.8, species_pool="all", iv_mode="perfect", ev_mode="trained")
    selected, rejected = [], Counter()
    used = set(known)
    for attempt in range(20):
        remaining = sum(counts) - len(selected)
        if not remaining:
            break
        teams, _ = generate(catalog, templates, count=remaining, seed=seed + attempt * 1000003,
                            prefix=f"batch{attempt}", **options)
        for team in teams:
            if any(len(p["moves"]) != 4 for p in team["party"]):
                rejected["fewer_than_four_moves"] += 1
                continue
            signature = party_identity(team)
            if signature in used:
                rejected["known_or_duplicate_team"] += 1
                continue
            used.add(signature)
            selected.append(team)
    if len(selected) != sum(counts):
        raise ValueError("could not fill disjoint team pools after 20 deterministic batches")
    random.Random(seed ^ 0x5419).shuffle(selected)
    output.mkdir(parents=True, exist_ok=False)

    def put(name, data):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(json.dumps(data, indent=2) + "\n")
        return path

    learner_path = put("inputs/starting.zip", learner)
    frozen_path = put("inputs/frozen-opponent.zip", frozen)
    reward_path = put("inputs/reward.json", reward)
    put("inputs/source-config.json", config)
    put("inputs/source-best.json", best)
    put("inputs/source-validation.json", original_suite)
    opponents = {}
    for key, entry in pool.items():
        row = dict(entry)
        if key in opponent_bytes:
            row["source"] = str(put(f"inputs/opponents/{row['sha256']}.zip", opponent_bytes[key]))
        opponents[key] = row
    groups, inventory, cursor = {}, {}, 0
    for split, count in zip(("training", "validation", "test"), counts):
        groups[split], inventory[split] = [], []
        for index, team in enumerate(selected[cursor:cursor+count], 1):
            team["name"] = f"{split}_{index:04d}"
            path = put(f"teams/{split}/{team['name']}.json", team)
            groups[split].append(path)
            inventory[split].append(dict(path=str(path), sha256=checksum(path.read_bytes()), signature=party_identity(team)))
        cursor += count
    suite_hashes = {}
    for offset, split in enumerate(("validation", "test"), 1):
        suite = _suite(groups[split], opponents, episodes=episodes, seed=seed+offset,
                       max_steps=original_suite["max_steps"])
        put(f"suites/{split}.json", suite)
        suite_hashes[split] = digest(suite)

    train = [sys.executable, str(root / "mgba/train_ppo.py"), "--backend", "showdown",
             "--resume", str(learner_path), "--opponent-checkpoint", str(frozen_path),
             "--reward-config", str(reward_path), "--team-dir", str(output / "teams/training"),
             "--output-dir", str(output / "train"), "--if-exists", "cancel",
             "--learning-rate", str(learning_rate), "--total-timesteps", str(timesteps),
             "--environments", str(environments), "--showdown-workers", "process",
             "--seed", str(saved.get("seed", 1)), "--max-steps", str(saved.get("max_steps", 200)),
             "--torch-threads", str(saved.get("torch_threads", 1)), "--device", str(saved.get("device", "cpu")),
             "--opponent-switch-chance", str(saved.get("opponent_switch_chance", 0)),
             "--opponent-switch-cooldown", str(saved.get("opponent_switch_cooldown", 2)),
             "--eval-suite", str(output / "suites/validation.json"), "--eval-every", "100000",
             "--eval-environments", str(environments), "--keep-evaluations", "2",
             "--checkpoint-every", "100000", "--keep-checkpoints", "2", "--latest-every-updates", "5",
             "--no-readable-log", "--battle-log", "--no-step-logs", "--log-max-mb", "2", "--log-backups", "1"]
    if saved.get("opponent_stochastic", False):
        train.append("--opponent-stochastic")
    if saved.get("mirror_teams", False):
        train.append("--mirror-teams")
    train.extend(["--node", str(saved.get("node", "node")), "--command-timeout", str(saved.get("command_timeout", 120))])
    header = "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(root)) + "\n"
    (output / "train.sh").write_text(header + "exec " + shlex.join(train) + "\n")
    commands = []
    for label, checkpoint in (("starting", learner_path), ("candidate", output / "train/checkpoints/best.zip")):
        commands.append(shlex.join([sys.executable, str(root / "mgba/evaluate_ppo.py"),
            "--checkpoint", str(checkpoint), "--suite", str(output / "suites/test.json"),
            "--reward-config", str(reward_path), "--output-dir", str(output / "final-test" / label),
            "--environments", str(environments), "--showdown-workers", "process",
            "--node", str(saved.get("node", "node")), "--device", str(saved.get("device", "cpu")),
            "--torch-threads", str(saved.get("torch_threads", 1)),
            "--no-readable-log", "--battle-log", "--no-step-logs", "--keep-episode-logs", "0",
            "--log-max-mb", "2", "--log-backups", "1"]))
    report_code = ("from pathlib import Path; from firered_rl.baseline_report import build_baseline_report, write_baseline_report; "
                   "import sys; p=Path(sys.argv[1]); r=build_baseline_report({k:{'reserved_test':p/k} for k in "
                   "('starting','candidate')}); write_baseline_report(r,p); print(p/'baseline-report.md')")
    commands.append(shlex.join([sys.executable, "-c", report_code, str(output / "final-test")]))
    guard = ("test -f " + shlex.quote(str(output / "train/checkpoints/best.zip")) +
             " || { echo 'Finish training and checkpoint selection before running test.sh' >&2; exit 1; }\n")
    (output / "test.sh").write_text(header + guard + "\n".join(commands) + "\n")
    styles = {split: dict(Counter(m["style"] for p in paths for m in read_json(p)["generation"]["members"]))
              for split, paths in groups.items()}
    manifest = dict(version=1, preparation_status="completed", execution_status_source="train/summary.json and final-test/baseline-report.json",
        source_run=str(source_run),
        source_suite_sha256=digest(original_suite), learner_sha256=checksum(learner), frozen_sha256=checksum(frozen),
        reward=reward, seed=seed, generation_options=options, counts=dict(zip(groups, counts)),
        realized_styles=styles, rejected=dict(rejected), checked_sources=exclusions, teams=inventory,
        suites=suite_hashes, training_command=train,
        generation_sources={p.name: checksum(p.read_bytes()) for p in (root / "config/generation").glob("*.json")},
        limits=["Overlap check covers recorded source inputs and supplied exclusions, not all checkpoint ancestry.",
                "Distinct sets may share species or curated moves; this is configuration-level separation.",
                "Generator composition also changes relative to the old category-based pool; size alone is not isolated."])
    put("experiment.json", manifest)
    return manifest
