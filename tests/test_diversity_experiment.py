import json
from pathlib import Path
import subprocess

import pytest
pytest.importorskip("sb3_contrib")
from firered_rl.diversity_experiment import setup
from firered_rl.generalization import checksum, party_identity
from test_generalization import comparison_inputs, write
from test_training_continue_cli import training

ROOT = Path(__file__).resolve().parents[1]


def source_metadata(run, opponent):
    path = run / "config.json"
    config = json.loads(path.read_text())
    config["frozen_opponent"] = {"sha256": checksum(opponent.read_bytes())}
    config["saved_args"] = {"seed": 19, "max_steps": 2, "torch_threads": 1, "device": "cpu"}
    write(path, config)
    (run / "inputs/frozen-opponent.zip").write_bytes(opponent.read_bytes())


def test_reproducible_disjoint_pools_and_provenance(comparison_inputs, tmp_path):
    runs, _, _, opponent = comparison_inputs
    source_metadata(runs[1], opponent)
    a = setup(ROOT, runs[1], tmp_path / "a", counts=(20, 10, 10), episodes=2, environments=2)
    b = setup(ROOT, runs[1], tmp_path / "b", counts=(20, 10, 10), episodes=2, environments=2)
    groups = [{r['signature'] for r in a['teams'][split]} for split in ('training', 'validation', 'test')]
    assert [len(s) for s in groups] == [20, 10, 10]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for split in a['teams']:
        assert [r['sha256'] for r in a['teams'][split]] == [r['sha256'] for r in b['teams'][split]]
        for row in a['teams'][split]:
            team = json.loads(Path(row['path']).read_text())
            assert all(len(p['moves']) == 4 for p in team['party'])
    with pytest.raises(ValueError, match="output already exists"):
        setup(ROOT, runs[1], tmp_path / "a", counts=(1, 1, 1), episodes=2)
    c = setup(ROOT, runs[1], tmp_path / "c", counts=(20, 10, 10), episodes=2,
              exclude_dirs=[tmp_path / "a/teams"])
    assert c['rejected']['known_or_duplicate_team'] > 0
    assert not set.union(*groups) & {r['signature'] for rows in c['teams'].values() for r in rows}


def test_generated_scripts_run_real_training_and_test(training, comparison_inputs, tmp_path):
    runs, _, _, opponent = comparison_inputs
    opponent.write_bytes(training.frozen.read_bytes())
    run = runs[1]
    source_metadata(run, opponent)
    suite = json.loads((run / "inputs/evaluation/suite.json").read_text())
    for entry in suite['opponents'].values():
        if entry['kind'] == 'checkpoint':
            entry['sha256'] = checksum(opponent.read_bytes())
    write(run / "inputs/evaluation/suite.json", suite)
    data = (training.run / "checkpoints/final.zip").read_bytes()
    (run / "checkpoints/best.zip").write_bytes(data)
    write(run / "checkpoints/best.json", {"checkpoint_sha256": checksum(data)})
    output = tmp_path / "experiment with spaces"
    setup(ROOT, run, output, counts=(3, 2, 2), episodes=2, timesteps=4, environments=2)
    # Source weights can disappear after setup; generated commands use snapshots.
    (run / "checkpoints/best.zip").unlink()
    opponent.unlink()
    for script in ('train.sh', 'test.sh'):
        result = subprocess.run(['bash', str(output / script)], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((output / "train/config.json").read_text())
    assert config['effective_ppo']['learning_rate'] == .0001
    assert config['effective_ppo']['n_steps'] == 4
    assert (output / "final-test/baseline-report.md").is_file()
    assert (output / "inputs/starting.zip").read_bytes() == data
