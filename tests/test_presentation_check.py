"""Frozen selection, novel teams, model exports, and complete saved evidence."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_presentation_check as runner
from test_transfer_lab import checkpoint, write
from test_replacement_diagnostics import suite
from test_replacement_diagnostic_runner import saved_check
from firered_rl.broad_team_curriculum import composition_signature, semantic_signature


@pytest.fixture
def source(tmp_path, monkeypatch):
    import run_bench_order_ablation
    import firered_rl.showdown_env
    root = tmp_path / 'source'
    actual_gate = run_bench_order_ablation.validate_saved_checks
    monkeypatch.setattr(run_bench_order_ablation, 'validate_saved_checks',
        lambda path, records: {'verified': True} if path == root else actual_gate(path, records))
    monkeypatch.setattr(firered_rl.showdown_env, 'validate_showdown', lambda *args: None)
    frozen = suite()
    write(root / 'inputs/reward.json', {'win': 5000, 'loss': -5000})
    teams = {}
    for group in ('train', 'mirror'):
        teams[group] = []
        for i, asset in enumerate(frozen['assets'].values()):
            path = write(root / 'inputs/teams' / group / f'team-{i}.json', asset['team'])
            teams[group].append(str(path.relative_to(root)))
    suites = {}
    for name in ('validation', 'validation-original', 'seen-original', 'seen-reversed',
                 'heldout-original', 'heldout-reversed', 'mirror'):
        path = write(root / 'inputs/suites' / f'{name}.json', frozen)
        suites[name] = str(path.relative_to(root))
    models, training = {}, {}
    for arm, best_steps, final_steps, wins in (('control', 128, 512, 2), ('bench-shuffle', 256, 640, 1)):
        models[arm] = {}
        for kind, steps in (('best', best_steps), ('final', final_steps)):
            path = checkpoint(root / 'runs' / arm / 'checkpoints' / f'{kind}.zip', steps)
            models[arm][kind] = dict(checkpoint=str(path.relative_to(root)), checkpoint_sha256=runner.checksum(path),
                timesteps=steps, checks={})
        write(root / 'runs' / arm / 'checkpoints/best.json', dict(timesteps=best_steps,
            checkpoint_sha256=models[arm]['best']['checkpoint_sha256'], requested=2, wins=wins))
        training[arm] = dict(ending_timesteps=final_steps)
    selected = dict(name='control', checkpoint=models['control']['best']['checkpoint'],
        checkpoint_sha256=models['control']['best']['checkpoint_sha256'], validation_rate=1.)
    write(root / 'result.json', dict(status='completed', initial=dict(checks={}), policies=models,
        training=training, selected_by_validation=selected))
    write(root / 'plan.json', dict(version=1, experiment='bench-order-ablation', smoke=False,
        teams=teams, suites=suites, pins={str(path.relative_to(root)): runner.checksum(path)
            for path in (root / 'inputs').rglob('*') if path.is_file()}))
    return root


@pytest.fixture
def prepared(source, tmp_path):
    root = tmp_path / 'presentation'
    return root, runner.freeze(root, SimpleNamespace(source_run=source, smoke=True))


def cached_fixture(root, plan, check, *, wins=None):
    suite_data, identity = runner.diagnostic.check_context(root, plan, check)
    directory = root / 'checks' / check['label']
    result = saved_check(directory, check, suite_data, wins=wins)
    result.update(identity, result_directory=str(directory.relative_to(root)))
    write(directory / 'summary.json', result)
    return directory, result


def test_source_read_and_exports_preserve_source_and_official_selection(prepared, source):
    root, plan = prepared
    before = {str(path): runner.checksum(path) for path in source.rglob('*') if path.is_file()}
    runner.source_inputs(source)
    runner.verify(root, plan)
    assert before == {str(path): runner.checksum(path) for path in source.rglob('*') if path.is_file()}
    assert (root / 'models/selected.zip').read_bytes() == (source / 'runs/control/checkpoints/best.zip').read_bytes()
    assert set(plan['models']) == {'official', 'control-final', 'bench-shuffle-final'}
    assert plan['official_selection']['fresh_scores_can_change_selection'] is False
    assert len(plan['teams']['demo']) == 2
    assert all((root / path).is_file() for path in plan['teams']['demo'])
    assert len(plan['checks']) == 6 and all(not check['assisted'] for check in plan['checks'])
    assert {check['checkpoint'] for check in plan['checks']} == {None, 'official'}


def test_selected_arm_is_derived_from_recorded_validation(source):
    result = runner.read_json(source / 'result.json')
    record = result['policies']['bench-shuffle']['best']
    result['selected_by_validation'] = dict(name='bench-shuffle', checkpoint=record['checkpoint'],
        checkpoint_sha256=record['checkpoint_sha256'], validation_rate=.5)
    write(source / 'result.json', result)
    path = source / 'runs/control/checkpoints/best.json'
    meta = runner.read_json(path); meta['wins'] = 0; write(path, meta)
    evidence = runner.source_inputs(source)
    assert evidence['models']['official']['source_arm'] == 'bench-shuffle'
    assert evidence['models']['official']['timesteps'] == 256


@pytest.mark.parametrize('change', ['weights', 'best_hash', 'selection_hash', 'steps', 'pins', 'escape', 'incomplete'])
def test_source_rejects_changed_provenance(source, change):
    result = runner.read_json(source / 'result.json')
    if change == 'weights': (source / 'runs/control/checkpoints/best.zip').write_bytes(b'changed')
    elif change == 'best_hash':
        path = source / 'runs/control/checkpoints/best.json'; meta = runner.read_json(path)
        meta['checkpoint_sha256'] = 'bad'; write(path, meta)
    elif change == 'selection_hash': result['selected_by_validation']['checkpoint_sha256'] = 'bad'
    elif change == 'steps': result['policies']['control']['final']['timesteps'] += 1
    elif change == 'pins': write(source / 'inputs/reward.json', {'different': True})
    elif change == 'escape': result['policies']['control']['final']['checkpoint'] = '../outside.zip'
    elif change == 'incomplete': result['status'] = 'running'
    write(source / 'result.json', result)
    with pytest.raises((ValueError, OSError)): runner.source_inputs(source)


def test_fresh_sampler_is_deterministic_and_excludes_compositions_and_semantics(source):
    config = runner.read_json(ROOT / 'config/experiments/team-curriculum.json')
    excluded = runner.exclusions(runner.source_inputs(source))
    teams, metadata = runner.fresh_teams(config, excluded, 64)
    assert (teams, metadata) == runner.fresh_teams(config, excluded, 64)
    assert len({composition_signature(team) for team in teams}) == 64
    assert not set(metadata['compositions']) & set(excluded['compositions'])
    assert not set(metadata['semantics']) & set(excluded['semantics'])
    assert all(len(team['party']) == 3 and all(len(mon['moves']) == 2 for mon in team['party']) for team in teams)
    changed = copy.deepcopy(teams[0]); changed['party'].reverse()
    for mon in changed['party']: mon['moves'].reverse()
    assert composition_signature(changed) == composition_signature(teams[0])
    assert semantic_signature(changed) == semantic_signature(teams[0])
    blocked = dict(compositions=metadata['compositions'], semantics=metadata['semantics'])
    _, next_metadata = runner.fresh_teams(config, blocked, 64)
    assert not set(metadata['compositions']) & set(next_metadata['compositions'])


def test_every_source_suite_contributes_exclusions(source):
    data = runner.source_inputs(source)
    path = data['suites']['heldout-reversed']
    frozen = runner.read_json(path)
    for asset in frozen['assets'].values(): asset['team']['party'][0]['species'] = 'Sceptile'
    write(path, frozen)
    excluded = runner.exclusions(data)
    assert any('sceptile' in comp for comp in excluded['compositions'])
    assert any('suite:heldout-reversed:' in name for name in excluded['sources'])


def test_smoke_budgets_and_orders_are_paired(prepared):
    root, plan = prepared
    assert plan['settings']['fresh_teams'] == 4 and plan['settings']['battles_per_order'] == 8
    original, reversed_suite = (runner.read_json(root / plan['suites'][f'fresh-{order}']) for order in ('original', 'reversed'))
    assert len(original['cases']) == len(reversed_suite['cases']) == 8
    for a, b in zip(original['cases'], reversed_suite['cases']):
        assert all(a[key] == b[key] for key in ('episode', 'matchup', 'seed', 'candidate_player'))
        side = a['candidate_player'] - 1
        assert a['assets'][1-side] == b['assets'][1-side]
        old = original['assets'][a['assets'][side]]['team']['party']
        new = reversed_suite['assets'][b['assets'][side]]['team']['party']
        assert new == [old[0], *old[:0:-1]]


@pytest.mark.parametrize('change', ['model', 'selection', 'settings', 'checks', 'team', 'membership', 'implementation'])
def test_continue_rejects_changed_frozen_inputs(prepared, change, monkeypatch):
    root, plan = prepared
    if change == 'model': (root / 'models/selected.zip').write_bytes(b'changed')
    elif change == 'selection': plan['official_selection']['selected_by_validation']['name'] = 'other'
    elif change == 'settings': plan['settings']['battles_per_order'] += 1
    elif change == 'checks': plan['checks'][0]['checkpoint'] = 'control-final'
    elif change == 'team': write(root / plan['teams']['fresh'][0], {'party': []})
    elif change == 'membership': write(root / 'demo-teams/extra.json', {})
    elif change == 'implementation': monkeypatch.setattr(runner, 'implementation_identity', lambda: {})
    with pytest.raises(ValueError): runner.verify(root, plan)


def test_complete_resume_does_not_launch_or_reselect(prepared, monkeypatch):
    root, plan = prepared
    declaration = copy.deepcopy(plan['official_selection'])
    for check in plan['checks']:
        cached_fixture(root, plan, check, wins=set(range(1, 9)) if check['baseline'] else set())
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('Complete resume launched evaluation'))
    runner.launch(root, plan, 2)
    result = runner.finish(root, plan)
    assert result['official_selection'] == declaration
    assert result['no_training'] and result['no_reselection']
    assert len(result['saved_evidence']['checked_directories']) == 6
    runner.collect(root, plan)
    assert 'Selected PPO' in (root / 'report.md').read_text()
    assert 'Greedy damage baseline' in (root / 'report.md').read_text()
    with ZipFile(root / 'presentation-results.zip') as archive:
        assert 'models/manifest.json' in archive.namelist()
        assert not any(name.endswith('.zip') or name.startswith('console/') for name in archive.namelist())


@pytest.mark.parametrize('missing', ['episodes.jsonl', 'decision-traces.jsonl', 'decision-audit.json'])
def test_final_gate_rejects_partial_saved_checks(prepared, missing):
    root, plan = prepared
    for check in plan['checks']: cached_fixture(root, plan, check)
    directory = root / 'checks' / plan['checks'][0]['label']
    (directory / missing).unlink()
    assert runner.diagnostic.cached_check(root, plan, plan['checks'][0]) is None
    with pytest.raises(RuntimeError, match='incomplete'): runner.finish(root, plan)
    assert directory.exists()  # Preserve partial evidence for a separate retry directory.


def test_only_one_or_two_jobs_and_team_inspection(prepared, capsys):
    root, plan = prepared
    with pytest.raises(ValueError): runner.launch(root, plan, 3)
    runner.inspect(root, plan, 'all', 1, False)
    text = capsys.readouterr().out
    assert 'fresh: 4 teams' in text and 'demo: 2 teams' in text
    assert 'team-0001.json' in text


def test_source_rejects_selected_arm_below_other_validation_score(source):
    result = runner.read_json(source / 'result.json')
    record = result['policies']['bench-shuffle']['best']
    result['selected_by_validation'] = dict(name='bench-shuffle', checkpoint=record['checkpoint'],
        checkpoint_sha256=record['checkpoint_sha256'], validation_rate=.5)
    write(source / 'result.json', result)
    with pytest.raises(ValueError, match='highest balanced validation'):
        runner.source_inputs(source)


def test_source_recipe_checks_build_and_fresh_species_pool(source):
    config = runner.read_json(ROOT / 'config/experiments/team-curriculum.json')
    data = runner.source_inputs(source)
    runner.validate_source_recipe(data, config)
    path = data['teams']['train'][0]
    team = runner.read_json(path); team['party'][0]['level'] = 100; write(path, team)
    with pytest.raises(ValueError, match='training build'): runner.validate_source_recipe(data, config)
    config['species_pools']['intro'][0] = 'Charizard'
    with pytest.raises(ValueError, match='original 12-species'):
        runner.fresh_teams(config, dict(compositions=[], semantics=[]), 4)


def test_continue_rejects_changed_recipe_or_catalog(prepared, monkeypatch):
    root, plan = prepared
    actual = runner.checksum
    monkeypatch.setattr(runner, 'checksum', lambda path: 'changed' if Path(path) == ROOT / 'config/generation/firered_us_v1_0.json' else actual(path))
    with pytest.raises(ValueError, match='recipe or catalog'): runner.verify(root, plan)
