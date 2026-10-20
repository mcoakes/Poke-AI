"""Matched coverage mixtures, fixed selection suites, source provenance and safe exports."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_training_coverage as runner
from test_switch_feature_ablation import prepared as switch_prepared, source, training as source_training, simulator
from test_transfer_lab import write
from firered_rl.evaluation import digest


@pytest.fixture
def coverage_source(switch_prepared, monkeypatch):
    root, plan = switch_prepared
    source_training(root, plan); simulator(monkeypatch)
    runner.switches.finish(root, plan)
    from sb3_contrib import MaskablePPO
    import firered_rl.shared_move_policy
    model = SimpleNamespace(policy=SimpleNamespace(features_extractor=SimpleNamespace(relational_enabled=True, bench_features_enabled=True)))
    monkeypatch.setattr(MaskablePPO, 'load', lambda *args, **kwargs: model)
    monkeypatch.setattr(firered_rl.shared_move_policy, 'policy_architecture', lambda value: 'matchup-switches')
    return root


@pytest.fixture
def prepared(coverage_source, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'implementation_identity', lambda: {'coverage-test': 'fixed'})
    root = tmp_path / 'coverage'
    plan = runner.freeze(root, SimpleNamespace(source_run=coverage_source, smoke=True, timesteps=500000))
    return root, plan


def model(path, steps, template_steps, arm):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(num_timesteps=steps, policy_architecture='matchup-switches', test_arm=arm,
        switch_ablation=dict(version=1, enabled=True, template_timesteps=template_steps,
            optimizer='preserved-existing/new-empty', seed=261120))
    with ZipFile(path, 'w') as archive: archive.writestr('data', json.dumps(data))
    return path


def training(root, plan, *, best_wins=1):
    suite = runner.read_json(root / plan['suites']['validation']); requested = len(suite['cases'])
    for job in plan['jobs']:
        run = root / 'runs' / job['name']; start = plan['settings']['migration_template_timesteps']
        for kind, steps in (('best', job['starting_timesteps']+128), ('final', job['target_timesteps'])):
            model(run / 'checkpoints' / (kind+'.zip'), steps, start, job['name'])
        write(run / 'checkpoints/best.json', dict(requested=requested, wins=best_wins,
            timesteps=job['starting_timesteps']+128, suite_sha256=digest(suite),
            checkpoint_sha256=runner.checksum(run / 'checkpoints/best.zip')))
        groups = {name: dict(weight=weight, fixed_pair=(name=='fixed-mirror'),
            files=[str(root / path) for path in plan['teams']['mirror' if name=='fixed-mirror' else name]])
            for name, weight in job['weights'].items()}
        write(run / 'config.json', dict(session=dict(id=1), shuffle_bench_order=False, policy_architecture='matchup-switches',
            resume_checkpoint=dict(sha256=plan['source_start_sha256']),
            switch_features=dict(enabled=True, whole_policy_training=True,
                migration=runner.switches.checkpoint_metadata(root / 'inputs/start.zip')),
            teams=dict(mode='team_mixture', manifest=dict(sha256=plan['pins'][job['mixture']]), groups=groups)))
        write(run / 'evaluation-history.json', dict(identity=dict(suite_sha256=digest(suite)), records=[dict(status='completed',
            checkpoint_sha256=runner.checksum(run / 'checkpoints/final.zip'), timesteps=job['target_timesteps'], requested=requested, wins=0)]))
        write(run / 'summary.json', dict(status='completed', stop_reason='timestep_budget', player_timesteps=job['target_timesteps'],
            elapsed_seconds=1, team_mixture_sampling=dict(weights=job['weights'],
                episode_selections=dict.fromkeys(job['weights'], 1), total=len(job['weights']))))


def test_exact_final_start_and_fallback_preserved(prepared, coverage_source):
    root, plan = prepared
    before = {str(path): runner.checksum(path) for path in coverage_source.rglob('*') if path.is_file()}
    runner.verify(root, plan)
    assert plan['settings']['starting_timesteps'] == 384
    assert plan['settings']['migration_template_timesteps'] == 128
    assert plan['settings']['seed'] == 261130
    assert all(job['checkpoint']=='inputs/start.zip' and job['bench_features_enabled'] and not job['shuffle_bench_order'] for job in plan['jobs'])
    assert (root / 'inputs/start.zip').read_bytes() == (coverage_source / 'runs/switch-features/checkpoints/final.zip').read_bytes()
    assert (root / 'models/previous-selected.zip').read_bytes() == (coverage_source / 'runs/control/checkpoints/best.zip').read_bytes()
    initial = runner.read_json(root / 'initial.json'); assert initial['imported'] and initial['timesteps']==384
    runner.bench.validate_saved_checks(root, [initial])
    assert before == {str(path): runner.checksum(path) for path in coverage_source.rglob('*') if path.is_file()}


def test_new_pool_balanced_disjoint_and_validation_bytes_unchanged(prepared, coverage_source):
    root, plan = prepared
    ledger = runner.read_json(root / 'inputs/new-team-inventory.json')
    assert ledger['teams']==12 and set(ledger['species_counts'].values())=={3}
    assert all(set(counts.values())=={1} for counts in ledger['slot_counts'].values())
    assert not set(ledger['compositions']) & set(ledger['exclusions']['compositions'])
    assert not set(ledger['semantics']) & set(ledger['exclusions']['semantics'])
    source_plan = runner.read_json(coverage_source / 'plan.json')
    for name, relative in plan['suites'].items():
        assert (root / relative).read_bytes() == (coverage_source / source_plan['suites'][name]).read_bytes()


def test_per_job_mixtures_and_command_substitution(prepared):
    root, plan = prepared
    assert [job['weights'] for job in plan['jobs']] == [{'old':.9,'fixed-mirror':.1}, {'old':.25,'new':.65,'fixed-mirror':.1}]
    for job in plan['jobs']:
        mixture = runner.read_json(root / job['mixture'])
        assert {name:group['weight'] for name,group in mixture['groups'].items()}==job['weights']
        command = runner.command(root, plan, job)
        assert command[command.index('--team-mixture')+1]==str(root / job['mixture'])
        assert command[command.index('--resume')+1]==str(root / 'inputs/start.zip')
        assert command[command.index('--policy-architecture')+1]=='matchup-switches'
        assert '--no-eval-stop' in command and '--no-shuffle-bench-order' in command


@pytest.mark.parametrize('change', ['source_model','fallback','weights','budget','membership','catalog'])
def test_saved_configuration_is_immutable(prepared, monkeypatch, change):
    root, plan = prepared
    if change=='source_model': (root/'inputs/start.zip').write_bytes(b'changed')
    elif change=='fallback': (root/'inputs/previous-selected.zip').write_bytes(b'changed')
    elif change=='weights': plan['jobs'][1]['weights']['new']=.5
    elif change=='budget': plan['settings']['timesteps']+=1
    elif change=='membership': write(root/'inputs/teams/new/extra.json',{})
    else:
        actual=runner.checksum
        monkeypatch.setattr(runner,'checksum',lambda path:'bad' if Path(path)==ROOT/'config/generation/firered_us_v1_0.json' else actual(path))
    with pytest.raises(ValueError): runner.verify(root,plan)


@pytest.mark.parametrize('change',['final_hash','final_steps','feature_flag','best_hash','selected','source_pin'])
def test_source_provenance_is_strict(coverage_source, change, monkeypatch):
    result=runner.read_json(coverage_source/'result.json')
    if change=='final_hash': result['policies']['switch-features']['final']['checkpoint_sha256']='bad'
    elif change=='final_steps': result['policies']['switch-features']['final']['timesteps']+=1
    elif change=='feature_flag':
        from sb3_contrib import MaskablePPO
        monkeypatch.setattr(MaskablePPO,'load',lambda *a,**k:SimpleNamespace(policy=SimpleNamespace(features_extractor=SimpleNamespace(relational_enabled=True,bench_features_enabled=False))))
    elif change=='best_hash': result['policies']['control']['best']['checkpoint_sha256']='bad'
    elif change=='selected': result['selected_by_validation']['name']='switch-features'
    elif change=='source_pin': (coverage_source/'inputs/reward.json').write_text('{}')
    write(coverage_source/'result.json',result)
    with pytest.raises(ValueError): runner.source_inputs(coverage_source,smoke=True)


@pytest.mark.parametrize('change',['manifest','group_weight','members','fixed_pair','feature_flag','counts'])
def test_actual_training_mixture_and_features_verified(prepared,change):
    root,plan=prepared;training(root,plan);job=plan['jobs'][1]
    path=root/'runs/balanced-pool/config.json';config=runner.read_json(path)
    summary=runner.read_json(root/'runs/balanced-pool/summary.json')
    if change=='manifest':config['teams']['manifest']['sha256']='wrong'
    elif change=='group_weight':config['teams']['groups']['new']['weight']=.5
    elif change=='members':config['teams']['groups']['new']['files'].pop()
    elif change=='fixed_pair':config['teams']['groups']['fixed-mirror']['fixed_pair']=False
    elif change=='feature_flag':config['switch_features']['enabled']=False
    else:summary['team_mixture_sampling']['total']+=1
    write(path,config)
    with pytest.raises(ValueError):runner.training_provenance(root,plan,job,summary)


def test_equal_or_worse_new_validation_keeps_fallback_selected(prepared,monkeypatch):
    root,plan=prepared;training(root,plan);calls=simulator(monkeypatch)
    result=runner.finish(root,plan)
    assert result['selected_by_validation']['name']=='previous-selected'
    assert result['new_run_selected']['name']=='small-pool'
    assert not result['new_selection_beats_previous']
    assert (root/'models/selected.zip').read_bytes()==(root/'inputs/previous-selected.zip').read_bytes()
    assert (root/'models/new-run-selected.zip').read_bytes()==(root/'runs/small-pool/checkpoints/best.zip').read_bytes()
    assert len(calls)==20 and len(result['exports'])==7
    runner.finish(root,plan);assert len(calls)==20
    runner.collect(root,plan)
    with ZipFile(root/'training-coverage-results.zip') as archive:
        assert 'models/manifest.json' in archive.namelist()
        assert not any(name.endswith('.zip') for name in archive.namelist())
    text=(root/'report.md').read_text()
    assert 'previous-selected.zip' in text and 'balanced-pool-final.zip' in text and '--inspect-teams new' in text


def test_strict_validation_improvement_exports_new_global_winner(prepared,monkeypatch):
    root,plan=prepared;training(root,plan,best_wins=2);simulator(monkeypatch)
    result=runner.finish(root,plan)
    assert result['selected_by_validation']['name']=='small-pool' and result['new_selection_beats_previous']
    assert (root/'models/selected.zip').read_bytes()==(root/'runs/small-pool/checkpoints/best.zip').read_bytes()


def test_partial_checks_retry_without_losing_source_and_resume_skips_complete(prepared,monkeypatch):
    root,plan=prepared;training(root,plan);calls=simulator(monkeypatch);runner.finish(root,plan)
    directory=root/'checks/balanced-pool-final-heldout-original';(directory/'decision-traces.jsonl').unlink()
    runner.finish(root,plan);assert len(calls)==21 and directory.exists()
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed training restarted'))
    runner.launch(root,plan,2)


def test_team_inspection_prints_both_mixtures_and_new_species(prepared,capsys):
    root,plan=prepared;runner.inspect(root,plan,'all',1,False)
    text=capsys.readouterr().out
    assert 'small-pool' in text and 'balanced-pool' in text and 'new: 12 teams' in text
    assert 'old: 2 teams' in text and 'mirror: 2 teams' in text


@pytest.mark.parametrize('change', ['suite', 'wins', 'nan'])
def test_previous_selection_requires_same_validation_identity_and_valid_counts(coverage_source, change):
    path=coverage_source/'runs/control/checkpoints/best.json'; meta=runner.read_json(path)
    if change=='suite': meta['suite_sha256']='different-suite'
    elif change=='wins': meta['wins']=meta['requested']+1
    else:
        result=runner.read_json(coverage_source/'result.json')
        result['selected_by_validation']['validation_rate']=float('nan')
        write(coverage_source/'result.json',result)
    write(path,meta)
    with pytest.raises(ValueError): runner.source_inputs(coverage_source,smoke=True)
