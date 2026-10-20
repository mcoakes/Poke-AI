"""Safe matched practice continuation with unchanged selection and retained fallback."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_switch_practice as runner
from test_teacher_training import prepared as teacher_prepared, teacher_source, coverage_prepared, coverage_source, switch_prepared, source
from test_teacher_training import warmstart as source_warmstart, training as source_training
from test_switch_feature_ablation import simulator
from test_training_coverage import model
from test_transfer_lab import write


@pytest.fixture
def practice_source(teacher_prepared,monkeypatch):
    root,plan=teacher_prepared
    source_warmstart(root,plan,monkeypatch);source_training(root,plan,best_wins=2);simulator(monkeypatch)
    runner.teacher.finish(root,plan)
    return root


@pytest.fixture
def prepared(practice_source,tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'implementation_identity',lambda:{'practice-test':'fixed'})
    root=tmp_path/'practice'
    plan=runner.freeze(root,SimpleNamespace(source_run=practice_source,smoke=True,timesteps=1000000))
    return root,plan


def preparation(root,plan,monkeypatch):
    import firered_rl.decision_weighted_warmstart as core
    def fake(root,plan,**kwargs):
        for name in ('ppo-only','switch-practice'):
            model(root/f'inputs/{name}.zip',plan['settings']['starting_timesteps'],plan['settings']['migration_template_timesteps'],name)
        (root/'models/practice-only.zip').write_bytes((root/'inputs/switch-practice.zip').read_bytes())
        report=dict(version=1,status='completed',source_sha256=plan['source_start_sha256'],source_timesteps=plan['settings']['starting_timesteps'],
            source_parameters_sha256='source-parameters',no_evaluation_team_labels=True,no_game_outcome_labels=True,
            ppo_counters_unchanged=True,fit=dict(critic_parameters_unchanged=True,shared_encoder_trained=True,
                frozen_critic_parameters=['critic'],gradient_updates=1,weights={'move':1.,'forced':2.,'switch':4.},mirror_objective_mass=.1,
                split_counts={'fit':{'switch':1},'holdout':{'switch':1}}),
            optimizer_resets={name:dict(state_entries=0,learning_rate=.0001,num_timesteps=plan['settings']['starting_timesteps'],
                parameters_sha256='source-parameters' if name=='ppo-only' else 'practiced-parameters') for name in ('ppo-only','switch-practice')})
        write(root/'practice/report.json',report);return report
    monkeypatch.setattr(core,'prepare',fake)
    monkeypatch.setattr(core,'verify',lambda root,plan:runner.read_json(root/'practice/report.json'))
    runner.prepare(root,plan)


def training(root,plan,best_wins=1):
    source_training(root,plan,best_wins=best_wins)


def test_exact_selected_parent_preserved_and_source_unchanged(prepared,practice_source):
    root,plan=prepared;runner.verify(root,plan)
    source_result=runner.read_json(practice_source/'result.json');selected=source_result['selected_by_validation']
    assert (root/'inputs/start.zip').read_bytes()==(practice_source/selected['checkpoint']).read_bytes()
    assert (root/'models/previous-selected.zip').read_bytes()==(root/'inputs/start.zip').read_bytes()
    assert plan['settings']['starting_timesteps']==selected['timesteps']
    assert plan['settings']['seed']==261150
    initial=runner.read_json(root/'initial.json');assert initial['imported'] and set(initial['checks'])==set(runner.CONDITIONS)
    assert initial['balanced_validation']==selected['validation_rate']
    runner.bench.validate_saved_checks(root,[initial])
    assert 'reused development' in plan['development_role']


def test_unchanged_suites_teams_and_training_mixture(prepared,practice_source):
    root,plan=prepared;old=runner.read_json(practice_source/'plan.json')
    for name,relative in plan['suites'].items():assert (root/relative).read_bytes()==(practice_source/old['suites'][name]).read_bytes()
    for job in plan['jobs']:assert job['weights']=={'old':.25,'new':.65,'fixed-mirror':.1}
    assert plan['practice']==runner.practice_settings(True)
    normal=runner.practice_settings(False)
    assert normal['battles']==6000 and normal['max_rows']==50000 and normal['epochs']==8
    assert normal['weights']=={'move':1.,'forced':2.,'switch':4.} and sum(normal['capacities'].values())==50000


@pytest.mark.parametrize('change',['budget','practice','source','team','endpoint','baseline','extra'])
def test_frozen_configuration_and_baseline_integrity(prepared,change):
    root,plan=prepared
    if change=='budget':plan['settings']['timesteps']+=1
    elif change=='practice':plan['practice']['weights']['switch']=20.
    elif change=='source':(root/'inputs/start.zip').write_bytes(b'changed')
    elif change=='team':(root/plan['teams']['new'][0]).write_text('{}')
    elif change=='endpoint':(root/plan['suites']['fresh-original']).write_text('{}')
    elif change=='baseline':(root/'checks/imported-start-mirror/decision-traces.jsonl').unlink()
    else:write(root/'inputs/extra.json',{})
    with pytest.raises((ValueError,RuntimeError)):runner.verify(root,plan)


def test_preparation_and_matched_commands(prepared,monkeypatch):
    root,plan=prepared
    with pytest.raises(ValueError):runner.command(root,plan,plan['jobs'][0])
    preparation(root,plan,monkeypatch)
    for job in plan['jobs']:
        cmd=runner.command(root,plan,job)
        assert cmd[cmd.index('--resume')+1]==str(root/job['checkpoint'])
        assert cmd[cmd.index('--team-mixture')+1]==str(root/'inputs/mixture.json')
        assert cmd[cmd.index('--seed')+1]=='261150'
        assert cmd[cmd.index('--eval-suite')+1]==str(root/plan['suites']['validation'])
        assert '--no-eval-stop' in cmd and '--no-shuffle-bench-order' in cmd
    import firered_rl.decision_weighted_warmstart as core
    monkeypatch.setattr(core,'prepare',lambda *a,**k:pytest.fail('Preparation restarted'))
    runner.prepare(root,plan)
    (root/'models/practice-only.zip').write_bytes(b'changed')
    with pytest.raises(ValueError):runner.verify(root,plan)


def test_previous_selected_wins_ties_and_seven_exports(prepared,monkeypatch):
    root,plan=prepared;preparation(root,plan,monkeypatch);training(root,plan,best_wins=2);calls=simulator(monkeypatch)
    result=runner.finish(root,plan)
    assert result['selected_by_validation']['name']=='previous-selected'
    assert result['endpoint_used_for_selection'] is False and result['fallback_retains_ties']
    assert len(result['exports'])==7 and set(result['practice_only']['checks'])=={'validation',*runner.CONDITIONS}
    assert (root/'models/selected.zip').read_bytes()==(root/'inputs/start.zip').read_bytes()
    count=len(calls);logs=(root/'console/launcher.log').read_bytes()
    runner.finish(root,plan);assert len(calls)==count and (root/'console/launcher.log').read_bytes()==logs
    runner.collect(root,plan)
    with ZipFile(root/'switch-practice-results.zip') as archive:
        assert 'models/manifest.json' in archive.namelist()
        assert not any(name.endswith(('.zip','.npy','.npz')) or name.startswith('console/') for name in archive.namelist())
    report=(root/'report.md').read_text()
    assert 'reused development' in report and 'tail -n 20 -F' in report and '--inspect-teams all' in report


def test_native_best_selection_uses_same_validation_not_development(prepared,monkeypatch):
    root,plan=prepared;preparation(root,plan,monkeypatch);training(root,plan,best_wins=8);simulator(monkeypatch)
    result=runner.finish(root,plan)
    assert result['selected_by_validation']['name']=='ppo-only'
    assert result['selected_by_validation']['validation_rate']==1


def test_partial_results_retried_and_completed_trainers_skipped(prepared,monkeypatch):
    root,plan=prepared;preparation(root,plan,monkeypatch);training(root,plan);calls=simulator(monkeypatch)
    result=runner.finish(root,plan);count=len(calls)
    directory=root/result['policies']['switch-practice']['final']['checks']['fresh-original']['result_directory']
    (directory/'decision-traces.jsonl').unlink();runner.finish(root,plan);assert len(calls)==count+1 and directory.exists()
    monkeypatch.setattr(runner.teacher.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed trainer restarted'))
    runner.launch(root,plan,2)


@pytest.mark.parametrize('change',['selected','hash','native_suite','fallback_score'])
def test_source_selection_and_native_evidence_are_strict(practice_source,change):
    path=practice_source/'result.json';data=runner.read_json(path)
    if change=='selected':data['selected_by_validation']['name']='teacher-ppo'
    elif change=='hash':data['policies']['ppo-only']['best']['checkpoint_sha256']='bad'
    elif change=='fallback_score':data['fallbacks']['current-selected']['balanced_validation']=.999
    else:
        path=practice_source/'runs/ppo-only/checkpoints/best.json';data=runner.read_json(path);data['suite_sha256']='other'
    write(path,data)
    with pytest.raises(ValueError):runner.source_inputs(practice_source,smoke=True)


def test_native_continue_uses_derived_session_seed_and_prepared_hash(prepared,monkeypatch):
    root,plan=prepared;preparation(root,plan,monkeypatch);training(root,plan)
    job=plan['jobs'][1];run=root/'runs/switch-practice';config=runner.read_json(run/'config.json')
    write(run/'sessions/000001/config.json',config)
    config['session']['id']=2;config['saved_args']={'seed':job['seed']};config['seed']=(job['seed']+1000003)%(2**31)
    config['resume_checkpoint']['sha256']='later';write(run/'config.json',config)
    value=runner.training_provenance(root,plan,job,runner.read_json(run/'summary.json'))
    assert value['initial_checkpoint_sha256']==plan['pins']['inputs/switch-practice.zip'] and value['warmstart'] is True
    config['seed']+=1;write(run/'config.json',config)
    with pytest.raises(ValueError):runner.training_provenance(root,plan,job,runner.read_json(run/'summary.json'))


@pytest.mark.parametrize('change',['empty_switch','weights','counter'])
def test_production_practice_safeguards_are_not_relaxed(prepared,monkeypatch,change):
    root,plan=prepared;preparation(root,plan,monkeypatch)
    normal=dict(plan,smoke=False);report=copy.deepcopy(plan['preparation']['metadata'])
    if change=='empty_switch':report['fit']['split_counts']['holdout']['switch']=0
    elif change=='weights':report['fit']['weights']['switch']=20.
    else:report['ppo_counters_unchanged']=False
    with pytest.raises(ValueError):runner.validate_preparation_report(normal,report)
