"""Frozen broader selector, teacher initialization, safe continuation and exports."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_teacher_training as runner
from test_training_coverage import prepared as coverage_prepared, coverage_source, switch_prepared, source, training as coverage_training, model
from test_switch_feature_ablation import simulator
from test_transfer_lab import write
from firered_rl.evaluation import digest


@pytest.fixture
def teacher_source(coverage_prepared,monkeypatch):
    root,plan=coverage_prepared
    coverage_training(root,plan,best_wins=2);simulator(monkeypatch)
    runner.coverage.finish(root,plan)
    return root


@pytest.fixture
def prepared(teacher_source,tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'implementation_identity',lambda:{'teacher-test':'fixed'})
    root=tmp_path/'teacher'
    plan=runner.freeze(root,SimpleNamespace(source_run=teacher_source,smoke=True,timesteps=500000))
    return root,plan


def warmstart(root,plan,monkeypatch):
    import firered_rl.teacher_warmstart
    def fake(root,plan,**kwargs):
        for name in ('ppo-only','teacher-ppo'):
            model(root/f'inputs/{name}.zip',plan['settings']['starting_timesteps'],plan['settings']['migration_template_timesteps'],name)
        (root/'models/teacher-only.zip').write_bytes((root/'inputs/teacher-ppo.zip').read_bytes())
        report=dict(version=1,status='completed',source_sha256=plan['source_start_sha256'],
            source_timesteps=plan['settings']['starting_timesteps'],source_parameters_sha256='source-parameters',
            no_evaluation_team_labels=True,no_game_outcome_labels=True,
            fit=dict(critic_parameters_unchanged=True,shared_encoder_trained=True,frozen_critic_parameters=['critic'],gradient_updates=1),
            optimizer_resets={name:dict(state_entries=0,learning_rate=plan['settings']['learning_rate'],
                num_timesteps=plan['settings']['starting_timesteps'],parameters_sha256='source-parameters' if name=='ppo-only' else 'learned-parameters')
                for name in ('ppo-only','teacher-ppo')})
        write(root/'warmstart/report.json',report)
        return report
    monkeypatch.setattr(firered_rl.teacher_warmstart,'prepare',fake)
    monkeypatch.setattr(firered_rl.teacher_warmstart,'verify',lambda root,plan:runner.read_json(root/'warmstart/report.json'))
    runner.prepare(root,plan)


def training(root,plan,best_wins=1):
    suite=runner.read_json(root/plan['suites']['validation']);requested=len(suite['cases'])
    for job in plan['jobs']:
        run=root/'runs'/job['name']
        for kind,steps in [('best',job['starting_timesteps']+128),('final',job['target_timesteps'])]:
            model(run/'checkpoints'/f'{kind}.zip',steps,plan['settings']['migration_template_timesteps'],job['name'])
        write(run/'checkpoints/best.json',dict(requested=requested,wins=best_wins,suite_sha256=digest(suite),
            timesteps=job['starting_timesteps']+128,checkpoint_sha256=runner.checksum(run/'checkpoints/best.zip')))
        groups={name:dict(weight=weight,fixed_pair=name=='fixed-mirror',
            files=[str(root/path) for path in plan['teams']['mirror' if name=='fixed-mirror' else name]]) for name,weight in job['weights'].items()}
        write(run/'config.json',dict(session=dict(id=1),shuffle_bench_order=False,policy_architecture='matchup-switches',
            seed=job['seed'],environments=plan['settings']['environments'],effective_ppo=dict(learning_rate=.0001,ent_coef=.01,n_steps=128,batch_size=128,n_epochs=4),
            resume_checkpoint=dict(sha256=plan['pins'][job['checkpoint']]),
            switch_features=dict(enabled=True,whole_policy_training=True,migration=runner.switches.checkpoint_metadata(root/'inputs/start.zip')),
            teams=dict(mode='team_mixture',manifest=dict(sha256=plan['pins'][job['mixture']]),groups=groups)))
        write(run/'summary.json',dict(status='completed',stop_reason='timestep_budget',player_timesteps=job['target_timesteps'],elapsed_seconds=1,
            team_mixture_sampling=dict(weights=job['weights'],episode_selections=dict.fromkeys(job['weights'],1),total=len(job['weights']))))
        write(run/'evaluation-history.json',dict(identity=dict(suite_sha256=digest(suite)),records=[dict(status='completed',
            checkpoint_sha256=runner.checksum(run/'checkpoints/final.zip'),timesteps=job['target_timesteps'],requested=requested,wins=0)]))


def test_parent_preserved_and_smoke_override_explicit(prepared,teacher_source):
    root,plan=prepared;runner.verify(root,plan)
    override=plan['source']['smoke_source_override']
    assert override and override['checkpoint']=='runs/balanced-pool/checkpoints/best.zip'
    assert plan['fallbacks']['current-selected']['source_checkpoint']==override['checkpoint']
    assert (root/'inputs/start.zip').read_bytes()==(teacher_source/override['checkpoint']).read_bytes()
    assert all(job['weights']=={'old':.25,'new':.65,'fixed-mirror':.1} for job in plan['jobs'])
    assert plan['settings']['seed']==261140 and plan['settings']['early_stopping'] is False
    assert 'heldout-original' not in plan['suites']
    assert (root/'inputs/suites/historical-heldout-original.json').exists()


def test_fresh_selector_and_endpoint_are_disjoint_balanced_and_pinned(prepared):
    root,plan=prepared
    from firered_rl.broad_team_curriculum import composition_signature
    pools={group:{composition_signature(runner.read_json(root/path)) for path in plan['teams'][group]}
        for group in ('old','new','mirror','validation','fresh-test')}
    assert not pools['validation'] & pools['fresh-test']
    assert not (pools['validation']|pools['fresh-test']) & (pools['old']|pools['new']|pools['mirror'])
    assert len(runner.read_json(root/plan['suites']['validation'])['cases'])==8
    for relative in plan['teacher_suites']['pins']: assert relative in plan['pins']


@pytest.mark.parametrize('change',['budget','mixture','source','new_team','endpoint','extra'])
def test_immutable_plan_and_inputs(prepared,change):
    root,plan=prepared
    if change=='budget':plan['settings']['timesteps']+=1
    elif change=='mixture':plan['jobs'][0]['weights']['new']=.5
    elif change=='source':(root/'inputs/start.zip').write_bytes(b'changed')
    elif change=='new_team':(root/plan['teams']['new'][0]).write_text('{}')
    elif change=='endpoint':(root/plan['suites']['fresh-original']).write_text('{}')
    else:write(root/'inputs/extra.json',{})
    with pytest.raises(ValueError):runner.verify(root,plan)


def test_preparation_pins_outputs_and_matches_budgets(prepared,monkeypatch):
    root,plan=prepared
    with pytest.raises(ValueError):runner.command(root,plan,plan['jobs'][0])
    warmstart(root,plan,monkeypatch);runner.verify(root,plan)
    for job in plan['jobs']:
        cmd=runner.command(root,plan,job)
        assert cmd[cmd.index('--resume')+1]==str(root/job['checkpoint'])
        assert cmd[cmd.index('--team-mixture')+1]==str(root/'inputs/mixture.json')
        assert cmd[cmd.index('--eval-suite')+1]==str(root/plan['suites']['validation'])
        assert '--no-eval-stop' in cmd and '--no-shuffle-bench-order' in cmd
    import firered_rl.teacher_warmstart
    monkeypatch.setattr(firered_rl.teacher_warmstart,'prepare',lambda *a,**k:pytest.fail('Preparation restarted'))
    runner.prepare(root,plan)
    (root/'models/teacher-only.zip').write_bytes(b'changed')
    with pytest.raises(ValueError):runner.verify(root,plan)


def test_fallbacks_all_use_new_selector_and_fallback_wins_ties(prepared,monkeypatch):
    root,plan=prepared;warmstart(root,plan,monkeypatch);training(root,plan);calls=simulator(monkeypatch)
    result=runner.finish(root,plan)
    assert result['selected_by_validation']['name']=='current-selected'
    assert result['endpoint_used_for_selection'] is False and result['fallback_retains_ties']
    for record in result['fallbacks'].values():assert set(record['checks'])=={'validation',*runner.CONDITIONS}
    assert set(result['teacher_only']['checks'])=={'validation',*runner.CONDITIONS}
    expected={'selected.zip','teacher-only.zip','ppo-only-best.zip','ppo-only-final.zip',
        'teacher-ppo-best.zip','teacher-ppo-final.zip','fallback-current-selected.zip','fallback-small-best.zip','fallback-feature-final.zip'}
    assert expected<={path.name for path in (root/'models').glob('*.zip')}
    count=len(calls);log=(root/'console/launcher.log').read_bytes()
    runner.finish(root,plan);assert len(calls)==count
    assert (root/'console/launcher.log').read_bytes()==log
    runner.collect(root,plan)
    with ZipFile(root/'teacher-training-results.zip') as archive:
        assert 'models/manifest.json' in archive.namelist()
        assert not any(name.endswith(('.zip','.npz')) or name.startswith('console/') for name in archive.namelist())
    report=(root/'report.md').read_text()
    assert 'tail -n 20 -F' in report and '--inspect-teams all' in report and 'watch -n 2 free -h' in report


def test_native_best_wins_by_validation_not_endpoint(prepared,monkeypatch):
    root,plan=prepared;warmstart(root,plan,monkeypatch);training(root,plan,best_wins=8);simulator(monkeypatch)
    result=runner.finish(root,plan)
    assert result['selected_by_validation']['name']=='ppo-only'
    assert result['selected_by_validation']['validation_rate']==1


def test_partial_evidence_retried_completed_trainers_skipped(prepared,monkeypatch):
    root,plan=prepared;warmstart(root,plan,monkeypatch);training(root,plan);calls=simulator(monkeypatch)
    result=runner.finish(root,plan);count=len(calls)
    path=root/result['policies']['teacher-ppo']['final']['checks']['fresh-original']['result_directory']
    (path/'decision-traces.jsonl').unlink();runner.finish(root,plan)
    assert len(calls)==count+1 and path.exists()
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:pytest.fail('Completed training restarted'))
    runner.launch(root,plan,2)


@pytest.mark.parametrize('change',['source_selection','source_hash','source_best_suite'])
def test_source_validation_and_membership_are_strict(teacher_source,change):
    if change=='source_best_suite':
        path=teacher_source/'runs/small-pool/checkpoints/best.json';data=runner.read_json(path);data['suite_sha256']='other'
    else:
        path=teacher_source/'result.json';data=runner.read_json(path)
        if change=='source_selection':data['selected_by_validation']['name']='balanced-pool'
        else:data['policies']['small-pool']['best']['checkpoint_sha256']='bad'
    write(path,data)
    with pytest.raises(ValueError):runner.source_inputs(teacher_source,smoke=True)


def test_resume_provenance_uses_prepared_arm_not_ancient_migration(prepared,monkeypatch):
    root,plan=prepared;warmstart(root,plan,monkeypatch);training(root,plan)
    job=plan['jobs'][1];summary=runner.read_json(root/'runs/teacher-ppo/summary.json')
    actual=runner.training_provenance(root,plan,job,summary)
    assert actual['initial_checkpoint_sha256']==plan['pins']['inputs/teacher-ppo.zip']
    assert actual['common_parent_sha256']==plan['source_start_sha256']
    path=root/'runs/teacher-ppo/config.json';config=runner.read_json(path)
    config['resume_checkpoint']['sha256']=plan['pins']['inputs/ppo-only.zip'];write(path,config)
    with pytest.raises(ValueError):runner.training_provenance(root,plan,job,summary)


def test_pretraining_opens_only_new_selector(prepared,monkeypatch):
    root,plan=prepared;calls=simulator(monkeypatch)
    result=runner.evaluate_fallbacks(root,plan,complete=False)
    assert all(set(record['checks'])=={'validation'} for record in result['records'].values())
    assert all('validation' in call for call in calls)
    assert not any((root/'checks').glob('*fresh*'))


@pytest.mark.parametrize('change',['critic','optimizer','source','timesteps'])
def test_warmstart_contract_rejects_invalid_training_evidence(prepared,monkeypatch,change):
    root,plan=prepared;warmstart(root,plan,monkeypatch)
    report=copy.deepcopy(plan['preparation']['metadata'])
    if change=='critic':report['fit']['critic_parameters_unchanged']=False
    elif change=='optimizer':report['optimizer_resets']['teacher-ppo']['state_entries']=1
    elif change=='source':report['source_sha256']='other'
    else:report['source_timesteps']+=1
    with pytest.raises(ValueError):runner.validate_preparation_report(plan,report)


def test_actual_optimizer_settings_must_match_both_arm_plan(prepared,monkeypatch):
    root,plan=prepared;warmstart(root,plan,monkeypatch);training(root,plan)
    job=plan['jobs'][1];path=root/'runs/teacher-ppo/config.json';config=runner.read_json(path)
    config['effective_ppo']['ent_coef']=.2;write(path,config)
    with pytest.raises(ValueError):runner.training_provenance(root,plan,job,runner.read_json(root/'runs/teacher-ppo/summary.json'))


def test_native_continuation_accepts_derived_session_seed(prepared,monkeypatch):
    root,plan=prepared;warmstart(root,plan,monkeypatch);training(root,plan)
    job=plan['jobs'][1];run=root/'runs/teacher-ppo';config=runner.read_json(run/'config.json')
    write(run/'sessions/000001/config.json',config)
    config['session']['id']=2;config['saved_args']={'seed':job['seed']}
    config['seed']=(job['seed']+1000003)%(2**31)
    config['resume_checkpoint']['sha256']='later-resume-checkpoint'
    write(run/'config.json',config)
    summary=runner.read_json(run/'summary.json')
    runner.training_provenance(root,plan,job,summary)
    config['seed']+=1;write(run/'config.json',config)
    with pytest.raises(ValueError):runner.training_provenance(root,plan,job,summary)
