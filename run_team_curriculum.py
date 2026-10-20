#!/usr/bin/env python3
"""Extend a qualified fixed-team specialist through configurable stages 4–10."""
import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, recovered_steps, pump
from run_staged_team_lab import memory_monitor

FOUNDATION_IDS = ['01-move-choice', '02-fixed-mirror', '03-replication-diagnostic']


def read_config(path, smoke=False):
    c = json.loads(Path(path).read_text())
    if c.get('version') != 1 or len(c.get('stages', [])) != 7:
        raise ValueError('Expected a version-1 curriculum with seven stages numbered 04–10.')
    if [s['id'][:2] for s in c['stages']] != [f'{i:02}' for i in range(4,11)]:
        raise ValueError('Stage IDs must begin 04 through 10 in order.')
    t = c['training']
    for key in ('environments','eval_every','validation_battles','test_battles','calibration_battles','early_stop_consecutive'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'Invalid training.{key}')
    if t['environments'] % 2 or any(t[k] % 2 for k in ('validation_battles','test_battles','calibration_battles')):
        raise ValueError('Environment and battle counts must be even.')
    if type(c['seed']) is not int or not 0 <= c['seed'] < 2**31-100:
        raise ValueError('Invalid seed')
    if len(t['learning_rates']) != 2 or any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in t['learning_rates']):
        raise ValueError('Provide two positive finite learning rates.')
    for x in [t['ent_coef'],t['retention_max_drop'],t['max_truncation_rate'],*t['rehearsal_weights'].values()]:
        if type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('Weights/thresholds must be finite fractions.')
    if set(t['rehearsal_weights']) != {'current','previous','mirror'} or t['rehearsal_weights']['current'] <= 0:
        raise ValueError('Rehearsal requires current/previous/mirror weights and positive current weight.')
    for stage in c['stages']:
        for key in ('timesteps','max_steps'):
            if type(stage['training'][key]) is not int or stage['training'][key] <= 0:
                raise ValueError(f'Invalid {stage["id"]}.{key}')
        for key in ('validation_win_rate','heldout_win_rate'):
            x = stage['training'][key]
            if type(x) not in (int,float) or not math.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f'Invalid {stage["id"]}.{key}')
    if c['stages'][-1]['counts'].get('benchmark', 0) < 2:
        raise ValueError('Stage 10 requires at least two reserved benchmark teams.')
    if smoke:
        t.update(environments=2,eval_every=128,validation_battles=8,test_battles=8,calibration_battles=8)
        for stage in c['stages']:
            stage['counts'] = {k: (2 if n else 0) for k,n in stage['counts'].items()}
            stage['training']['timesteps'] = 256
    return c


def source_evidence(source, smoke=False):
    """Import a passed policy-specific gate; retain failed replication as evidence."""
    source = Path(source).expanduser().resolve()
    plan = json.loads((source/'plan.json').read_text())
    if plan.get('smoke') and not smoke:
        raise ValueError('A smoke run cannot supply a qualified foundation.')
    result = {'source':str(source),'models':{},'evidence':{}}
    for old,new in [('move-choice','move-choice'),('mirror-stability','mirror')]:
        r = json.loads((source/'stages'/old/'result.json').read_text())
        if r.get('status') != 'passed' and not (smoke and r.get('status') == 'smoke_completed'):
            raise ValueError(f'Source {old} must have passed its own gate.')
        checkpoint = source / r['selected']
        if checksum(checkpoint) != r['checkpoint_sha256']:
            raise ValueError(f'Source selected checkpoint changed: {old}')
        result['models'][new] = checkpoint
        result['evidence'][old] = r
    replication = source/'stages/mirror-replication/result.json'
    result['evidence']['mirror-replication'] = json.loads(replication.read_text()) if replication.exists() else {'status':'unavailable'}
    result['reward'] = source/'inputs/reward.json'
    return result


def freeze(root, args):
    from firered_rl.broad_team_curriculum import build_pools, write_fixed_assets
    from firered_rl.curriculum_checks import coverage_suite
    from firered_rl.fixed_team_lab import make_suite
    from firered_rl.showdown_env import validate_showdown
    from sb3_contrib import MaskablePPO
    from firered_rl.shared_move_policy import SharedMovePolicy
    config = read_config(args.config, args.smoke)
    evidence = source_evidence(args.source_run,args.smoke)
    validate_showdown(ROOT)
    for path in evidence['models'].values():
        model = MaskablePPO.load(path,device='cpu')
        if not isinstance(model.policy,SharedMovePolicy):
            raise ValueError('Foundation checkpoint must use shared-moves architecture.')
        del model
    reference = args.reference_checkpoint if getattr(args,'reference_checkpoint',None) else ROOT/'models/observation-v2-shared-moves-400k-reference-01.zip'
    if getattr(args,'reference_checkpoint',None) and not reference.is_file():
        raise ValueError(f'Reference checkpoint not found: {reference}')
    if reference.is_file():
        model = MaskablePPO.load(reference,device='cpu');del model
    root.mkdir(parents=True,exist_ok=False)
    inputs = root/'inputs';inputs.mkdir()
    (root/'console').mkdir()
    write_json(inputs/'curriculum.json',config)
    shutil.copyfile(evidence['reward'],inputs/'reward.json')
    imports = {}
    for label,path in evidence['models'].items():
        target = inputs/f'foundation-{label}.zip'
        shutil.copyfile(path,target)
        imports[label] = str(target.relative_to(root))
    if reference.is_file():
        shutil.copyfile(reference,inputs/'working-reference.zip')
        imports['working-reference']='inputs/working-reference.zip'
    write_json(inputs/'source-evidence.json',{'source_run':evidence['source'], 'stages':evidence['evidence'],
        'policy':'Qualified stability model imported. Fresh-seed replication remains diagnostic; previous failed result is preserved.'})
    assets = build_pools(ROOT,inputs,config)
    fixed = write_fixed_assets(ROOT,inputs,config)
    suites = {}
    for name,scenario in [(FOUNDATION_IDS[0],'move-choice'),(FOUNDATION_IDS[1],'mirror')]:
        entry = fixed[scenario]
        paths = [Path(entry['root'])/p for p in entry['train_paths']]
        suites[name] = {}
        for label,permute in [('test',False),('permuted',True)]:
            suite = make_suite(ROOT,paths,config['training']['test_battles'],912001+(scenario=='mirror'),scenario,permute_moves=permute)
            dest = inputs/'suites'/f'{name}-{label}.json';dest.parent.mkdir(exist_ok=True)
            write_json(dest,suite);suites[name][label]=str(dest.relative_to(root))
    for index,stage in enumerate(config['stages']):
        stage_id = stage['id'];entry=assets[stage_id];suites[stage_id]={}
        for label,split,offset,requested,permuted in [
            ('validation','validation',1,config['training']['validation_battles'],False),
            ('calibration','heldout',2,config['training']['calibration_battles'],False),
            ('test','train',3,config['training']['test_battles'],False),
            ('heldout','heldout',4,config['training']['test_battles'],False),
            ('permuted','heldout',4,config['training']['test_battles'],True),
            *([('benchmark','benchmark',5,max(2000,config['training']['test_battles']) if not args.smoke else 8,False)] if entry.get('benchmark_paths') else [])]:
            paths=[Path(entry['root'])/p for p in entry[split+'_paths']]
            suite=coverage_suite(ROOT,paths,requested,913000+index*10+offset,
                                 max_steps=stage['training']['max_steps'],permute_moves=permuted)
            dest=inputs/'suites'/f'{stage_id}-{label}.json'
            write_json(dest,suite);suites[stage_id][label]=str(dest.relative_to(root))
        # Rehearsal draws both sides from the chosen stage to avoid accidental 3v6 matchups.
        weights=config['training']['rehearsal_weights']
        groups={'current':{'weight':weights['current']+(weights['previous'] if index==0 else 0),
                           'directory':str(Path(entry['root'])/entry['train_dir'])}}
        if weights['previous'] and index:
            for previous in config['stages'][:index]:
                prior=assets[previous['id']]
                groups[previous['id']]={'weight':weights['previous']/index,
                    'directory':str(Path(prior['root'])/prior['train_dir'])}
        if weights['mirror']:
            groups['fixed-mirror']={'weight':weights['mirror'],'teams':[str(Path(fixed['mirror']['root'])/p) for p in fixed['mirror']['train_paths']]}
        write_json(inputs/f'{stage_id}-mixture.json',dict(version=1,groups=groups))
        for arm in ('a','b'):
            (root/'console'/f'{stage_id}-{arm}.log').touch()
    plan=dict(version=1,smoke=args.smoke,config=config,assets=assets,fixed=fixed,imports=imports,suites=suites,
        pins={str(p.relative_to(root)):checksum(p) for p in inputs.rglob('*') if p.is_file()})
    write_json(root/'plan.json',plan)
    return plan


def verify(root,plan):
    from firered_rl.broad_team_curriculum import validate_pools
    from firered_rl.staged_team_lab import validate_assets
    if plan.get('version') != 1:
        raise ValueError('Unsupported curriculum plan')
    for name,digest in plan['pins'].items():
        if checksum(root/name)!=digest:
            raise ValueError(f'Pinned input changed: {name}')
    validate_pools(plan['assets'])
    for entry in plan['fixed'].values():validate_assets(entry)


def jobs_for(root,plan,stage,anchor):
    path=root/'stages'/stage['id']/'jobs.json'
    if path.exists():
        value=json.loads(path.read_text())
        for name,digest in value['pins'].items():
            if checksum(root/name)!=digest:raise ValueError(f'Stage start changed: {name}')
        return value['jobs']
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'inputs/stage-starts'/f"{stage['id']}.zip";checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists():
        if checksum(checkpoint)!=checksum(root/anchor):raise ValueError('Existing stage start differs')
    else:shutil.copyfile(root/anchor,checkpoint)
    start=checkpoint_steps(checkpoint)
    jobs=[dict(name=f"{stage['id']}-{arm}",stage=stage['id'],seed=plan['config']['seed']+int(stage['id'][:2]),
               checkpoint=str(checkpoint.relative_to(root)),learning_rate=rate,starting_timesteps=start,
               target_timesteps=start+stage['training']['timesteps'])
          for arm,rate in zip(('a','b'),plan['config']['training']['learning_rates'])]
    write_json(path,dict(jobs=jobs,pins={str(checkpoint.relative_to(root)):checksum(checkpoint)}))
    return jobs


def command(root,plan,job):
    base=[sys.executable,'-u',str(ROOT/'mgba/train_ppo.py')]
    run=root/'runs'/job['name']
    if run.exists():
        path=run/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
        if summary.get('status')=='completed' and (summary.get('stop_reason')=='validation_success' or recovered_steps(run)>=job['target_timesteps']):return None
        return base+['--output-dir',str(run),'--if-exists','continue','--total-timesteps',str(max(1,job['target_timesteps']-recovered_steps(run)))]
    t=plan['config']['training'];stage=next(s for s in plan['config']['stages'] if s['id']==job['stage'])
    return base+['--backend','showdown','--resume',str(root/job['checkpoint']),
        '--policy-architecture','shared-moves','--observation-version','v2','--rule-based-opponent',
        '--team-mixture',str(root/'inputs'/f"{stage['id']}-mixture.json"),
        '--reward-config',str(root/'inputs/reward.json'),'--learning-rate',str(job['learning_rate']),
        '--ent-coef',str(t['ent_coef']),'--seed',str(job['seed']),'--environments',str(t['environments']),
        '--rollout-steps','128','--batch-size','128','--epochs','4','--torch-threads','1','--device','cpu',
        '--showdown-workers','process','--max-steps',str(stage['training']['max_steps']),
        '--total-timesteps',str(stage['training']['timesteps']),
        '--eval-suite',str(root/plan['suites'][stage['id']]['validation']),
        '--eval-every',str(t['eval_every']),'--eval-environments','4',
        '--eval-stop-win-rate',str(stage['training']['validation_win_rate']),
        '--eval-stop-consecutive',str(t['early_stop_consecutive']),
        '--latest-every-updates','1','--checkpoint-every',str(t['eval_every']),'--keep-checkpoints','3','--keep-evaluations','2',
        '--battle-log','--no-readable-log','--no-step-logs','--log-max-mb','2','--log-backups','1',
        '--output-dir',str(run),'--if-exists','cancel']


def launch(root,plan,jobs,concurrency):
    pending,active=list(jobs),[]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        while pending or active:
            while pending and len(active)<concurrency:
                job=pending.pop(0);cmd=command(root,plan,job)
                if cmd is None:continue
                print(f"Starting {job['name']}\n{shlex.join(cmd)}",flush=True)
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                reader=threading.Thread(target=pump,args=(proc,root/'console'/f"{job['name']}.log"));reader.start()
                active.append((job,proc,reader))
            for row in list(active):
                job,proc,reader=row;code=proc.poll()
                if code is None:continue
                reader.join();active.remove(row)
                if code:raise RuntimeError(f"{job['name']} exited {code}; inspect console log")
                if json.loads((root/'runs'/job['name']/'summary.json').read_text()).get('status')!='completed':
                    raise RuntimeError(f"{job['name']} did not complete")
            if active:time.sleep(.25)
    finally:
        for _,proc,_ in active:
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
        if active:print('Waiting for active trainers to save and clean up.',flush=True)
        for _,proc,reader in active:proc.wait();reader.join()


def rate(result):return result['candidate_wins']/result['requested_battles']


def complete(result,max_truncation=0):
    return (result.get('status')=='completed' and result['attempted_battles']==result['requested_battles']
        and result['completed_battles']+result['truncated']==result['requested_battles']
        and result['truncated']/result['requested_battles']<=max_truncation)


def broad_gate(stage,checks,retention,config):
    issues=[];limit=config['training']['max_truncation_rate'];floor=stage['training']['heldout_win_rate']
    for name in ('heldout','permuted'):
        if not complete(checks[name],limit) or rate(checks[name])<floor:issues.append(f'{name}: below {floor:.0%} wins/requested or incomplete/excess truncation')
    if rate(checks['heldout'])-rate(checks['permuted'])>.03+1e-12:issues.append('move rotation loses more than 3 percentage points')
    for name,values in retention.items():
        if not complete(values['after'],limit) or not complete(values['before'],limit):issues.append(f'{name}: incomplete or excessive truncation in retention check')
        elif rate(values['after'])<rate(values['before'])-config['training']['retention_max_drop']-1e-12:
            issues.append(f'{name}: retention drop exceeds configured limit')
    return issues


def save_result(root,stage,result,plan):
    path=root/'stages'/stage/'result.json';path.parent.mkdir(parents=True,exist_ok=True)
    if plan['smoke']:result['status']='smoke_completed'
    write_json(path,result)
    print(f"Stage {stage}: {result['status']}; {'; '.join(result.get('issues',[])) or 'checks complete'}",flush=True)
    return result


def foundations(root,plan):
    from firered_rl.curriculum_checks import frozen_check
    for stage,label,threshold in [(FOUNDATION_IDS[0],'move-choice',.99),(FOUNDATION_IDS[1],'mirror',.90)]:
        result_path=root/'stages'/stage/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
        else:
            checks={k:frozen_check(ROOT,root,f'{stage}-{k}',root/v,checkpoint=root/plan['imports'][label]) for k,v in plan['suites'][stage].items()}
            issues=[k+': qualification failed' for k,s in checks.items() if not complete(s) or rate(s)<threshold
                    or (label=='move-choice' and s.get('first_surf_rate',0)<threshold)]
            result=save_result(root,stage,dict(status='blocked' if issues else 'passed',selected=plan['imports'][label],
                checkpoint_sha256=checksum(root/plan['imports'][label]),checks=checks,issues=issues),plan)
        if result['status']=='blocked':return stage
    stage=FOUNDATION_IDS[2];dest=root/'stages'/stage/'result.json'
    if not dest.exists():
        evidence=json.loads((root/'inputs/source-evidence.json').read_text())['stages']['mirror-replication']
        save_result(root,stage,dict(status='diagnostic',source_status=evidence.get('status'),evidence=evidence,
            issues=[],note='Fresh-seed reproducibility remains a diagnostic. It does not veto the separately qualified continuation.'),plan)
    return None


def run_stage(root,plan,stage,anchor,previous,concurrency):
    from firered_rl.curriculum_checks import frozen_check
    stage_id=stage['id'];suites={k:root/v for k,v in plan['suites'][stage_id].items()}
    directory=root/'stages'/stage_id;directory.mkdir(parents=True,exist_ok=True)
    print(f'\nStage {stage_id}',flush=True)
    for baseline in ('random','greedy','rules'):
        frozen_check(ROOT,root,f'{stage_id}-baseline-{baseline}',suites['calibration'],baseline=baseline)
    jobs=jobs_for(root,plan,stage,anchor)
    launch(root,plan,jobs,concurrency)
    candidates=[]
    for job in jobs:
        run=root/'runs'/job['name'];meta=json.loads((run/'checkpoints/best.json').read_text())
        path=run/'checkpoints/best.zip'
        if checksum(path)!=meta['checkpoint_sha256']:raise ValueError('Best checkpoint metadata mismatch')
        candidates.append(dict(name=job['name'],checkpoint=str(path.relative_to(root)),validation_rate=meta['wins']/meta['requested']))
    selected=max(candidates,key=lambda c:c['validation_rate'])
    # Selection precedes all held-out results. Never swap to a different arm after test failure.
    write_json(directory/'selection.json',dict(selected=selected,candidates=candidates,criterion='highest best validation wins/requested; earlier arm retains ties'))
    all_checks={};cache={}
    for candidate in candidates:
        for kind in ('best','final'):
            path=root/'runs'/candidate['name']/'checkpoints'/f'{kind}.zip';identity=checksum(path)
            if identity not in cache:
                cache[identity]={condition:frozen_check(ROOT,root,f"{candidate['name']}-{kind}-{condition}",suites[condition],checkpoint=path)
                    for condition in ('test','heldout','permuted')}
            all_checks[candidate['name']+'-'+kind]=cache[identity]
    selected_path=root/selected['checkpoint'];checks=cache[checksum(selected_path)]
    retention={}
    retention_suites={'fixed-mirror':root/plan['suites'][FOUNDATION_IDS[1]]['test']}
    if previous:retention_suites[previous]=root/plan['suites'][previous]['heldout']
    for label,suite in retention_suites.items():
        retention[label]={which:frozen_check(ROOT,root,f'{stage_id}-retain-{label}-{which}',suite,checkpoint=path)
            for which,path in [('before',root/plan['imports']['mirror'] if label=='fixed-mirror' else root/anchor),('after',selected_path)]}
    issues=broad_gate(stage,checks,retention,plan['config'])
    if selected['validation_rate'] < stage['training']['validation_win_rate']:
        issues.append('selected checkpoint did not meet the validation threshold')
    checkpoint=root/'inputs/selected'/f'{stage_id}.zip';checkpoint.parent.mkdir(exist_ok=True)
    if checkpoint.exists() and checksum(checkpoint)!=checksum(selected_path):raise ValueError('Selected copy changed')
    if not checkpoint.exists():shutil.copyfile(selected_path,checkpoint)
    return save_result(root,stage_id,dict(status='blocked' if issues else 'passed',selected=str(checkpoint.relative_to(root)),
        checkpoint_sha256=checksum(checkpoint),selected_run=selected['name'],checks=checks,retention=retention,
        all_checks=all_checks,issues=issues),plan)


def benchmark(root,plan,anchor):
    from firered_rl.curriculum_checks import frozen_check
    stage=plan['config']['stages'][-1]['id'];suite=root/plan['suites'][stage]['benchmark']
    results={}
    for opponent in ('rules','greedy','foundation'):
        results[opponent]=frozen_check(ROOT,root,f'final-benchmark-{opponent}',suite,checkpoint=root/anchor,
            opponent_kind='checkpoint' if opponent=='foundation' else opponent,
            opponent_checkpoint=root/plan['imports']['mirror'] if opponent=='foundation' else None)
    if plan['imports'].get('working-reference'):
        results['working-reference-vs-rules']=frozen_check(ROOT,root,'final-benchmark-working-reference',suite,
            checkpoint=root/plan['imports']['working-reference'])
    write_json(root/'benchmark.json',dict(results=results,selection='No checkpoint selection, training or progression depends on these final scores.'))


def inspect(root,plan,stage_filter,split,limit,full):
    entries={FOUNDATION_IDS[0]:plan['fixed']['move-choice'],FOUNDATION_IDS[1]:plan['fixed']['mirror'],**plan['assets']}
    stage_filter={'move-choice':FOUNDATION_IDS[0],'mirror':FOUNDATION_IDS[1]}.get(stage_filter,stage_filter)
    selected=entries if stage_filter=='all' else {k:v for k,v in entries.items() if k==stage_filter or k.split('-')[0]==stage_filter.zfill(2)}
    if not selected:raise ValueError('Unknown stage. Use --inspect-teams without a stage to list all.')
    for stage,entry in selected.items():
        paths=entry.get(split+'_paths',[])
        print(f"\n{stage}: {len(paths)} {split} teams")
        if 'diversity' in entry:print(json.dumps(entry['diversity'].get(split,entry['diversity']),indent=2))
        mixture=root/'inputs'/f'{stage}-mixture.json'
        if mixture.exists():print('Training mixture: '+str(mixture)+'\n'+mixture.read_text())
        for relative in paths[:limit]:
            path=Path(entry['root'])/relative;team=json.loads(path.read_text());print(str(path))
            if full:print(json.dumps(team,indent=2));continue
            for mon in team['party']:
                print(f"  {mon['species']}: {', '.join(mon['moves'])} | {mon.get('nature','Hardy')} | {mon.get('held_item','None')}")


def collect(root,plan):
    lines=['# Team curriculum','', 'SMOKE TEST ONLY; learning gates bypassed.' if plan['smoke'] else
           'Stages 1–3 reuse and requalify the fixed-team foundation. Training progression uses Rules; final benchmark names other opponents.', '',
           '| Stage | Status | Training teams | Seen-team test | Heldout | Rotated |','|---|---|---:|---:|---:|---:|']
    notes=[]
    for stage_id in [*FOUNDATION_IDS,*[s['id'] for s in plan['config']['stages']]]:
        path=root/'stages'/stage_id/'result.json';r=json.loads(path.read_text()) if path.exists() else {'status':'pending'}
        checks=r.get('checks',{});n=len(plan['assets'].get(stage_id,{}).get('train_paths',[]))
        scores=[f'{rate(checks[k]):.1%}' if k in checks else '—' for k in ('test','heldout','permuted')]
        lines.append(f"| {stage_id} | {r['status']} | {n or 'fixed/diagnostic'} | {' | '.join(scores)} |")
        notes += [f'- {stage_id}: {s}' for s in r.get('issues',[])]
        if stage_id==FOUNDATION_IDS[2] and r.get('source_status'):notes.append(f"- Replication source status: {r['source_status']}. Retained as diagnostic evidence, not relabeled as success.")
    lines += ['',*notes,'','## Frozen checks','','| Check | Opponent | Wins/requested | Rate | Truncated |','|---|---|---:|---:|---:|']
    for path in sorted((root/'checks').glob('*/summary.json')):
        r=json.loads(path.read_text())
        if r.get('status')=='completed':lines.append(f"| {path.parent.name} | {r.get('opponent_kind','rules')} | {r['candidate_wins']}/{r['requested_battles']} | {rate(r):.1%} | {r['truncated']} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    temp=root/'curriculum-results.tmp'
    with ZipFile(temp,'w',ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p.suffix not in ('.json','.jsonl','.md','.csv'):continue
            if rel.parts[0] in ('inputs','stages','checks') or p.name in ('plan.json','status.json','report.md','memory.csv','benchmark.json',
                'config.json','summary.json','failure.json','updates.jsonl','battle-blocks.jsonl','progress.csv','switching.json',
                'evaluation-history.json','evaluations.md','evaluations.csv','best.json'):
                z.write(p,rel)
    temp.replace(root/'curriculum-results.zip')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,default=Path('.local/run/staged-team-lab-01'))
    p.add_argument('--reference-checkpoint',type=Path,help='Optional broad working reference for final benchmark; defaults to the preserved 400k model if present')
    p.add_argument('--config',type=Path,default=ROOT/'config/experiments/team-curriculum.json')
    p.add_argument('--output-dir',type=Path,default=Path('.local/run/team-curriculum-01'))
    p.add_argument('--jobs',type=int,choices=(1,2),default=2)
    p.add_argument('--continue',dest='continuing',action='store_true')
    p.add_argument('--prepare-only',action='store_true',help='Generate and pin pools without starting training; continue later')
    p.add_argument('--inspect-teams',nargs='?',const='all',help='Read saved team pools: stage ID/number, or all')
    p.add_argument('--split',choices=('train','validation','heldout','benchmark'),default='train')
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--full-json',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Tiny integration run, explicitly bypassing learning gates')
    args=p.parse_args();root=args.output_dir.expanduser().resolve()
    if args.limit<0:p.error('--limit cannot be negative')
    if args.inspect_teams is not None:
        inspect(root,json.loads((root/'plan.json').read_text()),args.inspect_teams,args.split,args.limit,args.full_json)
        return 0
    import torch
    torch.set_num_threads(1)
    root.parent.mkdir(parents=True,exist_ok=True)
    lock=(root.parent/f'.{root.name}.curriculum.lock').open('a');plan=None;owned=False
    stopped=threading.Event();monitor=None;previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
        if args.continuing:
            plan=json.loads((root/'plan.json').read_text());verify(root,plan)
        else:plan=freeze(root,args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root/'console'))}/*.log\nInspect teams: python run_team_curriculum.py --output-dir {shlex.quote(str(root))} --inspect-teams 04",flush=True)
        if args.prepare_only:
            write_json(root/'status.json',dict(status='prepared'));return 0
        monitor=threading.Thread(target=memory_monitor,args=(root,stopped),daemon=True);monitor.start()
        write_json(root/'status.json',dict(status='running',stage='foundation-qualification'))
        blocked=foundations(root,plan);collect(root,plan)
        if blocked:
            write_json(root/'status.json',dict(status='blocked',stage=blocked));return 2
        anchor=plan['imports']['mirror'];previous_stage=None
        for stage in plan['config']['stages']:
            path=root/'stages'/stage['id']/'result.json'
            if path.exists():
                result=json.loads(path.read_text())
                if checksum(root/result['selected'])!=result['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            else:
                verify(root,plan)
                write_json(root/'status.json',dict(status='running',stage=stage['id']))
                result=run_stage(root,plan,stage,anchor,previous_stage,args.jobs)
            collect(root,plan)
            if result['status']=='blocked':
                write_json(root/'status.json',dict(status='blocked',stage=stage['id'],issues=result['issues']));return 2
            anchor=result['selected'];previous_stage=stage['id']
        benchmark(root,plan,anchor)
        write_json(root/'status.json',dict(status='smoke_completed' if plan['smoke'] else 'completed'))
        return 0
    except (KeyboardInterrupt,Exception) as error:
        if plan is not None and owned:write_json(root/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',message=str(error)))
        print(f'Stopped: {error or "interrupted"}',file=sys.stderr,flush=True)
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor:monitor.join()
        if plan is not None and owned:
            collect(root,plan);print(f"Report: {root/'report.md'}\nUpload: {root/'curriculum-results.zip'}",flush=True)
        signal.signal(signal.SIGTERM,previous);lock.close()


if __name__=='__main__':raise SystemExit(main())
