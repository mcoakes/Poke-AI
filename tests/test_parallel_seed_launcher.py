"""Standard-library checks; includes real subprocess orchestration with a fake trainer."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('seed_launcher', ROOT/'run_parallel_training.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def checkpoint(path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w') as z:
        z.writestr('data', json.dumps({'num_timesteps': steps}))


class ResumeSeedTest(unittest.TestCase):
    def test_explicit_and_inherited_seed_paths(self):
        tree = ast.parse((ROOT/'mgba/train_ppo.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "continuing or 'seed' in plan.explicit")
        code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<seed branch>', 'exec')
        for continuing, explicit, expected in [(False, {'seed'}, ('model', 22)),
                                               (False, set(), ('vec', 11)),
                                               (True, set(), ('model', 22))]:
            calls = []
            env = dict(continuing=continuing, plan=SimpleNamespace(explicit=explicit),
                       args=SimpleNamespace(seed=22),
                       model=SimpleNamespace(seed=11, set_random_seed=lambda v: calls.append(('model', v))),
                       vec=SimpleNamespace(seed=lambda v: calls.append(('vec', v))))
            exec(code, env)
            self.assertEqual(calls, [expected])


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'mgba').mkdir()
        (self.root/'teams').mkdir()
        (self.root/'teams/a.json').write_text('{}')
        for name in ['reward.json', 'suite.json']:
            (self.root/name).write_text('{}')
        checkpoint(self.root/'reference.zip', 400)
        (self.root/'mgba/train_ppo.py').write_text('''import json,signal,sys,time,os
from pathlib import Path
from zipfile import ZipFile
args=sys.argv[1:]
def arg(k):return args[args.index(k)+1]
run=Path(arg('--output-dir'));run.mkdir(parents=True,exist_ok=True)
(run/'config.json').write_text(json.dumps({'argv':args}))
with open(run.parent/'events.jsonl','a') as f:f.write(json.dumps({'event':'start','run':run.name,'time':time.time()})+'\\n')
steps=400
latest=run/'checkpoints/latest.zip'
if latest.exists():
 with ZipFile(latest) as z:steps=json.loads(z.read('data'))['num_timesteps']
budget=int(arg('--total-timesteps'));status='completed'
try:time.sleep(float(os.environ.get('FAKE_DELAY','.15')))
except KeyboardInterrupt:status='interrupted'
latest.parent.mkdir(exist_ok=True)
with ZipFile(latest,'w') as z:z.writestr('data',json.dumps({'num_timesteps':steps+(budget if status=='completed' else budget//2)}))
(run/'summary.json').write_text(json.dumps({'status':status}))
(run/'evaluations.md').write_text('fake evaluation')
with open(run.parent/'events.jsonl','a') as f:f.write(json.dumps({'event':'end','run':run.name,'time':time.time()})+'\\n')
print('done',run.name)
''')
        self.cmd = [sys.executable, str(ROOT/'run_parallel_training.py'), '--output-dir', 'batch',
                    '--checkpoint', 'reference.zip', '--team-dir', 'teams',
                    '--reward-config', 'reward.json', '--eval-suite', 'suite.json',
                    '--total-timesteps', '100', '--jobs', '2']

    def tearDown(self): self.tmp.cleanup()

    def run_cmd(self, args=None, **kw):
        return subprocess.run(args or self.cmd, cwd=self.root, text=True, capture_output=True, timeout=15, **kw)

    def test_dry_run_creates_nothing(self):
        r = self.run_cmd(self.cmd+['--dry-run'])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.root/'batch').exists())
        self.assertEqual(r.stdout.count('--seed'), 4)

    def test_queue_report_lock_and_completed_skip(self):
        r = self.run_cmd();self.assertEqual(r.returncode, 0, r.stderr)
        events=[json.loads(l) for l in (self.root/'batch/events.jsonl').read_text().splitlines()]
        active=peak=0
        for e in events:
            active += 1 if e['event']=='start' else -1
            peak=max(peak,active)
        self.assertEqual(peak,2);self.assertEqual(active,0)
        with ZipFile(self.root/'batch/batch-results.zip') as z:
            self.assertIn('seed-261020/summary.json',z.namelist())
        self.assertNotEqual(self.run_cmd().returncode,0)
        r=self.run_cmd(self.cmd+['--continue-batch'])
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len((self.root/'batch/events.jsonl').read_text().splitlines()),8)

    def test_graceful_stop_and_remaining_budget(self):
        env=os.environ.copy();env['FAKE_DELAY']='5'
        proc=subprocess.Popen(self.cmd,cwd=self.root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        events=self.root/'batch/events.jsonl'
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if events.exists() and len(events.read_text().splitlines())>=2:break
            time.sleep(.05)
        proc.send_signal(signal.SIGINT)
        out,err=proc.communicate(timeout=10)
        self.assertEqual(proc.returncode,130,(out,err))
        self.assertEqual(launcher.recovered_steps(self.root/'batch/seed-261020'),450)
        r=self.run_cmd(self.cmd+['--continue-batch','--jobs','4'])
        self.assertEqual(r.returncode,0,r.stderr)
        for seed in [261020,261021,261022,261023]:
            self.assertEqual(launcher.recovered_steps(self.root/f'batch/seed-{seed}'),500)

    def strategies(self):
        path = self.root/'experiments.json'
        path.write_text(json.dumps(dict(version=1, experiments=[
            dict(name='lower-lr', seeds=[261020,261021], learning_rate=.00005),
            dict(name='mixed', seeds=[261020,261021], rules_weight=.7, frozen_reference_weight=.3)])))
        return self.cmd+['--experiments',str(path)]

    def test_strategies_forward_flags_pin_pool_and_package(self):
        r=self.run_cmd(self.strategies());self.assertEqual(r.returncode,0,r.stderr)
        for seed in [261020,261021]:
            for strategy in ['lower-lr','mixed']:
                name=f'{strategy}-seed-{seed}'
                args=json.loads((self.root/'batch'/name/'config.json').read_text())['argv']
                def arg(flag):return args[args.index(flag)+1]
                self.assertEqual(arg('--seed'),str(seed))
                self.assertEqual(float(arg('--ent-coef')),.01)
                self.assertEqual(arg('--total-timesteps'),'100')
                self.assertEqual(arg('--eval-every'),'200000')
                self.assertEqual(arg('--resume'),str(self.root/'batch/inputs/reference.zip'))
                self.assertEqual(float(arg('--learning-rate')),.00005 if strategy=='lower-lr' else .0001)
                if strategy=='mixed':
                    self.assertNotIn('--rule-based-opponent',args)
                    payload=json.loads(Path(arg('--opponent-pool')).read_text())
                    self.assertEqual(payload['opponents']['rules']['weight'],.7)
                    self.assertEqual(payload['opponents']['reference']['weight'],.3)
                    pinned=Path(arg('--opponent-pool')).parent/payload['opponents']['reference']['path']
                    self.assertEqual(pinned.read_bytes(),(self.root/'reference.zip').read_bytes())
                else:
                    self.assertIn('--rule-based-opponent',args)
                    self.assertNotIn('--opponent-pool',args)
        # Editing the source experiment file cannot change a saved plan.
        (self.root/'experiments.json').write_text('invalid changed file')
        r=self.run_cmd(self.cmd+['--continue-batch']);self.assertEqual(r.returncode,0,r.stderr)
        with ZipFile(self.root/'batch/batch-results.zip') as z:
            self.assertIn('lower-lr-seed-261020/config.json',z.namelist())
            self.assertIn('inputs/mixed-seed-261020-opponents.json',z.namelist())
        pool=self.root/'batch/inputs/mixed-seed-261020-opponents.json'
        pool.write_text('{}')
        r=self.run_cmd(self.cmd+['--continue-batch'])
        self.assertNotEqual(r.returncode,0);self.assertIn('Pinned opponent pool changed',r.stderr)

    def test_legacy_batch_plan_is_still_resumable(self):
        self.assertEqual(self.run_cmd().returncode,0)
        path=self.root/'batch/batch-plan.json';plan=json.loads(path.read_text())
        plan.pop('experiments');plan.pop('reward_sha256')
        path.write_text(json.dumps(plan))
        r=self.run_cmd(self.cmd+['--continue-batch']);self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len((self.root/'batch/events.jsonl').read_text().splitlines()),8)

    def test_invalid_experiments_are_rejected_before_writing(self):
        cases=[dict(name='typo',seeds=[1],learning_rte=.01),
               dict(name='bad',seeds=[1],learning_rate=0),
               dict(name='bad',seeds=[1],ent_coef=float('nan')),
               dict(name='bad',seeds=[1],rules_weight=0,frozen_reference_weight=0),
               dict(name='../escape',seeds=[1]),dict(name='duplicates',seeds=[1,1])]
        for entry in cases:
            path=self.root/'experiments.json'
            path.write_text(json.dumps(dict(version=1,experiments=[entry])))
            r=self.run_cmd(self.cmd+['--experiments',str(path)])
            self.assertNotEqual(r.returncode,0)
            self.assertFalse((self.root/'batch').exists())

    def test_highest_counter_not_filename(self):
        run=self.root/'run'
        checkpoint(run/'checkpoints/latest.zip',450)
        checkpoint(run/'checkpoints/final.zip',420)
        checkpoint(run/'checkpoints/ppo_000460.zip',460)
        self.assertEqual(launcher.recovered_steps(run),460)


if __name__=='__main__': unittest.main()
