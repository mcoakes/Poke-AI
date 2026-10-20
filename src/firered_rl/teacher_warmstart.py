"""Training-only Rules imitation followed by a matched, fresh-optimizer PPO start.

No evaluation teams or game-result labels enter the fit. Native Showdown sessions
collect frozen student and Rules trajectories. The critic heads are frozen during
imitation; the actor's shared encoder is trained, so value predictions can change.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
from threading import Event
from types import SimpleNamespace

import numpy as np
import torch

VERSION = 1
DEFAULTS = dict(battles=2000, max_rows=50000, decisions_per_battle=24, epochs=8,
                batch_size=256, learning_rate=1e-4, validation_fraction=.2, seed=261140)
OUTPUTS = ('inputs/ppo-only.zip', 'inputs/teacher-ppo.zip', 'models/teacher-only.zip')


def _checksum(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Warmstart path escapes run directory')
    return path


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt('Teacher warmstart interrupted')


def parameter_digest(policy):
    """Stable parameter identity, independent of ZIP timestamps and Adam state."""
    h = hashlib.sha256()
    for name, parameter in policy.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def critic_parameters(policy):
    """Only modules exclusive to value prediction; shared encoder is not frozen."""
    result = {name: value for name, value in policy.named_parameters()
              if name.startswith('mlp_extractor.critic.') or name.startswith('value_net.')}
    if not result:
        raise ValueError('Warmstart requires the shared-move policy critic modules')
    return result


def freeze_critic(policy):
    frozen = critic_parameters(policy)
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name not in frozen)
    return list(frozen)


def reset_optimizer(model, learning_rate=None):
    """Keep all model tensors/counters, discard PPO moments in both comparison arms."""
    policy = model.policy
    before = parameter_digest(policy)
    learning_rate = (float(policy.optimizer.param_groups[0]['lr']) if learning_rate is None
                     else float(learning_rate))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Optimizer learning rate must be positive and finite')
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.optimizer = policy.optimizer_class(policy.parameters(), lr=learning_rate,
                                                **policy.optimizer_kwargs)
    if policy.optimizer.state or parameter_digest(policy) != before:
        raise ValueError('Optimizer reset changed model parameters')
    return dict(kind=policy.optimizer_class.__name__, state_entries=0,
                learning_rate=learning_rate, parameters_sha256=before,
                num_timesteps=int(model.num_timesteps))


def tied_best_mask(scores, mask, tolerance=1e-8):
    scores, mask = np.asarray(scores, dtype=np.float64), np.asarray(mask, dtype=bool)
    if scores.shape != (11,) or mask.shape != (11,) or not mask.any():
        raise ValueError('Expected eleven action scores and a nonempty authoritative mask')
    if tolerance < 0 or not math.isfinite(tolerance) or not np.isfinite(scores[mask]).all():
        raise ValueError('Legal teacher scores and tie tolerance must be finite')
    return mask & (scores >= scores[mask].max() - tolerance)


def imitation_loss(logits, legal, tied, soft, rehearsal):
    """Rules: -log probability mass on tied best actions. Mirror: source soft CE."""
    if logits.ndim != 2 or logits.shape[-1] != 11:
        raise ValueError('Expected a batch of eleven action logits')
    legal, tied, rehearsal = legal.bool(), tied.bool(), rehearsal.bool()
    if (legal.shape != logits.shape or tied.shape != logits.shape or soft.shape != logits.shape
            or rehearsal.shape != logits.shape[:1] or not legal.any(-1).all()):
        raise ValueError('Imitation targets/masks have inconsistent shapes')
    if (tied & ~legal).any() or (~rehearsal & ~tied.any(-1)).any():
        raise ValueError('Teacher targets must include legal best actions')
    if (not torch.isfinite(logits).all() or not torch.isfinite(soft).all()
            or (soft < 0).any() or (soft.masked_select(~legal) != 0).any()):
        raise ValueError('Imitation values must be finite and probabilities legal')
    if rehearsal.any() and not torch.allclose(soft[rehearsal].sum(-1),
            torch.ones_like(soft[rehearsal].sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError('Rehearsal probabilities must sum to one')
    log_probs = torch.log_softmax(logits.masked_fill(~legal, -1e9), -1)
    teacher = -torch.logsumexp(log_probs.masked_fill(~tied, -1e9), -1)
    retained = -(soft * log_probs).sum(-1)
    return torch.where(rehearsal, retained, teacher).mean()


def split_pairs(pair_ids, groups, *, seed, fraction=.2):
    """Disjoint unordered team-pair holdout across episodes and trajectory sources.

The one fixed-mirror pair stays entirely in fitting as soft-source rehearsal.
A one-pair group cannot provide a disjoint pair holdout and remains in fitting.
"""
    if len(pair_ids) != len(groups) or not 0 < fraction < 1:
        raise ValueError('Invalid imitation split')
    membership = {}
    for key, group in zip(pair_ids, groups):
        group = 'mirror' if group == 'fixed-mirror' else group
        if group not in ('old', 'new', 'mirror'):
            raise ValueError('Unknown training group')
        if key in membership and membership[key] != group:
            raise ValueError('One team pair cannot belong to multiple training groups')
        membership[key] = group
    rng = random.Random(seed); holdout = set()
    for group in ('old', 'new'):
        keys = sorted(key for key, value in membership.items() if value == group)
        rng.shuffle(keys)
        if len(keys) >= 2:
            count = max(1, min(len(keys)-1, round(len(keys) * fraction)))
            holdout.update(keys[:count])
    return np.asarray([key in holdout for key in pair_ids], dtype=bool)


def _request(root, plan):
    config = dict(DEFAULTS, **plan.get('warmstart', {}))
    if set(config) != set(DEFAULTS):
        raise ValueError('Unknown teacher-warmstart setting')
    for key in ('battles', 'max_rows', 'decisions_per_battle', 'epochs', 'batch_size', 'seed'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid warmstart {key}')
    if (config['battles'] % 2 or config['battles'] > 2000 or config['max_rows'] > 50000 or config['epochs'] > 8
            or config['decisions_per_battle'] > 24 or config['batch_size'] > 256
            or not 0 < config['validation_fraction'] < 1
            or not 0 < config['learning_rate'] <= 1e-3):
        raise ValueError('Warmstart exceeds its bounded collection/fit contract')
    inputs = {}
    for relative in ('inputs/start.zip', 'inputs/reward.json'):
        sha = _checksum(_inside(root, relative))
        if plan['pins'].get(relative) != sha:
            raise ValueError(f'Warmstart source input changed: {relative}')
        inputs[relative] = sha
    teams = {}
    for group in ('old', 'new', 'mirror'):
        paths = plan['teams'][group]
        if len(paths) < 2 or len(paths) != len(set(paths)) or (group == 'mirror' and len(paths) != 2):
            raise ValueError('Warmstart needs old/new pools and one two-file fixed mirror')
        teams[group] = list(paths)
        for relative in paths:
            sha = _checksum(_inside(root, relative))
            if plan['pins'].get(relative) != sha:
                raise ValueError(f'Warmstart training team changed: {relative}')
            inputs[relative] = sha
    if (type(plan['settings']['environments']) is not int or not 1 <= plan['settings']['environments'] <= 48
            or type(plan['settings']['max_steps']) is not int or not 1 <= plan['settings']['max_steps'] <= 100
            or not math.isfinite(float(plan['settings']['learning_rate']))
            or float(plan['settings']['learning_rate']) <= 0):
        raise ValueError('Invalid bounded warmstart runtime settings')
    from .rule_opponent import RuleBasedOpponent
    project = Path(__file__).resolve().parents[2]
    return dict(version=VERSION, config=config, inputs=inputs, teams=teams,
        environments=int(plan['settings']['environments']), max_steps=int(plan['settings']['max_steps']),
        ppo_learning_rate=float(plan['settings']['learning_rate']),
        teacher=RuleBasedOpponent(project).identity, implementation_sha256=_checksum(__file__),
        scope='training pools only; all legal multi-action decisions; fixed-mirror source soft rehearsal',
        weights=dict(old=.25, new=.65, mirror=.10))


def _suite(root, request):
    from .evaluation import digest, validate_suite
    from .broad_team_curriculum import semantic_signature
    seed, count = request['config']['seed'], request['config']['battles'] // 2
    rng = random.Random(seed); assets = {}; pools = {}
    for group, paths in request['teams'].items():
        pools[group] = []
        for relative in paths:
            asset = dict(team=json.loads(_inside(root, relative).read_text()))
            key = digest(asset); assets[key] = asset; pools[group].append(key)
    # Deterministic exact group proportions whenever the half-budget is a multiple of20.
    counts = dict(old=int(count*.25), mirror=int(count*.10))
    counts['new'] = count-sum(counts.values())
    schedule = [group for group in ('old', 'new', 'mirror') for _ in range(counts[group])]
    rng.shuffle(schedule); cases = []
    for index, group in enumerate(schedule):
        keys = list(pools[group]) if group == 'mirror' else rng.sample(pools[group], 2)
        pair = digest(sorted(semantic_signature(assets[key]['team']) for key in keys))
        cases.append(dict(episode=index+1, matchup=index+1, assets=keys,
            seed=rng.randrange(2**31-1), candidate_player=1+(index % 2),
            training_group=group, imitation_pair=pair))
    return validate_suite(dict(version=1, backend='showdown', seed=seed, swap_sides=False,
        max_steps=request['max_steps'], assets=assets, cases=cases,
        rule_opponent=request['teacher'], diagnostic=dict(purpose='training-only imitation collection',
            same_schedule_for_student_and_rules=True, group_battles_per_trajectory=counts)))


class _Dataset:
    """Fixed-capacity disk-backed arrays; collection never retains all observations in RAM."""
    def __init__(self, directory, capacity):
        self.directory = Path(directory); self.directory.mkdir()
        self.capacity = capacity; self.arrays = {}; self.rows = 0
        self.pairs, self.groups = [], []
        self.counts = Counter(); self.schema = {}

    def initialize(self, obs):
        for name, value in obs.items():
            value = np.asarray(value, dtype=np.float32)
            self.schema[name] = list(value.shape)
            self.arrays['obs__'+name] = np.lib.format.open_memmap(
                self.directory / ('obs__'+name+'.npy'), mode='w+', dtype=np.float32,
                shape=(self.capacity, *value.shape))
        for name, shape, dtype in (('legal', (11,), np.bool_), ('tied', (11,), np.bool_),
                ('soft', (11,), np.float32), ('rehearsal', (), np.bool_),
                ('episode', (), np.int32), ('decision', (), np.int32), ('forced', (), np.bool_)):
            self.arrays[name] = np.lib.format.open_memmap(self.directory/(name+'.npy'),
                mode='w+', dtype=dtype, shape=(self.capacity, *shape))

    def flush(self):
        for value in self.arrays.values(): value.flush()

    def batch(self, indexes, device):
        observations = {name: torch.as_tensor(np.asarray(self.arrays['obs__'+name][indexes]), device=device)
                        for name in self.schema}
        targets = [torch.as_tensor(np.asarray(self.arrays[name][indexes]), device=device)
                   for name in ('legal', 'tied', 'soft', 'rehearsal')]
        return observations, targets


class _Collector:
    def __init__(self, dataset, source, rules, request, trajectory, cancel):
        self.data, self.source, self.rules = dataset, source, rules
        self.request, self.trajectory, self.cancel = request, trajectory, cancel

    def __call__(self, case, decision, obs, action):
        _cancelled(self.cancel)
        data, config = self.data, self.request['config']
        legal = np.asarray(obs['action_mask'], dtype=bool)
        data.counts['observed_decisions'] += 1
        if decision > config['decisions_per_battle']:
            data.counts['skipped_decision_limit'] += 1; return
        if legal.sum() <= 1:
            data.counts['skipped_single_legal'] += 1; return
        if data.rows >= data.capacity:
            data.counts['skipped_row_capacity'] += 1; return
        if not data.arrays: data.initialize(obs)
        group = case['training_group']; rehearsal = group == 'mirror'
        soft = np.zeros(11, dtype=np.float32); tied = np.zeros(11, dtype=bool)
        if rehearsal:
            with torch.no_grad():
                tensor, _ = self.source.policy.obs_to_tensor(obs)
                distribution = self.source.policy.get_distribution(tensor, action_masks=legal[None])
                soft = distribution.distribution.probs[0].cpu().numpy().astype(np.float32)
            if np.any(soft[~legal]) or not np.isclose(soft.sum(), 1):
                raise ValueError('Source rehearsal distribution violates legal mask')
        else:
            _, candidates = self.rules.choose(obs, legal)
            scores = np.full(11, -np.inf)
            for row in candidates: scores[row['action']] = row['score']
            tied = tied_best_mask(scores, legal)
        index = data.rows
        for name, value in obs.items(): data.arrays['obs__'+name][index] = value
        for name, value in dict(legal=legal, tied=tied, soft=soft, rehearsal=rehearsal,
                episode=case['episode']+(0 if self.trajectory == 'student' else config['battles']//2),
                decision=decision, forced=bool(obs['context'][1])).items():
            data.arrays[name][index] = value
        data.pairs.append(case['imitation_pair']); data.groups.append(group)
        data.counts['trajectory/'+self.trajectory] += 1
        data.counts['group/'+group] += 1
        data.counts['target/'+('source_soft' if rehearsal else 'rules_tied_best')] += 1
        data.counts['decision/'+('forced' if obs['context'][1] else 'ordinary')] += 1
        data.counts['executed/'+('move' if action < 4 else 'switch' if action < 10 else 'wait')] += 1
        if tied.sum() > 1: data.counts['tied_best_rows'] += 1
        data.rows += 1


def _collect(project, root, attempt, request, source, cancel, announce):
    from .evaluation import evaluate, materialize_assets
    from .linked_reward import LinkedRewardConfig
    from .rule_opponent import RuleBasedOpponent
    from .showdown_env import ShowdownBattleSession, ShowdownFactory
    suite = _suite(root, request); _write(attempt/'collection-suite.json', suite)
    data = _Dataset(attempt/'dataset', request['config']['max_rows'])
    summaries = {}; rules = RuleBasedOpponent(project)
    for trajectory, candidate in (('student', source), ('rules', rules)):
        _cancelled(cancel); run = attempt/('collection-'+trajectory); run.mkdir()
        _write(run/'suite.json', suite); paths = materialize_assets(suite, run)
        selected = [[] for _ in range(min(request['environments'], len(suite['cases'])))]
        args = SimpleNamespace(seed=suite['seed'], node='node', command_timeout=120,
            observation_version='v2', step_logs=False, log_max_mb=2, log_backups=1, keep_episode_logs=0)
        sessions = []
        def progress(message):
            first = message.split()[0]
            done = int(first.split('=')[1].split('/')[0])
            if done % 50 == 0 or done == len(suite['cases']):
                announce(f'Warmstart {trajectory}: {message}')
        try:
            for i in range(len(selected)):
                factory = ShowdownFactory(project, run/f'pair-{i+1}', [55355+2*i, 55356+2*i],
                    lambda i=i: tuple(selected[i]), args, cancel)
                sessions.append(ShowdownBattleSession(factory, max_steps=suite['max_steps'],
                    reward_config=LinkedRewardConfig.load(root/'inputs/reward.json')))
            summaries[trajectory] = evaluate(sessions, selected, suite, paths,
                [candidate, rules], run, cancel, aborters=[s.factory.stop for s in sessions],
                announce=progress, decision_observer=_Collector(data, source, rules, request, trajectory, cancel))
            if (summaries[trajectory]['status'] != 'completed'
                    or summaries[trajectory]['attempted_battles'] != len(suite['cases'])):
                raise ValueError('Incomplete warmstart collection')
        finally:
            for session in sessions: session.close()
            data.flush()
            _write(attempt/'collection-progress.json', dict(rows=data.rows, counts=dict(data.counts),
                completed_trajectories=list(summaries)))
    return data, summaries


def _fit(model, data, request, cancel, announce):
    config = request['config']; policy = model.policy
    split = split_pairs(data.pairs, data.groups, seed=config['seed'] ^ 0xF17,
                        fraction=config['validation_fraction'])
    fit, heldout = np.flatnonzero(~split), np.flatnonzero(split)
    if not len(fit) or not len(heldout):
        raise ValueError('Collection did not provide disjoint fit and holdout team pairs')
    np.save(data.directory/'holdout.npy', split)
    _write(data.directory/'row-groups.json', dict(pairs=data.pairs, groups=data.groups))
    frozen_names = freeze_critic(policy)
    frozen = {name: value.detach().clone() for name, value in critic_parameters(policy).items()}
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],
                                  lr=config['learning_rate'], eps=1e-5)
    rng = np.random.default_rng(config['seed']); torch.manual_seed(config['seed'])
    device, size = model.device, config['batch_size']
    def loss(indexes, training=False):
        observations, targets = data.batch(indexes, device)
        logits = policy.get_distribution(observations, action_masks=targets[0]).distribution.logits
        return imitation_loss(logits, *targets)
    def measure(indexes):
        policy.set_training_mode(False); total = 0.; counts = Counter(); soft_sum = 0.
        with torch.no_grad():
            for start in range(0, len(indexes), size):
                _cancelled(cancel); part = indexes[start:start+size]
                observations, targets = data.batch(part, device)
                legal, tied, soft, rehearsal = targets
                logits = policy.get_distribution(observations, action_masks=legal).distribution.logits
                total += float(imitation_loss(logits, *targets)) * len(part)
                predicted = logits.argmax(-1)
                correct = tied.gather(1, predicted[:, None]).squeeze(1).cpu().numpy()
                teacher = ~rehearsal.cpu().numpy()
                forced = np.asarray(data.arrays['forced'][part])
                teacher_moves = tied[:, :4].any(-1).cpu().numpy()
                teacher_switches = tied[:, 4:10].any(-1).cpu().numpy()
                categories = dict(all=teacher, ordinary=teacher & ~forced, forced=teacher & forced,
                    move=teacher & teacher_moves & ~teacher_switches,
                    switch=teacher & teacher_switches & ~teacher_moves,
                    mixed=teacher & teacher_moves & teacher_switches)
                for category, subset in categories.items():
                    counts[category+'/rows'] += int(subset.sum())
                    counts[category+'/correct'] += int(correct[subset].sum())
                if rehearsal.any():
                    log_probs = torch.log_softmax(logits.masked_fill(~legal, -1e9), -1)
                    soft_sum += float(-(soft[rehearsal] * log_probs[rehearsal]).sum())
                    counts['rehearsal_rows'] += int(rehearsal.sum())
        agreement = {name: dict(rows=counts[name+'/rows'], correct=counts[name+'/correct'],
            rate=counts[name+'/correct']/counts[name+'/rows'] if counts[name+'/rows'] else None)
            for name in ('all', 'ordinary', 'forced', 'move', 'switch', 'mixed')}
        return dict(loss=total / len(indexes), rules_tie_aware_agreement=agreement,
            rehearsal_rows=counts['rehearsal_rows'],
            rehearsal_soft_ce=soft_sum/counts['rehearsal_rows'] if counts['rehearsal_rows'] else None)
    initial_fit, initial_holdout = measure(fit), measure(heldout)
    initial = dict(epoch=0, fit_loss=initial_fit['loss'], holdout_loss=initial_holdout['loss'],
                   fit_metrics=initial_fit, holdout_metrics=initial_holdout)
    history = [initial]; best = deepcopy(policy.state_dict()); best_epoch = 0
    best_loss = initial['holdout_loss']; updates = 0
    announce(f"Warmstart initial: fit={initial['fit_loss']:.6f}; pair-holdout={best_loss:.6f}")
    for epoch in range(1, config['epochs']+1):
        policy.set_training_mode(True); indexes = rng.permutation(fit)
        for start in range(0, len(indexes), size):
            _cancelled(cancel); value = loss(indexes[start:start+size])
            if not torch.isfinite(value): raise ValueError('Nonfinite imitation loss')
            optimizer.zero_grad(set_to_none=True); value.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], .5)
            if not torch.isfinite(norm): raise ValueError('Nonfinite imitation gradient')
            optimizer.step(); updates += 1
        fit_metrics, holdout_metrics = measure(fit), measure(heldout)
        row = dict(epoch=epoch, fit_loss=fit_metrics['loss'], holdout_loss=holdout_metrics['loss'],
                   fit_metrics=fit_metrics, holdout_metrics=holdout_metrics); history.append(row)
        if row['holdout_loss'] < best_loss:
            best_loss, best_epoch = row['holdout_loss'], epoch; best = deepcopy(policy.state_dict())
        announce(f"Warmstart epoch {epoch}/{config['epochs']}: fit={row['fit_loss']:.6f}; "
                 f"pair-holdout={row['holdout_loss']:.6f}; best epoch={best_epoch}")
    policy.load_state_dict(best)
    if any(not torch.equal(value, critic_parameters(policy)[name]) for name, value in frozen.items()):
        raise ValueError('Imitation changed frozen critic parameters')
    policy.set_training_mode(False)
    fit_pairs = {data.pairs[i] for i in fit}; heldout_pairs = {data.pairs[i] for i in heldout}
    if fit_pairs & heldout_pairs: raise ValueError('Imitation pair split leaked')
    return dict(history=history, selected_epoch=best_epoch, selected_holdout_loss=best_loss,
        gradient_updates=updates, fit_rows=len(fit), holdout_rows=len(heldout),
        fit_pairs=len(fit_pairs), holdout_pairs=len(heldout_pairs),
        split='disjoint unordered training-team pairs across both trajectory policies; fixed mirror fit-only',
        initial_metrics=dict(fit=initial_fit, holdout=initial_holdout),
        selected_metrics=dict(fit=history[best_epoch]['fit_metrics'], holdout=history[best_epoch]['holdout_metrics']),
        frozen_critic_parameters=frozen_names, critic_parameters_unchanged=True,
        shared_encoder_trained=True, value_predictions_may_change=True,
        selection='minimum imitation loss on heldout training-team pairs; epoch0 retains ties; no game validation/test labels')


def verify(root, plan):
    """Verify a completed warmstart without loading its potentially large dataset."""
    root = Path(root).resolve(); request = _request(root, plan)
    path = root/'warmstart/report.json'
    if not path.is_file(): raise ValueError('Teacher warmstart is not complete')
    report = json.loads(path.read_text())
    if report.get('status') != 'completed' or report.get('request') != request:
        raise ValueError('Completed warmstart request differs')
    for relative, sha in report['artifacts'].items():
        path = _inside(root, relative)
        if not path.is_file() or _checksum(path) != sha:
            raise ValueError(f'Warmstart artifact changed: {relative}')
    if set(OUTPUTS) - set(report['artifacts']):
        raise ValueError('Warmstart model outputs missing from manifest')
    return report


def prepare(root, plan, *, announce=print, cancel=None):
    """Prepare both PPO inputs, retaining an auditable teacher-only snapshot.

Completed outputs are verified. Interrupted attempts are preserved and collection
restarts in a fresh numbered directory; different source/configuration is rejected.
The caller pins the returned artifacts/report after completion and excludes raw
warmstart/attempt-*/dataset arrays from compact diagnostic-result archives.
"""
    from sb3_contrib import MaskablePPO
    from .shared_move_policy import policy_architecture
    root = Path(root).resolve(); cancel = Event() if cancel is None else cancel
    request = _request(root, plan); directory = root/'warmstart'; directory.mkdir(exist_ok=True)
    request_path = directory/'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Partial warmstart source/configuration changed; use a new run')
    if (directory/'report.json').exists(): return verify(root, plan)
    _write(request_path, request); _cancelled(cancel)
    attempts = list(directory.glob('attempt-*'))
    number = 1 + max([int(p.name.split('-')[-1]) for p in attempts] or [0])
    attempt = directory/f'attempt-{number:04d}'; attempt.mkdir()
    project = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    source = MaskablePPO.load(root/'inputs/start.zip', device='cpu')
    if (policy_architecture(source) != 'matchup-switches'
            or not source.policy.features_extractor.relational_enabled
            or not source.policy.features_extractor.bench_features_enabled):
        raise ValueError('Teacher warmstart requires the feature-enabled selected policy')
    source.policy.set_training_mode(False)
    source_parameters = parameter_digest(source.policy); source_steps = int(source.num_timesteps)
    try:
        data, collections = _collect(project, root, attempt, request, source, cancel, announce)
        learner = MaskablePPO.load(root/'inputs/start.zip', device='cpu')
        fit = _fit(learner, data, request, cancel, announce)
        if parameter_digest(source.policy) != source_parameters:
            raise ValueError('Frozen source changed during collection/fitting')
        resets = {'ppo-only': reset_optimizer(source, request['ppo_learning_rate']),
                  'teacher-ppo': reset_optimizer(learner, request['ppo_learning_rate'])}
        if source.num_timesteps != learner.num_timesteps or source.num_timesteps != source_steps:
            raise ValueError('Imitation changed PPO timestep counters')
        learner.teacher_warmstart = dict(version=VERSION, source_sha256=request['inputs']['inputs/start.zip'],
            source_parameters_sha256=source_parameters, selected_epoch=fit['selected_epoch'],
            optimizer_reset=True, scope='all-action Rules imitation; fixed-mirror source soft rehearsal')
        source.teacher_warmstart = dict(version=VERSION, source_sha256=request['inputs']['inputs/start.zip'],
            source_parameters_sha256=source_parameters, selected_epoch=None,
            optimizer_reset=True, scope='matched PPO-only control; no imitation')
        staging = attempt/'outputs'; staging.mkdir()
        source.save(staging/'ppo-only.zip'); learner.save(staging/'teacher-ppo.zip')
        shutil.copyfile(staging/'teacher-ppo.zip', staging/'teacher-only.zip')
        # Verify actual serialized states; reset evidence is more than a prose claim.
        for name, expected in (('ppo-only', source), ('teacher-ppo', learner)):
            restored = MaskablePPO.load(staging/(name+'.zip'), device='cpu')
            if (restored.policy.optimizer.state or restored.num_timesteps != source_steps
                    or parameter_digest(restored.policy) != parameter_digest(expected.policy)):
                raise ValueError('Serialized warmstart output differs')
            del restored
        _cancelled(cancel)
        for relative in OUTPUTS:
            destination = root/relative; destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                preserved = attempt/'previous-partial-outputs'/relative
                preserved.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(destination, preserved)
            shutil.copyfile(staging/destination.name, destination)
        data.flush()
        ledger = dict(rows=data.rows, capacity=data.capacity, counts=dict(data.counts),
            observation_shapes=data.schema, files='disk-backed .npy arrays; only first rows entries are used',
            collection_limits=request['config'])
        _write(attempt/'dataset.json', ledger)
        artifacts = {relative: _checksum(root/relative) for relative in OUTPUTS}
        for path in attempt.rglob('*'):
            if path.is_file() and ('dataset' in path.parts or path.name in
                    ('dataset.json', 'collection-suite.json', 'summary.json', 'episodes.jsonl')):
                artifacts[str(path.relative_to(root))] = _checksum(path)
        report = dict(version=VERSION, status='completed', request=request,
            attempt=str(attempt.relative_to(root)), source_sha256=request['inputs']['inputs/start.zip'],
            source_parameters_sha256=source_parameters, source_timesteps=source_steps,
            dataset=ledger, collection=collections, fit=fit, optimizer_resets=resets,
            critic_note='Critic-only parameters frozen during BC; shared actor/critic encoder changes can alter values.',
            max_process_rss_mib=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024,
            artifacts=artifacts, outputs={name.split('/')[-1]:name for name in OUTPUTS},
            no_evaluation_team_labels=True, no_game_outcome_labels=True)
        _write(directory/'report.json', report)
        announce(f"Warmstart complete: {data.rows} rows, selected epoch {fit['selected_epoch']}; "
                 'both PPO optimizers reset. teacher-only.zip retained.')
        return report
    except BaseException as error:
        _write(attempt/'failure.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            type=type(error).__name__, message=str(error), source_sha256=request['inputs']['inputs/start.zip']))
        raise
