"""Bounded, decision-weighted practice on fresh training-only battle states.

The existing architecture and reward stay unchanged. A stratified reservoir
keeps rare switching examples; soft current-parent mirror rehearsal receives
exactly ten percent of each fitting objective. The old warmstart implementation
is reused for stable primitives and is never modified or monkey-patched.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import shutil
from threading import Event
from types import SimpleNamespace

import numpy as np
import torch

from . import teacher_warmstart as base

VERSION = 1
BUCKETS = ('move', 'forced', 'switch', 'mirror')
DEFAULTS = dict(battles=6000, max_rows=50000, decisions_per_battle=24, epochs=8,
    batch_size=256, learning_rate=1e-4, validation_fraction=.2, seed=261150,
    capacities=dict(move=30000, forced=10000, switch=5000, mirror=5000),
    weights=dict(move=1., forced=2., switch=4.), mirror_mass=.1)
OUTPUTS = ('inputs/ppo-only.zip', 'inputs/switch-practice.zip', 'models/practice-only.zip')


def bucket_for_targets(tied, *, forced=False, rehearsal=False):
    tied = np.asarray(tied, dtype=bool)
    if tied.shape != (11,):
        raise ValueError('Expected eleven tied-action flags')
    if rehearsal:
        return 'mirror'
    if not tied.any():
        raise ValueError('A Rules-labelled row needs at least one best action')
    if forced:
        return 'forced'
    # A tie spanning moves and switches is deliberately kept at ordinary weight1.
    return 'switch' if tied[4:10].sum() == tied.sum() else 'move'


class StratifiedReservoir:
    """Independent, uniform Algorithm-R reservoirs, with synchronized row metadata."""
    def __init__(self, capacities, seed):
        if set(capacities) != set(BUCKETS) or any(type(v) is not int or v < 1 for v in capacities.values()):
            raise ValueError('Every decision bucket needs a positive integer capacity')
        self.capacities = dict(capacities)
        self.metadata, self.buckets = [], []
        self.slots = {key: [] for key in BUCKETS}
        self.stats = {key: dict(seen=0, kept=0, replaced=0, discarded=0) for key in BUCKETS}
        self.rng = {key: random.Random(int(seed) ^ int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))
                    for key in BUCKETS}
        self.seen_trajectories = Counter()

    def offer(self, bucket, metadata):
        if bucket not in self.stats or not isinstance(metadata, dict) or 'trajectory' not in metadata:
            raise ValueError('Invalid reservoir bucket or trajectory metadata')
        state = self.stats[bucket]; state['seen'] += 1
        self.seen_trajectories[str(metadata['trajectory'])] += 1
        if state['kept'] < self.capacities[bucket]:
            index = len(self.metadata)
            self.slots[bucket].append(index); state['kept'] += 1
            self.metadata.append(deepcopy(metadata)); self.buckets.append(bucket)
            return index
        position = self.rng[bucket].randrange(state['seen'])
        if position >= self.capacities[bucket]:
            state['discarded'] += 1
            return None
        index = self.slots[bucket][position]; state['replaced'] += 1
        self.metadata[index] = deepcopy(metadata)
        return index

    def summary(self):
        return deepcopy(self.stats)

    def trajectory_counts(self):
        return dict(seen=dict(self.seen_trajectories),
            kept=dict(Counter(str(row['trajectory']) for row in self.metadata)))


def _weights(weights):
    if set(weights) != set(BUCKETS[:3]) or any(not math.isfinite(float(v)) or v <= 0 for v in weights.values()):
        raise ValueError('Teacher class weights must be positive and finite')
    return {key: float(value) for key, value in weights.items()}


def weighted_imitation_loss(logits, legal, tied, soft, buckets, *, weights=None,
                            mirror_mass=.1, with_mirror=True):
    """Normalize teacher weights by their sum; mirror has its own fixed mass.

Fitting constructs every minibatch with teacher and mirror examples. The
pair-heldout criterion calls with_mirror=False and has no rehearsal rows.
"""
    weights = _weights(DEFAULTS['weights'] if weights is None else weights)
    if buckets.shape != logits.shape[:1] or not ((buckets >= 0) & (buckets < len(BUCKETS))).all():
        raise ValueError('Invalid decision bucket indices')
    if not 0 < mirror_mass < 1:
        raise ValueError('Mirror objective mass must be between zero and one')
    total = logits.sum() * 0; denominator = 0.
    for index, bucket in enumerate(BUCKETS[:3]):
        subset = buckets == index; count = int(subset.sum())
        if count:
            value = base.imitation_loss(logits[subset], legal[subset], tied[subset], soft[subset],
                                       torch.zeros(count, dtype=torch.bool, device=logits.device))
            total = total + value * count * weights[bucket]
            denominator += count * weights[bucket]
    if denominator == 0:
        raise ValueError('A weighted objective needs teacher-labelled rows')
    teacher = total / denominator
    if not with_mirror:
        return teacher
    rehearsal = buckets == 3; count = int(rehearsal.sum())
    if not count:
        raise ValueError('A fitting objective requires mirror rehearsal rows')
    mirror = base.imitation_loss(logits[rehearsal], legal[rehearsal], tied[rehearsal], soft[rehearsal],
                                torch.ones(count, dtype=torch.bool, device=logits.device))
    return (1-mirror_mass) * teacher + mirror_mass * mirror


def ppo_counters(model):
    return dict(num_timesteps=int(model.num_timesteps),
                n_updates=int(getattr(model, '_n_updates', 0)),
                progress_remaining=float(getattr(model, '_current_progress_remaining', 1.)))


def _request(root, plan):
    config = deepcopy(DEFAULTS)
    config.update(deepcopy(plan.get('practice', {})))
    if set(config) != set(DEFAULTS):
        raise ValueError('Unknown switch-practice setting')
    for key in ('battles', 'max_rows', 'decisions_per_battle', 'epochs', 'batch_size', 'seed'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid practice {key}')
    if (config['battles'] % 2 or config['battles'] > 6000 or config['max_rows'] > 50000
            or config['epochs'] > 8 or not 2 <= config['batch_size'] <= 256
            or config['decisions_per_battle'] > 24 or not 0 < config['validation_fraction'] < 1
            or not math.isfinite(config['learning_rate']) or not 0 < config['learning_rate'] <= 1e-3
            or config['mirror_mass'] != .1):
        raise ValueError('Practice exceeds its bounded collection/fit contract')
    capacities = config['capacities']
    if (set(capacities) != set(BUCKETS) or any(type(v) is not int or v < 1 for v in capacities.values())
            or sum(capacities.values()) != config['max_rows']):
        raise ValueError('Reservoir capacities must sum to the bounded row budget')
    _weights(config['weights'])
    if config['weights'] != DEFAULTS['weights']:
        raise ValueError('This experiment fixes teacher class weights at1/2/4')
    inputs, teams = {}, {}
    for relative in ('inputs/start.zip', 'inputs/reward.json'):
        sha = base._checksum(base._inside(root, relative))
        if plan['pins'].get(relative) != sha:
            raise ValueError(f'Practice source changed: {relative}')
        inputs[relative] = sha
    for group in ('old', 'new', 'mirror'):
        paths = plan['teams'][group]
        if len(paths) < 2 or len(paths) != len(set(paths)) or (group == 'mirror' and len(paths) != 2):
            raise ValueError('Practice needs old/new pools and the fixed mirror pair')
        teams[group] = list(paths)
        for relative in paths:
            sha = base._checksum(base._inside(root, relative))
            if plan['pins'].get(relative) != sha:
                raise ValueError(f'Practice training input changed: {relative}')
            inputs[relative] = sha
    settings = plan['settings']
    if (type(settings['environments']) is not int or not 1 <= settings['environments'] <= 48
            or type(settings['max_steps']) is not int or not 1 <= settings['max_steps'] <= 100
            or not math.isfinite(float(settings['learning_rate'])) or settings['learning_rate'] <= 0):
        raise ValueError('Invalid bounded practice runtime settings')
    from .rule_opponent import RuleBasedOpponent
    project = Path(__file__).resolve().parents[2]
    return dict(version=VERSION, config=config, inputs=inputs, teams=teams,
        environments=settings['environments'], max_steps=settings['max_steps'],
        ppo_learning_rate=float(settings['learning_rate']), smoke=bool(plan.get('smoke', False)),
        teacher=RuleBasedOpponent(project).identity,
        implementation_sha256=base._checksum(__file__), primitive_sha256=base._checksum(base.__file__),
        scope='fresh training-only states; no game outcomes or endpoint labels; weighted Rules ties and soft current-parent mirror',
        weights=dict(old=.25, new=.65, mirror=.10))


class _Collector:
    def __init__(self, data, reservoir, source, rules, request, trajectory, cancel):
        self.data, self.reservoir, self.source, self.rules = data, reservoir, source, rules
        self.request, self.trajectory, self.cancel = request, trajectory, cancel

    def __call__(self, case, decision, obs, action):
        base._cancelled(self.cancel)
        data, config = self.data, self.request['config']
        data.counts['observed_decisions'] += 1
        legal = np.asarray(obs['action_mask'], dtype=bool)
        if decision > config['decisions_per_battle']:
            data.counts['skipped_decision_limit'] += 1; return
        if legal.sum() <= 1:
            data.counts['skipped_single_legal'] += 1; return
        group = case['training_group']; rehearsal = group == 'mirror'
        tied = np.zeros(11, dtype=bool)
        if not rehearsal:
            _, candidates = self.rules.choose(obs, legal)
            scores = np.full(11, -np.inf)
            for row in candidates: scores[row['action']] = row['score']
            tied = base.tied_best_mask(scores, legal)
        bucket = bucket_for_targets(tied, forced=bool(obs['context'][1]), rehearsal=rehearsal)
        episode = case['episode']+(0 if self.trajectory == 'student' else config['battles']//2)
        metadata = dict(pair=case['imitation_pair'], group=group, trajectory=self.trajectory,
                        episode=episode, decision=decision, bucket=bucket)
        index = self.reservoir.offer(bucket, metadata)
        data.counts['seen/group/'+group] += 1
        data.counts['seen/decision/'+('forced' if obs['context'][1] else 'ordinary')] += 1
        data.counts['seen/executed/'+('move' if action < 4 else 'switch' if action < 10 else 'wait')] += 1
        if tied.sum() > 1: data.counts['seen/tied_best_rows'] += 1
        if index is None:
            return
        if not data.arrays: data.initialize(obs)
        soft = np.zeros(11, dtype=np.float32)
        if rehearsal:
            with torch.no_grad():
                tensor, _ = self.source.policy.obs_to_tensor(obs)
                soft = self.source.policy.get_distribution(tensor, action_masks=legal[None]).distribution.probs[0].cpu().numpy().astype(np.float32)
            if np.any(soft[~legal]) or not np.isclose(soft.sum(), 1):
                raise ValueError('Current-parent rehearsal violates legal mask')
        if index == data.rows:
            data.pairs.append(metadata['pair']); data.groups.append(group); data.rows += 1
        else:
            data.pairs[index], data.groups[index] = metadata['pair'], group
        for name, value in obs.items(): data.arrays['obs__'+name][index] = value
        for name, value in dict(legal=legal, tied=tied, soft=soft, rehearsal=rehearsal,
                episode=episode, decision=decision, forced=bool(obs['context'][1])).items():
            data.arrays[name][index] = value


def _collect(project, root, attempt, request, source, cancel, announce):
    from .evaluation import evaluate, materialize_assets
    from .linked_reward import LinkedRewardConfig
    from .rule_opponent import RuleBasedOpponent
    from .showdown_env import ShowdownBattleSession, ShowdownFactory
    suite = base._suite(root, request)
    suite['diagnostic']['purpose'] = 'fresh training-only stratified switch practice'
    base._write(attempt/'collection-suite.json', suite)
    data = base._Dataset(attempt/'dataset', request['config']['max_rows'])
    reservoir = StratifiedReservoir(request['config']['capacities'], request['config']['seed'])
    summaries = {}; rules = RuleBasedOpponent(project)
    for trajectory, candidate in (('student', source), ('rules', rules)):
        base._cancelled(cancel); run = attempt/('collection-'+trajectory); run.mkdir()
        base._write(run/'suite.json', suite); paths = materialize_assets(suite, run)
        selected = [[] for _ in range(min(request['environments'], len(suite['cases'])))]
        args = SimpleNamespace(seed=suite['seed'], node='node', command_timeout=120,
            observation_version='v2', step_logs=False, log_max_mb=2, log_backups=1, keep_episode_logs=0)
        sessions = []
        def progress(message):
            done = int(message.split()[0].split('=')[1].split('/')[0])
            if done % 100 == 0 or done == len(suite['cases']):
                announce(f'Switch practice {trajectory}: {message}')
        try:
            for i in range(len(selected)):
                factory = ShowdownFactory(project, run/f'pair-{i+1}', [55355+2*i, 55356+2*i],
                    lambda i=i: tuple(selected[i]), args, cancel)
                sessions.append(ShowdownBattleSession(factory, max_steps=suite['max_steps'],
                    reward_config=LinkedRewardConfig.load(root/'inputs/reward.json')))
            summaries[trajectory] = evaluate(sessions, selected, suite, paths, [candidate, rules], run,
                cancel, aborters=[s.factory.stop for s in sessions], announce=progress,
                decision_observer=_Collector(data, reservoir, source, rules, request, trajectory, cancel))
            if (summaries[trajectory]['status'] != 'completed'
                    or summaries[trajectory]['attempted_battles'] != len(suite['cases'])):
                raise ValueError('Incomplete switch-practice collection')
        finally:
            for session in sessions: session.close()
            data.flush()
            base._write(attempt/'collection-progress.json', dict(rows=data.rows,
                counts=dict(data.counts), reservoir=reservoir.summary(),
                trajectories=reservoir.trajectory_counts(), completed_trajectories=list(summaries)))
    if data.rows != len(reservoir.metadata) or data.rows != sum(v['kept'] for v in reservoir.summary().values()):
        raise ValueError('Reservoir row accounting differs')
    for index, metadata in enumerate(reservoir.metadata):
        if (data.pairs[index] != metadata['pair'] or data.groups[index] != metadata['group']
                or int(data.arrays['episode'][index]) != metadata['episode']
                or int(data.arrays['decision'][index]) != metadata['decision']):
            raise ValueError('Reservoir replaced row metadata is inconsistent')
    return data, reservoir, summaries


def _fit(model, data, reservoir, request, cancel, announce):
    config, policy = request['config'], model.policy
    split = base.split_pairs(data.pairs, data.groups, seed=config['seed'] ^ 0xF17,
                             fraction=config['validation_fraction'])
    buckets = np.asarray([BUCKETS.index(value) for value in reservoir.buckets], dtype=np.int64)
    fit, heldout = np.flatnonzero(~split), np.flatnonzero(split)
    teacher_fit = fit[buckets[fit] != 3]; mirror_fit = fit[buckets[fit] == 3]
    if not len(teacher_fit) or not len(heldout) or not len(mirror_fit):
        raise ValueError('Practice requires teacher fit/holdout pairs and current-parent mirror rehearsal')
    split_counts = {name: dict(Counter(reservoir.buckets[i] for i in indexes))
                    for name, indexes in (('fit', fit), ('holdout', heldout))}
    if not request['smoke'] and any(split_counts[name].get('switch', 0) == 0 for name in ('fit', 'holdout')):
        raise ValueError('No ordinary-switch examples in fit or pair holdout; collect a new practice run, do not invent labels')
    np.save(data.directory/'holdout.npy', split); np.save(data.directory/'buckets.npy', buckets)
    base._write(data.directory/'row-groups.json', dict(pairs=data.pairs, groups=data.groups,
        buckets=reservoir.buckets, metadata=reservoir.metadata))
    frozen_names = base.freeze_critic(policy)
    frozen = {name: value.detach().clone() for name, value in base.critic_parameters(policy).items()}
    optimizer = torch.optim.Adam([value for value in policy.parameters() if value.requires_grad],
                                  lr=config['learning_rate'], eps=1e-5)
    rng = np.random.default_rng(config['seed']); torch.manual_seed(config['seed'])
    size, device = config['batch_size'], model.device
    weights = _weights(config['weights'])
    def values(indexes):
        observations, targets = data.batch(indexes, device)
        logits = policy.get_distribution(observations, action_masks=targets[0]).distribution.logits
        return logits, targets, torch.as_tensor(buckets[indexes], device=device)
    def measure(indexes):
        policy.set_training_mode(False); sums = Counter(); counts = Counter(); correct = Counter()
        with torch.no_grad():
            for start in range(0, len(indexes), size):
                base._cancelled(cancel); part = indexes[start:start+size]
                logits, targets, bucket_ids = values(part)
                legal, tied, soft, rehearsal = targets
                predictions = logits.argmax(-1)
                matches = tied.gather(1, predictions[:, None]).squeeze(1)
                for index, bucket in enumerate(BUCKETS):
                    subset = bucket_ids == index; count = int(subset.sum())
                    if not count: continue
                    loss = base.imitation_loss(logits[subset], legal[subset], tied[subset], soft[subset], rehearsal[subset])
                    sums[bucket] += float(loss) * count; counts[bucket] += count
                    if bucket != 'mirror': correct[bucket] += int(matches[subset].sum())
        teacher_count = sum(counts[key] for key in BUCKETS[:3])
        denominator = sum(counts[key]*weights[key] for key in BUCKETS[:3])
        weighted = sum(sums[key]*weights[key] for key in BUCKETS[:3])/denominator
        unweighted = sum(sums[key] for key in BUCKETS[:3])/teacher_count
        mirror = sums['mirror']/counts['mirror'] if counts['mirror'] else None
        agreement = {key: dict(rows=counts[key], correct=correct[key],
                    rate=correct[key]/counts[key] if counts[key] else None) for key in BUCKETS[:3]}
        agreement['ordinary'] = dict(rows=counts['move']+counts['switch'],
            correct=correct['move']+correct['switch'], rate=(correct['move']+correct['switch'])/(counts['move']+counts['switch'])
            if counts['move']+counts['switch'] else None)
        agreement['all'] = dict(rows=teacher_count, correct=sum(correct.values()), rate=sum(correct.values())/teacher_count)
        return dict(weighted_teacher_loss=weighted, unweighted_teacher_loss=unweighted,
            mirror_soft_ce=mirror, mirror_rows=counts['mirror'],
            objective=(1-config['mirror_mass'])*weighted+config['mirror_mass']*mirror if mirror is not None else weighted,
            rules_tie_aware_agreement=agreement,
            effective_teacher_weight_mass={key: counts[key]*weights[key]/denominator for key in BUCKETS[:3]})
    initial_fit, initial_holdout = measure(fit), measure(heldout)
    initial = dict(epoch=0, fit_loss=initial_fit['objective'], holdout_loss=initial_holdout['weighted_teacher_loss'],
                   fit_metrics=initial_fit, holdout_metrics=initial_holdout)
    history = [initial]; best = deepcopy(policy.state_dict()); best_epoch = 0
    best_loss = initial['holdout_loss']; updates = 0
    announce(f"Practice initial: weighted pair-holdout={best_loss:.6f}; "
             f"ordinary-switch fit/holdout={split_counts['fit'].get('switch',0)}/{split_counts['holdout'].get('switch',0)}")
    mirror_size = max(1, min(len(mirror_fit), round(size*config['mirror_mass']), size-1))
    teacher_size = size-mirror_size
    for epoch in range(1, config['epochs']+1):
        policy.set_training_mode(True); indexes = rng.permutation(teacher_fit)
        for start in range(0, len(indexes), teacher_size):
            base._cancelled(cancel)
            # A separate mirror sample ensures exactly10% objective mass even in the final batch.
            part = np.concatenate([indexes[start:start+teacher_size], rng.choice(mirror_fit, mirror_size, replace=False)])
            logits, targets, bucket_ids = values(part)
            loss = weighted_imitation_loss(logits, targets[0], targets[1], targets[2], bucket_ids,
                weights=weights, mirror_mass=config['mirror_mass'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite weighted imitation loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([v for v in policy.parameters() if v.requires_grad], .5)
            if not torch.isfinite(norm): raise ValueError('Nonfinite weighted imitation gradient')
            optimizer.step(); updates += 1
        fit_metrics, holdout_metrics = measure(fit), measure(heldout)
        row = dict(epoch=epoch, fit_loss=fit_metrics['objective'], holdout_loss=holdout_metrics['weighted_teacher_loss'],
                   fit_metrics=fit_metrics, holdout_metrics=holdout_metrics); history.append(row)
        if row['holdout_loss'] < best_loss:
            best_loss, best_epoch = row['holdout_loss'], epoch; best = deepcopy(policy.state_dict())
        announce(f"Practice epoch {epoch}/{config['epochs']}: weighted holdout={row['holdout_loss']:.6f}; "
                 f"unweighted={holdout_metrics['unweighted_teacher_loss']:.6f}; best epoch={best_epoch}")
    policy.load_state_dict(best); policy.set_training_mode(False)
    if any(not torch.equal(value, base.critic_parameters(policy)[name]) for name, value in frozen.items()):
        raise ValueError('Practice changed frozen critic-only parameters')
    fit_pairs, heldout_pairs = {data.pairs[i] for i in fit}, {data.pairs[i] for i in heldout}
    if fit_pairs & heldout_pairs: raise ValueError('Practice pair split leaked')
    return dict(history=history, selected_epoch=best_epoch, selected_holdout_loss=best_loss,
        gradient_updates=updates, fit_rows=len(fit), holdout_rows=len(heldout), split_counts=split_counts,
        fit_pairs=len(fit_pairs), holdout_pairs=len(heldout_pairs),
        split='disjoint unordered training-team pairs across both trajectories; mirror fit-only',
        initial_metrics=dict(fit=initial_fit, holdout=initial_holdout),
        selected_metrics=dict(fit=history[best_epoch]['fit_metrics'],holdout=history[best_epoch]['holdout_metrics']),
        frozen_critic_parameters=frozen_names, critic_parameters_unchanged=True,
        shared_encoder_trained=True, value_predictions_may_change=True,
        weights=weights, mirror_objective_mass=config['mirror_mass'],
        selection='minimum weighted teacher loss on heldout training-team pairs; epoch0 retains ties; no game endpoint labels')


def verify(root, plan):
    root = Path(root).resolve(); request = _request(root, plan)
    path = root/'practice/report.json'
    if not path.is_file(): raise ValueError('Switch practice has not completed')
    report = json.loads(path.read_text())
    if report.get('status') != 'completed' or report.get('request') != request:
        raise ValueError('Completed switch-practice request differs')
    for relative, sha in report['artifacts'].items():
        path = base._inside(root, relative)
        if not path.is_file() or base._checksum(path) != sha:
            raise ValueError(f'Switch-practice artifact changed: {relative}')
    if set(OUTPUTS)-set(report['artifacts']):
        raise ValueError('Practice outputs missing from artifact manifest')
    return report


def prepare(root, plan, *, announce=print, cancel=None):
    """Prepare matched fresh-Adam PPO inputs; completed artifacts are hash-verified.

Interrupted attempts are preserved and collection restarts in a new numbered
attempt. No old run, old core module, source checkpoint, reward or architecture
is mutated. Raw reservoir arrays remain local and should be omitted from uploads.
"""
    from sb3_contrib import MaskablePPO
    from .shared_move_policy import policy_architecture
    root = Path(root).resolve(); cancel = Event() if cancel is None else cancel
    request = _request(root, plan); directory = root/'practice'; directory.mkdir(exist_ok=True)
    request_path = directory/'request.json'
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError('Partial switch-practice source/configuration changed; use a new run')
    if (directory/'report.json').exists(): return verify(root, plan)
    base._write(request_path, request); base._cancelled(cancel)
    number = 1+max([int(p.name.split('-')[-1]) for p in directory.glob('attempt-*')] or [0])
    attempt = directory/f'attempt-{number:04d}'; attempt.mkdir()
    project = Path(__file__).resolve().parents[2]; torch.set_num_threads(1)
    source = MaskablePPO.load(root/'inputs/start.zip', device='cpu')
    if (policy_architecture(source) != 'matchup-switches' or not source.policy.features_extractor.relational_enabled
            or not source.policy.features_extractor.bench_features_enabled):
        raise ValueError('Switch practice requires the selected feature-enabled policy')
    source.policy.set_training_mode(False)
    source_parameters, counters = base.parameter_digest(source.policy), ppo_counters(source)
    try:
        data, reservoir, collections = _collect(project, root, attempt, request, source, cancel, announce)
        learner = MaskablePPO.load(root/'inputs/start.zip', device='cpu')
        fit = _fit(learner, data, reservoir, request, cancel, announce)
        if base.parameter_digest(source.policy) != source_parameters:
            raise ValueError('Frozen current-parent policy changed')
        resets = {'ppo-only': base.reset_optimizer(source, request['ppo_learning_rate']),
                  'switch-practice': base.reset_optimizer(learner, request['ppo_learning_rate'])}
        if ppo_counters(source) != counters or ppo_counters(learner) != counters:
            raise ValueError('Supervised practice changed PPO counters')
        for name, model, selected in (('ppo-only',source,None),('switch-practice',learner,fit['selected_epoch'])):
            model.decision_weighted_warmstart = dict(version=VERSION,
                source_sha256=request['inputs']['inputs/start.zip'], source_parameters_sha256=source_parameters,
                selected_epoch=selected, optimizer_reset=True,
                scope='matched PPO-only control' if selected is None else 'fresh training-only weighted decision practice')
        staging = attempt/'outputs'; staging.mkdir()
        source.save(staging/'ppo-only.zip'); learner.save(staging/'switch-practice.zip')
        shutil.copyfile(staging/'switch-practice.zip', staging/'practice-only.zip')
        for name, expected in (('ppo-only',source),('switch-practice',learner)):
            restored = MaskablePPO.load(staging/(name+'.zip'), device='cpu')
            if (restored.policy.optimizer.state or ppo_counters(restored) != counters
                    or base.parameter_digest(restored.policy) != base.parameter_digest(expected.policy)):
                raise ValueError('Serialized practice output differs')
            del restored
        base._cancelled(cancel)
        for relative in OUTPUTS:
            destination = root/relative; destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                previous = attempt/'previous-partial-outputs'/relative; previous.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(destination, previous)
            shutil.copyfile(staging/destination.name, destination)
        data.flush()
        ledger = dict(rows=data.rows, capacity=data.capacity, counts=dict(data.counts),
            reservoir=reservoir.summary(), trajectory_counts=reservoir.trajectory_counts(),
            kept_by_group=dict(Counter(data.groups)), kept_by_bucket=dict(Counter(reservoir.buckets)),
            observation_shapes=data.schema, collection_limits=request['config'],
            files='disk-backed .npy arrays; only first rows entries are retained data',
            sampling='uniform Algorithm-R reservoir independently per decision bucket; current parent and Rules trajectories')
        base._write(attempt/'dataset.json',ledger)
        artifacts={relative:base._checksum(root/relative) for relative in OUTPUTS}
        for path in attempt.rglob('*'):
            if path.is_file() and ('dataset' in path.parts or path.name in
                    ('dataset.json','collection-suite.json','summary.json','episodes.jsonl')):
                artifacts[str(path.relative_to(root))]=base._checksum(path)
        report=dict(version=VERSION,status='completed',request=request,attempt=str(attempt.relative_to(root)),
            source_sha256=request['inputs']['inputs/start.zip'],source_parameters_sha256=source_parameters,
            source_timesteps=counters['num_timesteps'],ppo_counters=counters,ppo_counters_unchanged=True,
            dataset=ledger,collection=collections,fit=fit,optimizer_resets=resets,
            critic_note='Only critic-head parameters frozen; shared encoder training can change value predictions.',
            max_process_rss_mib=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024,
            artifacts=artifacts,outputs={Path(name).name:name for name in OUTPUTS},
            no_evaluation_team_labels=True,no_game_outcome_labels=True)
        base._write(directory/'report.json',report)
        announce(f"Switch practice complete: {data.rows} retained rows; best epoch {fit['selected_epoch']}; both PPO optimizers reset.")
        return report
    except BaseException as error:
        base._write(attempt/'failure.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            type=type(error).__name__,message=str(error),source_sha256=request['inputs']['inputs/start.zip']))
        raise
