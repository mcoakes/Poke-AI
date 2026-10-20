"""Shuffle only the learner's occupied bench after canonical team sampling.

This does not expand the sampling pool. Per-pair permutation RNGs are separate
from TrainingTeams RNGs; fixed-pair rehearsal groups retain their exact order.
"""
from collections import Counter
import copy
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
from threading import RLock

from .periodic_evaluation import atomic_write


VERSION = 1
COUNTERS = ('episode_selections', 'eligible', 'skipped_fixed_pair', 'identity', 'changed')


def validate_options(args):
    if getattr(args, 'shuffle_bench_order', False) and (
            args.backend != 'showdown' or not args.rule_based_opponent
            or getattr(args, 'opponent_pool', None) or getattr(args, 'opponent_checkpoint', None)):
        raise ValueError('--shuffle-bench-order requires --backend showdown and --rule-based-opponent only')


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


class BenchOrderTeams:
    """Transparent canonical source with candidate-P1 bench augmentation.

    Construct after mixture snapshotting and continuation reseeding. Source
    files/descriptions and sampling counts are forwarded without modification.
    Summary counters describe this invocation's episode starts, including starts
    that have not finished; continuation deliberately starts fresh battle RNGs.
    """
    VERSION = VERSION

    def __init__(self, source, directory, seed):
        self.source = source
        if Path(directory).is_symlink():
            raise ValueError('bench variant directory cannot be a symlink')
        self.directory = Path(directory).resolve()
        self.shuffle_seed = int(seed)
        self.lock = RLock()
        self.counts = Counter(dict.fromkeys(COUNTERS, 0))
        self.per_pair = {}
        self.canonical = {str(Path(path).resolve()): checksum
                          for path, checksum in source.hashes.items()}
        self.fixed_files = set()
        if getattr(source, 'mixture', None) is not None:
            varying = set()
            for group in source.mixture_groups.values():
                selected = self.fixed_files if group['fixed'] else varying
                selected.update(str(Path(path).resolve()) for path in group['files'])
            if self.fixed_files & varying:
                raise ValueError('bench shuffle needs distinct frozen paths for fixed and sampled groups; snapshot the mixture first')
        elif source.fixed:
            self.fixed_files = set(self.canonical)
        self.documents = {}
        for name in self.canonical:
            self.documents[name] = self._read_canonical(name)
        self.maximum_variants = sum(len(list(permutations(range(1, len(team['party']))))) - 1
            for name, team in self.documents.items() if name not in self.fixed_files)
        self.manifest_path = self.directory / 'manifest.json'
        if self.directory.is_symlink():
            raise ValueError('bench variant directory cannot be a symlink')
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            if self.manifest_path.is_symlink():
                raise ValueError('bench variant manifest cannot be a symlink')
            self.manifest = json.loads(self.manifest_path.read_text())
            if (self.manifest.get('version') != VERSION
                    or self.manifest.get('sources') != self.canonical
                    or self.manifest.get('fixed_pair_sources') != sorted(self.fixed_files)):
                raise ValueError('bench variant source identity changed')
            self._verify_manifest()
        else:
            if any(self.directory.iterdir()):
                raise ValueError('bench variant directory has no recognized manifest')
            self.manifest = dict(version=VERSION, sources=self.canonical,
                fixed_pair_sources=sorted(self.fixed_files), variants={})
            self._save_manifest()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def _read_canonical(self, name):
        path = Path(name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'canonical bench team is not a regular file: {path}')
        raw = path.read_bytes()
        if _hash(raw) != self.canonical[name]:
            raise ValueError(f'canonical team changed during bench shuffle: {path}')
        team = json.loads(raw)
        if not isinstance(team.get('party'), list) or not 1 <= len(team['party']) <= 6:
            raise ValueError('bench shuffle requires one to six occupied party entries')
        return team

    def _variant(self, source, order):
        team = copy.deepcopy(self.documents[source])
        count = len(team['party'])
        if (len(order) != count or order[0] != 0 or sorted(order) != list(range(count))):
            raise ValueError('invalid bench permutation; lead must stay first')
        team['party'] = [team['party'][index] for index in order]
        raw = (json.dumps(team, indent=2) + '\n').encode()
        name = _hash(source.encode())[:12] + '-' + self.canonical[source] + '-' + ''.join(map(str, order)) + '.json'
        return name, raw, dict(source=source, source_sha256=self.canonical[source],
                              permutation=list(order), sha256=_hash(raw))

    def _save_manifest(self):
        atomic_write(self.manifest_path, json.dumps(self.manifest, indent=2) + '\n')

    def _verify_manifest(self):
        variants = self.manifest.get('variants')
        if not isinstance(variants, dict) or len(variants) > self.maximum_variants:
            raise ValueError('invalid or unbounded bench variant manifest')
        expected_files = {'manifest.json', *variants}
        if any(path.is_symlink() or not path.is_file() or path.name not in expected_files
               for path in self.directory.iterdir()):
            raise ValueError('unexpected file in bench variant directory')
        for name, entry in variants.items():
            source = entry.get('source')
            if source not in self.canonical or source in self.fixed_files:
                raise ValueError('invalid canonical source in bench variant manifest')
            expected_name, raw, expected = self._variant(source, entry.get('permutation', []))
            if name != expected_name or entry != expected:
                raise ValueError('bench variant manifest content changed')
            path = self.directory / name
            if path.exists():
                if path.is_symlink() or _hash(path.read_bytes()) != expected['sha256']:
                    raise ValueError(f'frozen bench variant changed: {path}')
            else:
                # Manifest is committed first. A crash before its variant write
                # is recoverable from the unchanged canonical input and order.
                atomic_write(path, raw)

    def _materialize(self, source, order):
        name, raw, entry = self._variant(source, order)
        path = self.directory / name
        previous = self.manifest['variants'].get(name)
        if previous is not None:
            if previous != entry or path.is_symlink() or not path.is_file() or _hash(path.read_bytes()) != entry['sha256']:
                raise ValueError(f'frozen bench variant changed: {path}')
        else:
            if path.exists():
                raise ValueError(f'unrecorded bench variant exists: {path}')
            self.manifest['variants'][name] = entry
            if len(self.manifest['variants']) > self.maximum_variants:
                raise ValueError('bench variant count exceeds possible canonical permutations')
            self._save_manifest()
            atomic_write(path, raw)
        return path

    def for_pair(self, index):
        canonical_choose = self.source.for_pair(index)
        rng = random.Random(self.shuffle_seed + int(index) * 1000003 + 0xB3C401)

        def choose():
            first, second = canonical_choose()  # Original RNG and group draw come first.
            name = str(Path(first).resolve())
            with self.lock:
                if name not in self.canonical:
                    raise ValueError('canonical sampler returned an unknown team')
                self._read_canonical(name)
                other_name = str(Path(second).resolve())
                if other_name not in self.canonical:
                    raise ValueError('canonical sampler returned an unknown opponent team')
                self._read_canonical(other_name)
                extra = Counter(episode_selections=1)
                if name in self.fixed_files:
                    extra['skipped_fixed_pair'] += 1
                    selected = first
                else:
                    extra['eligible'] += 1
                    order = list(range(1, len(self.documents[name]['party'])))
                    rng.shuffle(order)
                    order.insert(0, 0)
                    identity = order == list(range(len(order)))
                    extra['identity' if identity else 'changed'] += 1
                    selected = first if identity else self._materialize(name, order)
                self.counts.update(extra)
                self.per_pair.setdefault(str(index), Counter(dict.fromkeys(COUNTERS, 0))).update(extra)
                return selected, second
        return choose

    def configuration(self):
        return dict(version=VERSION, seed=self.shuffle_seed,
            scope='candidate_player_1_nonfixed_groups', lead='unchanged', opponent='unchanged',
            moves='unchanged', rehearsal='fixed-pair mixture groups are excluded',
            canonical_sampling='sample canonical pair first; independent per-pair permutation RNG',
            validation='unchanged frozen suites', manifest=str(self.manifest_path))

    def get_summary(self):
        with self.lock:
            return dict(**self.configuration(), counts=dict(self.counts),
                per_pair={key: dict(value) for key, value in self.per_pair.items()},
                stored_variants=len(self.manifest['variants']), maximum_variants=self.maximum_variants,
                counting='episode starts including unfinished episodes; current invocation only')
