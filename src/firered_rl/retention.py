"""Conservative, reference-aware retention for completed curriculum attempts.

``plan_campaign_retention`` is read-only and returns bounded per-directory previews.
``apply_campaign_retention`` re-plans before deleting and appends a durable audit row.
Callers already holding the curriculum lock pass ``assume_locked=True``; standalone
callers acquire the same lock automatically. No metadata or frozen pool is rewritten.
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

_PATTERN = {"stage": re.compile(r"stage-(\d+)"), "attempt": re.compile(r"attempt-(\d+)"),
            "pair": re.compile(r"pair-\d+"),
            "episode": re.compile(r"episode-\d+"),
            "eval": re.compile(r"(?:eval|before)-(?:current|expanded)-\d+")}
_HEX = re.compile(r"[0-9a-f]{64}")
_HISTORY = "retention-history.jsonl"
_PAIR_FILES = {"battle-readable.log", "battle.log", "showdown-stderr.log", "workers.json"}
_EPISODE_FILES = {"gym-steps.jsonl", "showdown-protocol.log", "teams.json", ".completed.json"}


def _json(path):
    if path.is_symlink():
        raise ValueError(f"refusing symlink metadata: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read retention metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"retention metadata must be an object: {path}")
    return value


def _safe(root, value):
    """Return a lexical in-root path only when none of its components is a link."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            return None
    return path


def _children(directory, kind):
    if not directory.is_dir() or directory.is_symlink():
        return []
    with os.scandir(directory) as entries:
        paths = [Path(entry.path) for entry in entries
                 if _PATTERN[kind].fullmatch(entry.name) and entry.is_dir(follow_symlinks=False)]
    return sorted(paths, key=lambda p: int(p.name.rsplit("-", 1)[-1]))


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stamp(path, root):
    info = path.lstat()
    return [str(path.relative_to(root)), info.st_dev, info.st_ino, info.st_mode,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _walk(path, *, debug=False, depth=0):
    """Streaming traversal; never follows links and memory is bounded by depth."""
    yield path
    if path.is_symlink() or not path.is_dir():
        return
    with os.scandir(path) as entries:
        for entry in entries:
            child = Path(entry.path)
            if entry.is_dir(follow_symlinks=False) and (not debug or
                    depth == 0 and _PATTERN["episode"].fullmatch(entry.name)):
                yield from _walk(child, debug=debug, depth=depth+1)
            else:
                yield child


def _debug_file(path, pair):
    parts = path.relative_to(pair).parts
    def known(name, names):
        if name in names:
            return True
        base, separator, suffix = name.rpartition(".")
        return bool(separator and suffix.isdigit() and base in names and base.endswith((".log", ".jsonl")))
    return ((len(parts) == 1 and known(parts[0], _PAIR_FILES)) or
            (len(parts) == 2 and _PATTERN["episode"].fullmatch(parts[0])
             and known(parts[1], _EPISODE_FILES)))


def _scan(path, root, protected):
    count = size = skipped_links = checksum = 0
    samples = []
    debug = path.is_dir()
    for child in _walk(path, debug=debug):
        record = _stamp(child, root)
        checksum = (checksum + int.from_bytes(hashlib.sha256(json.dumps(record).encode()).digest(),
                                             "big")) % (1 << 256)
        if stat.S_ISLNK(record[3]):
            skipped_links += 1
        elif (stat.S_ISREG(record[3]) and child not in protected
              and (not debug or _debug_file(child, path))):
            count += 1
            size += record[4]
            if len(samples) < 5:
                samples.append(record[0])
    return dict(files=count, bytes=size, fingerprint=f"{checksum:064x}",
                skipped_symlinks=skipped_links, sample=samples)


@contextmanager
def campaign_retention_lock(run_dir):
    """Use the runner's existing advisory lock; never wait on a live campaign."""
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir() or (root / ".lock").is_symlink():
        raise ValueError("campaign directory must exist and its lock must not be a symlink")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root / ".lock", flags, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"campaign is active; stop it before pruning: {root}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def plan_campaign_retention(run_dir, *, keep_attempts=2, keep_best=2, keep_debug_attempts=0):
    """Plan known debug/log/checkpoint deletions. ``-1`` retains all in that group.

    Latest candidate retention is per stage; best candidates are ranked within each
    stage/evaluation-mode/rule-revision cohort. The newest recoverable candidate,
    the newest unfinished attempt and all promoted models remain protected regardless
    of these counts. Full historical evaluation replay needs retained candidates;
    compact evaluation outcome records and frozen opponent copies always survive.
    """
    policy = dict(keep_attempts=keep_attempts, keep_best=keep_best,
                  keep_debug_attempts=keep_debug_attempts)
    if any(type(value) is not int or value < -1 for value in policy.values()):
        raise ValueError("retention counts must be -1 (all), zero, or a positive integer")
    root = Path(run_dir).expanduser().resolve()
    config = _json(root / "campaign.json")
    if (config.get("version") != 1 or not isinstance(config.get("input_hashes"), dict)
            or not config["input_hashes"] or not isinstance(config.get("curriculum"), dict)
            or not isinstance(config["curriculum"].get("stages"), list)
            or not config["curriculum"]["stages"]):
        raise ValueError("unrecognized campaign.json; refusing cleanup")
    protected, metadata, warnings = {}, [root / "campaign.json"], []

    def read_metadata(path):
        if _safe(root, str(path)) != path:
            raise ValueError(f"unsafe metadata path: {path}")
        return _json(path)

    def protect(value, reason, expected=None, required=True):
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing checkpoint path for {reason}")
        path = _safe(root, value)
        if path is None:
            warnings.append(f"External or symlink reference left untouched: {value}")
            return
        protected.setdefault(path, []).append(reason)
        if required and not path.is_file():
            raise ValueError(f"required retained file missing ({reason}): {path}")
        if expected is not None:
            if not isinstance(expected, str) or not _HEX.fullmatch(expected):
                raise ValueError(f"invalid saved checksum ({reason})")
            if required and _sha(path) != expected:
                raise ValueError(f"retained file checksum changed ({reason}): {path}")

    for relative, expected in config["input_hashes"].items():
        if not isinstance(relative, str) or _safe(root, relative) is None:
            raise ValueError("unsafe campaign input reference; refusing cleanup")
        protect(relative, "pinned campaign input", expected)
    custom = config.get("evaluation", {}).get("dumb_checkpoint")
    if custom:
        if not isinstance(custom, dict):
            raise ValueError("invalid Dumb checkpoint metadata")
        protect(custom.get("path"), "Dumb evaluation model", custom.get("sha256"))
    candidates, checkpoint_dirs, debug_roots, compact_logs = [], [], [], []
    evaluation_copies = []
    for stage in _children(root, "stage"):
        stage_number = int(stage.name.split("-")[-1])
        if not 1 <= stage_number <= len(config["curriculum"]["stages"]):
            raise ValueError(f"unexpected stage directory: {stage}")
        for name in ("stage.json", "result.json"):
            path = stage / name
            if not path.exists():
                continue
            saved = read_metadata(path)
            metadata.append(path)
            if name == "stage.json":
                protect(str(stage / "anchor.zip"), "stage anchor", saved.get("anchor_sha256"))
            elif saved.get("status") not in ("hold", "promote"):
                raise ValueError(f"invalid stage result: {path}")
            elif saved["status"] == "promote":
                protect(saved.get("candidate"), "promoted stage candidate", saved.get("candidate_sha256"))
            opponent = saved.get("opponent")
            if opponent:
                if not isinstance(opponent, dict):
                    raise ValueError(f"invalid stage opponent: {path}")
                protect(opponent.get("path"), "stage opponent", opponent.get("sha256"))
        attempts = []
        attempt_paths = _children(stage, "attempt")
        for attempt in attempt_paths:
            marker = attempt / "attempt.json"
            info = read_metadata(marker)
            metadata.append(marker)
            if not isinstance(info.get("resume"), str):
                raise ValueError(f"attempt has no resume path: {marker}")
            report_path = attempt / "report.json"
            summary_path = attempt / "train/summary.json"
            summary = read_metadata(summary_path) if summary_path.exists() else {}
            if summary:
                metadata.append(summary_path)
            report = read_metadata(report_path) if report_path.exists() else None
            candidate = None
            score = None
            if report is not None:
                metadata.append(report_path)
                if report.get("status") not in ("hold", "promote") or report.get("stage") != stage_number:
                    raise ValueError(f"invalid attempt report: {report_path}")
                value = report.get("candidate")
                if not isinstance(value, str) or not isinstance(report.get("candidate_sha256"), str):
                    raise ValueError(f"invalid candidate metadata: {report_path}")
                if not _HEX.fullmatch(report["candidate_sha256"]):
                    raise ValueError(f"invalid candidate checksum: {report_path}")
                candidate = _safe(root, value)
                current = report.get("current", {}).get("all", {})
                score = report.get("win_rate", current.get("win_rate"))
                if score is None and current.get("attempts"):
                    score = current.get("wins", 0) / current["attempts"]
                if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"invalid evaluation score: {report_path}")
                if report["status"] == "promote":
                    protect(value, "promoted attempt candidate", report["candidate_sha256"])
            unfinished = report is None
            entry = dict(path=attempt, candidate=candidate, score=score, report=report,
                         unfinished=unfinished, summary=summary, info=info)
            attempts.append(entry)
            if unfinished and attempt == attempt_paths[-1]:
                for name in ("latest.zip", "final.zip"):
                    path = attempt / "train/checkpoints" / name
                    if _safe(root, str(path)) == path and path.is_file():
                        protect(str(path), "unfinished attempt checkpoint")
                protect(info["resume"], "unfinished attempt fallback resume", info.get("resume_sha256"))
        existing = [a for a in attempts if a["candidate"] is not None and a["candidate"].is_file()]
        retained = existing if keep_attempts == -1 else (existing[-keep_attempts:] if keep_attempts else [])
        cohorts = {}
        for item in existing:
            report = item["report"]
            key = (report.get("mode", "original"), report.get("settings_revision", 0))
            if not isinstance(key[0], str) or type(key[1]) is not int:
                raise ValueError("invalid evaluation cohort metadata")
            cohorts.setdefault(key, []).append(item)
        for group in cohorts.values():
            ranked = sorted(group, key=lambda a: (a["score"], a["path"].name), reverse=True)
            retained += ranked if keep_best == -1 else ranked[:keep_best]
        if existing:
            retained.append(existing[-1])
        # The runner looks at the numerically newest attempt, not directory mtime.
        if attempts and attempts[-1]["report"]:
            newest = attempts[-1]
            protect(newest["report"]["candidate"], "latest attempt resume", newest["report"]["candidate_sha256"])
        for item in retained:
            protect(str(item["candidate"]), "retained latest/best candidate", item["report"]["candidate_sha256"])
        debug_keep = {a["path"] for a in (attempts if keep_debug_attempts == -1 else
                      attempts[-keep_debug_attempts:] if keep_debug_attempts else [])}
        if attempts and attempts[-1]["unfinished"]:
            debug_keep.add(attempts[-1]["path"])
        for item in attempts:
            attempt = item["path"]
            checkpoint = attempt / "train/checkpoints"
            if _safe(root, str(checkpoint)) == checkpoint and checkpoint.is_dir():
                checkpoint_dirs.append(checkpoint)
            # Evaluations preserve candidate bytes independently of final.zip.
            # Completed campaign cleanup must also account for these copies.
            if not item["unfinished"]:
                for evaluation in _children(attempt, "eval"):
                    snapshot = evaluation / "candidate.zip"
                    provenance = evaluation / "config.json"
                    if _safe(root, str(snapshot)) != snapshot or not snapshot.is_file() or not provenance.is_file():
                        continue
                    info = read_metadata(provenance).get("models", {}).get("candidate", {})
                    if not isinstance(info.get("snapshot_path"), str) or _safe(root, info["snapshot_path"]) != snapshot:
                        continue
                    if _sha(snapshot) != info.get("sha256"):
                        raise ValueError(f"evaluation candidate snapshot hash mismatch: {snapshot}")
                    metadata.append(provenance)
                    evaluation_copies.append(snapshot)
            if attempt not in debug_keep:
                debug_roots += _children(attempt / "train", "pair")
                for evaluation in _children(attempt, "eval"):
                    debug_roots += _children(evaluation, "pair")
                compact_logs += [attempt / "train" / name for name in
                                 ("episodes.jsonl", "updates.jsonl", "opponent-switches.jsonl")]
            candidates.append(dict(attempt=str(attempt.relative_to(root)),
                candidate=str(item["candidate"].relative_to(root)) if item["candidate"] else None,
                retained=item["candidate"] in protected if item["candidate"] else False,
                debug_retained=attempt in debug_keep, unfinished=item["unfinished"]))
        # Baseline evaluations are finished before any completed attempt exists.
        if attempts and not attempts[-1]["unfinished"] and keep_debug_attempts != -1:
            for evaluation in _children(stage, "eval"):
                debug_roots += _children(evaluation, "pair")
    actions = []
    for path in evaluation_copies:
        if path not in protected:
            actions.append(dict(path=str(path.relative_to(root)), category="checkpoints", kind="file"))
    for directory in checkpoint_dirs:
        with os.scandir(directory) as entries:
            for entry in entries:
                if (not entry.is_file(follow_symlinks=False) or
                        not re.fullmatch(r"latest\.zip|final\.zip|ppo_\d+\.zip", entry.name)):
                    continue
                path = Path(entry.path)
                if path not in protected:
                    actions.append(dict(path=str(path.relative_to(root)), category="checkpoints", kind="file"))
    for path in compact_logs:
        if _safe(root, str(path)) == path and path.is_file() and path not in protected:
            actions.append(dict(path=str(path.relative_to(root)), category="training_details", kind="file"))
    for path in debug_roots:
        if _safe(root, str(path)) == path:
            actions.append(dict(path=str(path.relative_to(root)), category="battle_debug", kind="directory"))
    totals = {key: dict(files=0, bytes=0) for key in ("checkpoints", "training_details", "battle_debug")}
    for action in actions:
        action.update(_scan(root / action["path"], root, protected))
        for key in ("files", "bytes"):
            totals[action["category"]][key] += action[key]
    signatures = {str(path.relative_to(root)): _sha(path) for path in metadata}
    signature = hashlib.sha256(json.dumps(dict(policy=policy, metadata=signatures,
        protected={str(p.relative_to(root)): _stamp(p, root) for p in protected if p.exists()},
        actions=[{key: a[key] for key in ("path", "category", "fingerprint", "files", "bytes")}
                 for a in sorted(actions, key=lambda a: a["path"])]), sort_keys=True).encode()).hexdigest()
    return dict(version=1, run_dir=str(root), policy=policy, fingerprint=signature,
                totals=totals, files=sum(v["files"] for v in totals.values()),
                bytes=sum(v["bytes"] for v in totals.values()), actions=actions,
                protected={str(p.relative_to(root)): why for p, why in protected.items()},
                attempts=candidates, warnings=sorted(set(warnings)))


def _remove_empty(path):
    if path.is_symlink() or not path.is_dir():
        return
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False) and _PATTERN["episode"].fullmatch(entry.name):
                try:
                    Path(entry.path).rmdir()
                except OSError:
                    pass
    try:
        path.rmdir()
    except OSError:
        pass  # Preserved links/protected files leave their containing directory intact.


def apply_campaign_retention(plan, *, assume_locked=False):
    """Apply a fresh matching plan. Acquires the campaign lock unless already held."""
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("invalid retention plan")
    root = Path(plan["run_dir"])
    with nullcontext() if assume_locked else campaign_retention_lock(root):
        current = plan_campaign_retention(root, **plan["policy"])
        if current["fingerprint"] != plan.get("fingerprint"):
            raise ValueError("campaign files changed since the retention preview; create a new plan")
        protected = {root / relative for relative in current["protected"]}
        touched = {str(Path(*Path(a["path"]).parts[:2]))
                   for a in current["actions"] if a["files"] and len(Path(a["path"]).parts) >= 3
                   and Path(a["path"]).parts[1].startswith("attempt-")}
        result = dict(time=datetime.now(timezone.utc).isoformat(), policy=current["policy"],
                      plan_fingerprint=current["fingerprint"], status="completed", files=0, bytes=0,
                      actions=[], attempts=[a for a in current["attempts"] if a["attempt"] in touched])
        if not current["files"]:
            return result
        history = root / _HISTORY
        if history.is_symlink():
            raise ValueError("retention history must not be a symlink")
        try:
            for action in current["actions"]:
                if not action["files"]:
                    continue
                path = _safe(root, action["path"])
                if path is None:
                    raise ValueError(f"unsafe path appeared during retention: {action['path']}")
                now = _scan(path, root, protected)
                if now["fingerprint"] != action["fingerprint"]:
                    raise ValueError(f"file changed during retention: {action['path']}")
                removed = dict(path=action["path"], category=action["category"], files=0, bytes=0)
                result["actions"].append(removed)
                for child in _walk(path, debug=action["kind"] == "directory"):
                    if child in protected or (action["kind"] == "directory" and not _debug_file(child, path)):
                        continue
                    info = child.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    child.unlink()
                    removed["files"] += 1
                    removed["bytes"] += info.st_size
                    result["files"] += 1
                    result["bytes"] += info.st_size
                if action["kind"] == "directory":
                    _remove_empty(path)
        except Exception as error:
            result.update(status="failed", error=str(error))
            raise
        finally:
            flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(history, flags, 0o600), "a") as stream:
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return result
