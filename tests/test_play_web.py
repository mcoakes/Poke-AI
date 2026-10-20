import hashlib
import http.client
import json
from dataclasses import replace
from pathlib import Path
import sys
import threading
import time

import pytest

from firered_rl.play_web import MatchJobs, PlayServer, PlaySettings, TeamCatalog

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalog():
    return TeamCatalog(ROOT)


@pytest.fixture
def jobs(tmp_path, catalog):
    checkpoint = tmp_path / "latest.zip"
    checkpoint.write_bytes(b"fixed policy snapshot")
    settings = PlaySettings(project=tmp_path, rom=tmp_path / "FireRed.gba",
                            source=tmp_path / "FireRed.sav", emulator=tmp_path / "mgba-qt",
                            checkpoint=checkpoint, workers=8)
    manager = MatchJobs(settings, catalog)
    manager.models()
    yield manager
    manager.shutdown()


def request_payload(jobs):
    return dict(checkpoint=next(iter(jobs.checkpoints)), workers=2, speed="1",
                teams=[jobs.catalog.sample(category, 3)["team"] for category in ("competitive", "casual")])


def wait_job(jobs):
    jobs.thread.join(timeout=4)
    assert not jobs.thread.is_alive()
    return jobs.snapshot()


def test_catalog_uses_national_sprite_ids_and_independent_profile_sizes(catalog):
    species = {s["name"]: s for s in catalog.public()["species"]}
    assert species["Treecko"]["sprite"] == "/sprites/252.png"
    assert {s["dex"] for s in species.values()} == set(range(1, 387))
    for category in ("competitive", "casual", "random", "mixed"):
        for size in range(1, 7):
            result = catalog.sample(category, size)
            assert len(result["team"]["party"]) == size
            assert all(len(mon["moves"]) == 4 for mon in result["team"]["party"])
            assert result["category"] == category or category == "mixed"


def test_validate_rejects_bad_catalog_values_and_ev_budget(catalog):
    team = catalog.sample("competitive", 1)["team"]
    team["party"][0]["evs"] = {"hp": 255, "atk": 255, "def": 1}
    with pytest.raises(ValueError, match="510"):
        catalog.validate(team)
    team["party"][0]["evs"] = {}
    team["party"][0]["moves"][0] = "Invented move"
    with pytest.raises(ValueError, match="unknown Gen III move"):
        catalog.validate(team)
    team["party"][0]["moves"] = ["Tackle"]
    assert catalog.validate(team)["party"][0]["moves"] == ["Tackle"]
    team["party"][0]["moves"] = []
    with pytest.raises(ValueError, match="one and four"):
        catalog.validate(team)


def test_saved_teams_cover_every_bundled_party_including_legacy_templates(catalog):
    groups = catalog.available_teams()
    indexed = {team["id"] for group in groups for team in group["teams"]}
    expected = {path.relative_to(ROOT / "config").as_posix()
                for path in (ROOT / "config").rglob("*.json")
                if "party" in json.loads(path.read_text())}
    assert indexed == expected
    assert "training_profiles/six_types/teams/six_types.json" in indexed
    fire = catalog.select("teams/fire_vs_water.json")["team"]
    water = catalog.select("teams/water_vs_fire.json")["team"]
    assert fire["party"][1]["moves"][-1] == "Take Down"
    assert water["party"][0]["moves"][1] == "Ice Beam"
    caterpie = catalog.select("teams/mixed_example/mixed_014.json")["team"]["party"][-1]
    assert caterpie["moves"] == ["Tackle", "String Shot"]


def test_team_discovery_groups_paths_refreshes_and_rejects_outside_files(catalog, tmp_path):
    team = {"name": "Same name", "party": [dict(species="Pikachu", level=50,
             moves=["Thunderbolt"], held_item="LightBall", evs={"spe": 252}, ivs={"atk": 0})],
             "generation": {"seed": 123}}
    root = tmp_path / "config"
    for relative in ("team.json", "misc/team.json", "custom/nested/team.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(team))
    (root / "rewards.json").write_text('{"win": 5000}')
    (root / "bad.json").write_text('not JSON')
    (root / "bad_team.json").write_text('{"party": [{}]}')
    (root / "array.json").write_text('[]')
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(team))
    (root / "outside.json").symlink_to(outside)
    catalog.project = tmp_path
    groups = catalog.available_teams()
    assert {g["id"]: g["label"] for g in groups} == {
        ".": "Misc", "custom/nested": "custom/nested", "misc": "misc"}
    assert sum(len(g["teams"]) for g in groups) == 3
    original = (root / "team.json").read_bytes()
    selected = catalog.select("team.json")["team"]
    assert selected["party"][0]["held_item"] == "Light Ball"
    assert selected["party"][0]["ivs"]["atk"] == 0
    selected["party"][0]["moves"].append("Surf")
    assert catalog.select("team.json")["team"]["party"][0]["moves"] == ["Thunderbolt"]
    assert (root / "team.json").read_bytes() == original
    for key in ("../outside.json", "outside.json", str(outside), {}, None):
        with pytest.raises(ValueError, match="select an available team"):
            catalog.select(key)
    (root / "added.json").write_text(json.dumps(team))
    assert sum(len(g["teams"]) for g in catalog.available_teams()) == 4
    (root / "team.json").unlink()
    (root / "team.json").symlink_to(outside)
    with pytest.raises(ValueError, match="inside config"):
        catalog.select("team.json")
    (root / "added.json").write_text('{"party": []}')
    with pytest.raises(ValueError, match="unavailable"):
        catalog.select("added.json")


def test_models_always_include_named_models_and_prefer_smart_v10(jobs, tmp_path):
    root = tmp_path / "models"
    (root / "campaign").mkdir(parents=True)
    for name in ("smart_v10.zip", "smart_v11.zip", "campaign/smart_v10.zip", "campaign/dumb.zip"):
        (root / name).write_bytes(b"model")
    recent = tmp_path / "runs"
    recent.mkdir()
    for index in range(205):
        (recent / f"checkpoint_{index}.zip").write_bytes(b"recent")
    (recent / "latest.zip").write_bytes(b"newest")
    jobs.settings = replace(jobs.settings, checkpoint=None, checkpoint_dir=recent)
    rows = jobs.models()
    assert rows[0]["label"] == "models/smart_v10.zip"
    assert len([row for row in rows if row["label"].startswith("models/")]) == 4
    assert len([row for row in rows if row["label"].startswith("runs/")]) == 200
    jobs.settings = replace(jobs.settings, checkpoint_dir=root)
    rows = jobs.models()
    assert len(rows) == len({row["id"] for row in rows}) == 4
    jobs.settings = replace(jobs.settings, checkpoint=root / "smart_v11.zip")
    assert jobs.models()[0]["label"] == "models/smart_v11.zip"
    jobs.settings = replace(jobs.settings, checkpoint=Path("latest"), checkpoint_dir=recent)
    assert jobs.models()[0]["id"] == "latest"


@pytest.mark.parametrize("filename", ["smartv10.zip", "SmartV10.zip", "Smart-v10.zip"])
def test_smart_v10_filename_variants_and_fallback(jobs, tmp_path, filename):
    root = tmp_path / "models"
    root.mkdir()
    (root / "dumb.zip").write_bytes(b"model")
    preferred = root / filename
    preferred.write_bytes(b"model")
    jobs.settings = replace(jobs.settings, checkpoint=None)
    assert jobs.models()[0]["label"] == f"models/{filename}"
    preferred.unlink()
    assert jobs.models()[0]["label"] == "models/dumb.zip"


@pytest.mark.parametrize("change", [dict(workers=3), dict(workers=0), dict(checkpoint="/etc/passwd"),
                                  dict(speed="nan"), dict(speed="bad"), dict(stochastic="yes"),
                                  dict(player_name="TooLongName"), dict(player_name="Matt💥"),
                                  dict(player_name=123)])
def test_invalid_launch_settings_do_not_start_tools(jobs, change):
    payload = request_payload(jobs)
    payload.update(change)
    with pytest.raises(ValueError):
        jobs.start(payload)
    assert jobs.thread is None


@pytest.mark.parametrize("player_name", [None, " Matt ", ""])
def test_generation_verifies_both_states_then_launches_exact_teams_and_checkpoint(jobs, monkeypatch, player_name):
    calls = []
    payload = request_payload(jobs)
    jobs.catalog.available_teams()
    payload["teams"] = [jobs.catalog.select(path)["team"] for path in
                        ("teams/fire_vs_water.json", "teams/mixed_example/mixed_014.json")]
    if player_name is not None:
        payload["player_name"] = player_name

    def run(command, log):
        calls.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        if "build_states.py" in command[2]:
            assert command[command.index("--workers") + 1] == "2"
            inputs = command[command.index("--teams") + 1:command.index("--workers")]
            for i, file in enumerate(inputs):
                expected = dict(payload["teams"][i])
                if i == 0 and player_name:
                    expected["trainer_name"] = player_name.strip()
                assert json.loads(Path(file).read_text()) == expected
                state = output / "counter" / f"player{i + 1}.ss1"
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_bytes(f"verified state {i}".encode())
                state.with_suffix(".ss1.json").write_text(json.dumps(dict(ok=True, state_sha256=hashlib.sha256(state.read_bytes()).hexdigest())))
            jobs.settings.checkpoint.write_bytes(b"new policy")
        else:
            agent = Path(command[command.index("--checkpoint") + 1])
            assert agent.read_bytes() == b"fixed policy snapshot"
            assert "--no-wait" in command and command[command.index("--human-player") + 1] == "1"
            assert command[command.index("--setup-speed") + 1] == "unlimited"
            assert command[command.index("--speed") + 1] == "1.0"
            output.mkdir()
            (output / "result.json").write_text(json.dumps(dict(human_outcome="won", decisions=5, truncated=False)))
    monkeypatch.setattr(jobs, "_run", run)
    jobs.start(payload)
    result = wait_job(jobs)
    assert result["status"] == "complete" and result["states_ready"] == 2
    assert len(calls) == 2
    assert (Path(result["directory"]) / "job.json").is_file()


@pytest.mark.parametrize("failure", ["generation", "unverified_state"])
def test_failed_generation_never_launches_a_battle(jobs, monkeypatch, failure):
    calls = []
    def run(command, log):
        calls.append(command)
        if failure == "generation":
            raise RuntimeError("capture failed at counter")
        output = Path(command[command.index("--output-dir") + 1]) / "counter"
        output.mkdir(parents=True)
        (output / "player1.ss1").write_bytes(b"state")
        (output / "player1.ss1.json").write_text('{"ok": false}')
    monkeypatch.setattr(jobs, "_run", run)
    jobs.start(request_payload(jobs))
    assert wait_job(jobs)["status"] == "failed"
    assert len(calls) == 1


def test_cancel_reaps_real_child_and_does_not_launch_battle(jobs, monkeypatch):
    real_run = jobs._run
    launched = threading.Event()
    def run(command, log):
        launched.set()
        return real_run([sys.executable, "-u", "-c", "import time; time.sleep(30)"], log)
    monkeypatch.setattr(jobs, "_run", run)
    jobs.start(request_payload(jobs))
    assert launched.wait(2)
    deadline = time.monotonic() + 2
    while jobs.process is None and time.monotonic() < deadline:
        time.sleep(.01)
    child = jobs.process
    assert child is not None
    with pytest.raises(RuntimeError, match="already running"):
        jobs.start(request_payload(jobs))
    jobs.cancel()
    assert wait_job(jobs)["status"] == "cancelled"
    assert child.poll() is not None


def test_http_origin_token_and_static_path_boundaries(jobs, catalog):
    server = PlayServer(0, jobs, catalog)
    server.assets = ROOT / "web/play"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request(method, path, None if payload is None else json.dumps(payload), headers or {})
        response = connection.getresponse()
        status, body = response.status, response.read()
        connection.close()
        return status, body
    try:
        status, body = call("GET", "/api/catalog")
        assert status == 200 and len(json.loads(body)["catalog"]["species"]) == 386
        assert call("GET", "/sprites/252.png")[1].startswith(b"\x89PNG")
        assert call("GET", "/sprites/../../mgba/play_agent.py")[0] == 404
        assert call("GET", "/api/catalog", headers={"Host": "example.com"})[0] == 403
        payload = dict(category="competitive", size=2)
        assert call("POST", "/api/team", payload)[0] == 403
        headers = {"Content-Type": "application/json", "X-Play-Token": server.token, "Origin": "https://unrelated.example"}
        assert call("POST", "/api/team", payload, headers)[0] == 403
        headers["Origin"] = f"http://127.0.0.1:{server.server_port}"
        status, body = call("POST", "/api/team", payload, headers)
        assert status == 200 and len(json.loads(body)["team"]["party"]) == 2
        status, body = call("GET", "/api/teams")
        assert status == 200 and json.loads(body)["categories"]
        status, body = call("POST", "/api/team", {"id": "teams/water_vs_fire.json"}, headers)
        assert status == 200 and len(json.loads(body)["team"]["party"]) == 6
        assert call("POST", "/api/team", {"id": "../../outside.json"}, headers)[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_every_species_has_a_suggested_set():
    from firered_rl.play_web import TeamCatalog
    from pathlib import Path
    catalog = TeamCatalog(Path(__file__).resolve().parents[1])
    assert len(catalog.presets) == len(catalog.data['species']) == 386
    for name, preset in catalog.presets.items():
        assert len(set(preset['moves'])) == 4
        assert set(preset['moves']) <= set(catalog.data['moves'])
        assert sum(preset['evs'].values()) <= 510
        if preset.get('source') != 'experimental':
            assert set(preset['moves']) <= set(catalog.data['species'][name]['moves'])


def classroom_manifest(project):
    directory = project / 'models/classroom'
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('coached-ai.zip', 'rewards-only-ai.zip'):
        (directory / name).write_bytes(name.encode())
    manifest = project / 'config/classroom-models.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    data = dict(version=1, default='coached', models=[
        dict(id='rewards-only', label='Rewards-only AI', checkpoint='models/classroom/rewards-only-ai.zip'),
        dict(id='coached', label='Coached AI (default)', checkpoint='models/classroom/coached-ai.zip')])
    manifest.write_text(json.dumps(data))
    return manifest, data


def test_classroom_model_list_replaces_discovery_and_orders_default_first(jobs, tmp_path):
    manifest, data = classroom_manifest(tmp_path)
    (tmp_path / 'models/smart_v10.zip').write_bytes(b'other model')
    recent = tmp_path / 'recent/checkpoints'
    recent.mkdir(parents=True)
    (recent / 'latest.zip').write_bytes(b'recent training model')
    settings = replace(jobs.settings, checkpoint=None, checkpoint_dir=recent.parent,
                       model_list=manifest.relative_to(tmp_path))
    classroom = MatchJobs(settings, jobs.catalog)
    assert classroom.models() == [dict(id='coached', label='Coached AI (default)'),
                                  dict(id='rewards-only', label='Rewards-only AI')]
    assert classroom.checkpoints == {row['id']: (tmp_path / row['checkpoint']).resolve()
                                     for row in data['models']}
    assert 'latest' not in classroom.checkpoints
    # The configured directory is irrelevant, including after a refresh.
    (recent / 'ppo_999999.zip').write_bytes(b'new training model')
    assert [row['id'] for row in classroom.models()] == ['coached', 'rewards-only']


@pytest.mark.parametrize('change', ['root', 'unknown_root', 'version', 'boolean_version', 'no_default',
    'unknown_default', 'empty_models', 'models_object', 'unknown_entry', 'missing_entry', 'empty_id',
    'empty_label', 'bad_path_type', 'duplicate_id', 'missing_file', 'directory', 'latest', 'not_zip'])
def test_classroom_model_list_rejects_invalid_configuration_on_startup(jobs, tmp_path, change):
    manifest, data = classroom_manifest(tmp_path)
    if change == 'root': data = []
    elif change == 'unknown_root': data['extra'] = True
    elif change == 'version': data['version'] = 2
    elif change == 'boolean_version': data['version'] = True
    elif change == 'no_default': data.pop('default')
    elif change == 'unknown_default': data['default'] = 'not-listed'
    elif change == 'empty_models': data['models'] = []
    elif change == 'models_object': data['models'] = {}
    elif change == 'unknown_entry': data['models'][0]['extra'] = True
    elif change == 'missing_entry': data['models'][0].pop('label')
    elif change == 'empty_id': data['models'][0]['id'] = '  '
    elif change == 'empty_label': data['models'][0]['label'] = ''
    elif change == 'bad_path_type': data['models'][0]['checkpoint'] = None
    elif change == 'duplicate_id': data['models'][1]['id'] = data['models'][0]['id']
    elif change == 'missing_file': data['models'][0]['checkpoint'] = 'models/classroom/missing.zip'
    elif change == 'directory':
        (tmp_path / 'directory.zip').mkdir()
        data['models'][0]['checkpoint'] = 'directory.zip'
    elif change == 'latest': data['models'][0]['checkpoint'] = 'latest'
    else:
        (tmp_path / 'not-a-model.txt').write_text('model')
        data['models'][0]['checkpoint'] = 'not-a-model.txt'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        MatchJobs(replace(jobs.settings, checkpoint=None, model_list=manifest), jobs.catalog)


@pytest.mark.parametrize('contents', [None, '{invalid JSON'])
def test_classroom_model_list_missing_or_malformed_file_has_helpful_error(jobs, tmp_path, contents):
    manifest = tmp_path / 'model-list.json'
    if contents is not None: manifest.write_text(contents)
    with pytest.raises(ValueError, match='cannot read --model-list'):
        MatchJobs(replace(jobs.settings, checkpoint=None, model_list=manifest), jobs.catalog)


def test_classroom_model_list_conflicts_with_explicit_checkpoint(jobs, tmp_path):
    manifest, _ = classroom_manifest(tmp_path)
    with pytest.raises(ValueError, match='cannot be combined with --checkpoint'):
        MatchJobs(replace(jobs.settings, model_list=manifest), jobs.catalog)


def test_classroom_cli_rejects_checkpoint_conflict_before_rom_validation():
    import os
    import subprocess
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, str(ROOT / 'mgba/play_gui.py'), '--rom', 'missing.gba',
        '--source', 'missing.sav', '--model-list', 'classroom.json', '--checkpoint', 'latest'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert '--checkpoint' in result.stderr and '--model-list' in result.stderr
    assert 'not allowed with argument' in result.stderr


def test_classroom_launch_rejects_unlisted_ids_and_uses_static_listed_path(jobs, tmp_path, monkeypatch):
    manifest, _ = classroom_manifest(tmp_path)
    jobs.settings = replace(jobs.settings, checkpoint=None, model_list=manifest)
    jobs.models()
    payload = request_payload(jobs)
    for unlisted in ('latest', str(tmp_path / 'latest.zip'), 'unlisted-model'):
        with pytest.raises(ValueError, match='select an available PPO checkpoint'):
            jobs.start(dict(payload, checkpoint=unlisted))
        assert jobs.thread is None
    chosen = []
    def work(run, teams, checkpoint, workers, speed, stochastic):
        chosen.append(checkpoint)
        jobs.update(status='complete')
    monkeypatch.setattr(jobs, '_work', work)
    jobs.start(dict(payload, checkpoint='coached'))
    assert wait_job(jobs)['status'] == 'complete'
    assert chosen == [(tmp_path / 'models/classroom/coached-ai.zip').resolve()]
