# Troubleshooting

[Documentation](README.md) · [Dependencies](setup/README.md) · [Logs](battle-logs.md)

Use the check matching the failing stage. Keep the printed run path and original
error; its per-pair logs usually contain more useful evidence than a screenshot alone.

## npm installs with dependency warnings

`added ... packages` followed by an audit summary means installation completed.
The pinned Showdown package currently pulls `sockjs`, which pulls `uuid@8.3.2`.
The audit checked on 2026-09-13 reports three moderate package entries for that
chain, all stemming from the same
[UUID buffer-bounds advisory](https://github.com/advisories/GHSA-w5hq-g745-h8pq).

Read the report without changing dependencies:

```bash
npm --prefix sim/showdown audit --omit=optional
```

Do not apply `npm audit fix --force` to this pin: the checked report proposes
Showdown **0.9.2**, a downgrade from required **0.11.11**. The adapter checks its
version because it uses private simulator APIs. Dependency updates need a coordinated
pin/lockfile change and simulator validation.

The project runs a local battle library, without starting the Showdown server.
A checked reset and turn did not load `sockjs` or `uuid`, so that tested path did
not invoke the affected code. The dependency advisory still exists in the
installed package tree. You can proceed with [local training](showdown-training.md);
the warning is not an installation failure. The separate deprecation message
reports UUID's support status rather than a failed simulator operation.

## Python, Node or .NET cannot be found

Activate the project virtual environment and install the needed extras from
[Python setup](setup/python.md). Check `node --version` against
[Showdown setup](setup/showdown.md). For .NET, follow
[save-editor setup](setup/save-editor.md), including PATH selection when an older
system SDK hides a project-local installation. The editor targets `net10.0`.

## CMake cannot find zipcmp, zipmerge or ziptool

Some distributions split utilities from `libzip-dev`. Install all three:

```bash
sudo apt install --reinstall libzip-dev zipcmp zipmerge ziptool
command -v zipcmp zipmerge ziptool
```

Rerun the configure command in [mGBA setup](setup/mgba.md) before building. A
failed configure does not generate a usable Makefile. Compilation can use
`cmake --build build/mgba --parallel 4` if unrestricted parallel jobs exhaust RAM.

## Windows open but adapters are unavailable

Use the project-local binary, apply all native patches, then configure, build
and install as described in [mGBA setup](setup/mgba.md). The launchers depend on
script autoloading; an unpatched system binary can open a game without an adapter.
Close and relaunch processes after updating Lua or installing a new binary.

```bash
python mgba/test_adapter.py --port 55355
```

The response identifies the loaded adapter and ROM. A file on disk can be newer
than the script loaded in an existing process. `AGB-BPRE` is normalized to `BPRE`;
an unsupported checksum still requires the correct ROM/profile.

## Address already in use

Check the requested listeners:

```bash
ss -ltnp 'sport = :55355 or sport = :55356'
```

Stop the launcher using those ports, or select another range with `--base-port`
or `--ports` where supported. Do not kill unrelated processes. If there is no
listener, check the full bind error and retry after the prior session closes.

## A generated save has no Continue or the counter is locked

A `.sav` is a raw 131,072-byte battery save; `.ss1` is an emulator snapshot.
Load a battery save using Continue, and a snapshot using the emulator's state
loader. For manual battery replacement, close mGBA first and match the ROM's save
basename and configured save location. [First save setup](setup/first-save.md)
explains how to preserve the working source.

Badge/warp preparation does not deliver Oak's Parcel. Complete that quest in the
source game and verify the wired Direct Corner service before generating states.
Use [build_states.py](automatic_states.md) for team creation and counter capture.
A capture failure prints its working directory and retains diagnostic output.

## Inspection shows an unknown menu or invalid counts

During transitions, `busy_or_unknown` and unavailable cursors can be normal.
Inspect again at a stable command or party menu. In a cable battle, include the
peer port to resolve authoritative simulation data:

```bash
python mgba/inspect_state.py --port 55356 --peer-port 55355 --json
```

A count with `valid: false` is not a trustworthy zero. Do not infer victory from
it. Retain the complete JSON from both perspectives if the state remains invalid.

## Connection or battle controller fails

Keep the failing match's `connection.jsonl`, both `playerN/adapter.log` files,
`mgba.log`, and the final `gym-steps.jsonl` report if play had begun. Include the
run's `failure.json` when present. `battle-readable.log` shows submitted actions;
`battle.log` supplies nearby game text.

Use one [visible pair](link-test.md) at `--speed 1` to inspect the failing stage.
A network-adapter probe can briefly mention wireless state before wired selection;
a genuinely unsupported wireless service needs the correct Direct Corner setup.
Never reset just one side of a connected pair. Restart the pair together.

## No reward mean yet, or resumed batch size fails

Reward means become available after the first ended episode; PPO update messages
can precede them. See [metric definitions](battle-logs.md).

Resume restores the checkpoint's rollout and batch sizes. The batch size must
divide saved rollout steps × learning-player count. A frozen opponent halves the
number of learning players. For example, rollout 16 and batch 32 require two
learning players: four physical players in frozen-opponent mode.

A frozen opponent defaults to choosing the most probable legal action, while the
learner samples. Add `--opponent-stochastic` for sampling on both sides. Unequal
wins alone do not establish a role bug; compare teams, actions and episode counts
using the [training reports](battle-logs.md).
