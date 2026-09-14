# Linked environment pools

[Setup](setup/mgba.md) · [Build states](automatic_states.md) · [Launcher flags](reference/launch-pool.md)

A pool pairs an even number of players into local native mGBA cable sessions.
Each pair shares one mGBA process with two cores; different pairs run concurrently.
Four environments make two battles, and eight make four.

## Launch four players

Use generated counter states from before any cable connection:

```bash
python mgba/launch_pool.py --rom "$PWD/FireRed.gba" --environments 4 \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --speed 4
```

| Pair | Environment slots | Default ports | States |
|---|---|---|---|
| 1 | agent-1, agent-2 | 55355, 55356 | fire, water |
| 2 | agent-3, agent-4 | 55357, 55358 | fire, water |

Two state arguments repeat for every pair. Alternatively provide one per player,
in order. `--base-port` changes the consecutive port range; `--ports` supplies all
ports explicitly. Choose one port option. Use `--headless` for Xvfb and
`--speed unlimited` to remove the pacing limit. Actual throughput depends on the
machine; [benchmark it](benchmark.md) with the intended workload.

## Automatic connection

Leave controls alone while the launcher talks to the wired Direct Corner NPC,
chooses Colosseum and Single Battle, confirms saving, completes the cable
handshake, and walks both players onto their battle pads. It releases input at
the first command menus after verifying both sides.

```text
match 1: PASS, both players at the first link-battle command menu
match 2: PASS, both players at the first link-battle command menu
POOL READY: 2 linked pairs / 4 environments.
```

Completion order can differ. `--lobby-only` stops in the Colosseum before battle.
Without `--commands`, the launcher leaves the games ready and open. It does not
pick actions or train a policy. Add a [switch sequence](link-commands.md), use the
[benchmark](benchmark.md) for repeated scripted battles, or use
[train_ppo.py](ppo-training.md) to train and start new episodes automatically.

The queue is local to the launcher. Agent IDs label player slots; there is no
network matchmaking service or attachment of existing emulator processes.

## Outputs and control

`pool.json` records assignments, state paths, ports, process IDs and readiness.
Each `match-N/` contains connection and adapter diagnostics. Generated per-player
save directories isolate the input states and original battery save from in-game
save operations.

Ctrl+C stops all sessions. A pair failure stops the pool and retains diagnostics;
see [troubleshooting](troubleshooting.md). Never reset one core inside a live
cable pair. Restart the pair from its pre-connection states instead. Independent
NPC tools that suspend a process, such as `test_turns.py`, do not schedule a linked
pair and must not be used for its actions.

To diagnose one pair, use [launch_link.py](link-test.md). Full patch and parallel
CMake build commands are in [mGBA setup](setup/mgba.md).
