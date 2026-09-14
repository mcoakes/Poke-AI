# Rewards

[Training](ppo-training.md) · [Observations and actions](turn-environment.md) · [Reports](battle-logs.md)

PPO uses the same reward calculation with either backend. The default is
[damage_knockout_win.json](../config/rewards/damage_knockout_win.json), including
when `--reward-config` is omitted or a checkpoint is resumed.

## Default points

| Event | Points |
|---|---:|
| Remove 25% of an opponent's maximum HP in one decision | +6.25 |
| Remove 50% in one decision | +25 |
| Remove 100% in one decision | +100 |
| Knock out an opposing Pokémon | +500 |
| Win / lose | +5000 / −5000 |
| Each battle turn, for each player | −50 |
| Damage-reward limit per opposing Pokémon per battle | +100 |
| Draw or truncation | No terminal bonus |

Damage points are `100 × (HP lost / maximum HP)²`, calculated per opposing
party member and summed. Bonuses stack: a full-HP final knockout and win earns
`100 + 500 + 5000 − 50 = 5550` points if that Pokémon's damage budget is unspent.
Two separate 50% reductions pay 25 + 25; one 100% reduction pays 100.

## Battle length and healing

| Win length | Win bonus | Turn costs | Subtotal before damage and knockouts |
|---:|---:|---:|---:|
| 30 turns | 5000 | −1500 | 3500 |
| 60 turns | 5000 | −3000 | 2000 |

With otherwise equal rewards, the faster win earns 1500 more points. Damage
budgets do not refill after healing, Rest or switching. Against six Pokémon,
total damage points cannot exceed 600, so another 30 turns of damage rewards
cannot outweigh those extra turn costs.

A faint or Baton Pass can create another selection within the same turn. The
turn ledger charges that turn once, including its terminal resolution. Startup
and emulator frames have no reward cost. Automatic actions that advance several
game turns incur the corresponding turn costs.

Damage is a net HP difference between decisions, including poison, recoil and
other HP costs; it is not attributed to a particular attack. Healing in that
interval can offset damage. Switching preserves each Pokémon's HP baseline.
Knockouts pay on the transition to zero HP, without paying again for replacement.
There is no separate penalty for receiving damage, fainting, switching or waiting.
These weights favor burst damage, knockouts and prompt wins; they do change the
learning objective compared with a win-only score.

## Configuration

Copy the default JSON and pass `--reward-config config/rewards/my_rewards.json`.
All settings are read again at startup, even with `--resume`.

| JSON key | Default | Meaning |
|---|---:|---|
| `damage` | 100 | Multiplier for the fraction of maximum HP lost |
| `damage_exponent` | 2 | Burst preference; 1 makes damage linear |
| `knockout` | 500 | Bonus per opposing knockout |
| `win` | 5000 | Terminal victory bonus |
| `loss` | −5000 | Terminal defeat reward |
| `turn_penalty` | 50 | Cost per game turn; 0 disables it |
| `damage_cap_per_pokemon` | 100 | Per-opponent budget; 0 disables the cap |
| `scale` | 0.001 | Multiplier applied before rewards reach PPO |

Custom files omitting the turn penalty or damage cap use 0 for those fields.
[terminal.json](../config/rewards/terminal.json) provides terminal-only rewards.
Reward settings must be finite; see validation in [linked_reward.py](../src/firered_rl/linked_reward.py).

Reports show **unscaled points**. PPO receives scaled values, so 5550 points
becomes 5.55; value predictions and losses use those scaled units. Episode returns
are undiscounted sums, while PPO also applies its saved discount factor.

`config.json` records the settings, and `gym-steps.jsonl` records the component
breakdown and cumulative battle turns. Compare reward means using the same
configuration. The separate NPC test utilities and human-match scoreboard do not
use this PPO reward configuration.
