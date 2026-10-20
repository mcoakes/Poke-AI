# Six types

This controlled team is designed to make type matchups easier to study. It is
not a competitive team: coverage moves are deliberately excluded.

| Pokémon | Type | Four moves |
|---|---|---|
| Arcanine | Fire | Flamethrower, Fire Blast, Flame Wheel, Ember |
| Blastoise | Water | Surf, Hydro Pump, Water Pulse, Water Gun |
| Sceptile | Grass | Leaf Blade, Giga Drain, Mega Drain, Absorb |
| Jolteon | Electric | Thunderbolt, Thunder, Shock Wave, Thunder Shock |
| Glalie | Ice | Ice Beam, Blizzard, Icy Wind, Powder Snow |
| Machamp | Fighting | Cross Chop, Brick Break, Karate Chop, Low Kick |

Every move is in the bundled species learnset and matches the Pokémon's type.
All six are level 50, with 31 IVs, 252 attacking EVs, 252 Speed EVs and 4 HP EVs.
Machamp uses Attack/Adamant; the other five use Special Attack/Modest, following
Gen III's type-based physical/special split. Held items are None to keep the
experiment simpler. Arcanine leads on both sides.

Edit [teams/six_types.json](teams/six_types.json) to change the roster.

```bash
frozen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"

python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$frozen_checkpoint" --total-timesteps 100000
```

Resume keeps saved PPO settings unless you supply explicit hyperparameter flags. Check the `Effective PPO`
startup line. This directory currently contains one team, so both players always
receive it. Add more JSONs to sample a larger pool, or use `--mirror-teams` to
keep each randomly chosen matchup mirrored.
