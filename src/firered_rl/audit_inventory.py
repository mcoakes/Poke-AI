"""Describe and probe the current observation contract without changing it.

The inventory describes explicit policy inputs, not facts a trained embedding
might infer from species or move IDs. Probes use synthetic reports and the real
encoders/preprocessor, before BattleFeatures' dense projection. They do not
load a checkpoint, start a simulator, train weights, or claim battle coverage.
"""
from copy import deepcopy


def observation_inventory():
    """Return JSON-serializable rows for the current schema (no torch needed)."""
    rows = [
        ("own_party_species", "present", "party[].species_id",
         "party[:,1] -> species embedding, masked by validity",
         "Base species remain separate from temporary active species."),
        ("active_species", "present", "battle.battlers[].species_id",
         "active[:,1] -> species embedding for each side",
         "Active species do not replace the base party species."),
        ("levels_and_types", "present", "party[].level/base_type_ids; battlers[].level/type_ids",
         "level / 100; two type embeddings per own party member and active battler",
         "Opponent bench levels and types are absent."),
        ("own_hp", "present", "party[] and player battler current_hp/max_hp",
         "HP fraction clipped to [0,1], plus max_hp / 1024",
         "Own active HP is represented separately from the party copy."),
        ("opponent_active_hp", "partial", "opponent battler current_hp/max_hp",
         "round(100 * current_hp / max_hp), with encoded maximum 100",
         "Percentage rounding hides small changes and the true maximum HP."),
        ("opponent_bench_hp", "missing", "battle.opponent_party[].current_hp/max_hp",
         "None: opponent_party contains only validity and species ID",
         "Bench HP and fainted state are unavailable explicitly."),
        ("own_moves", "present", "party[].moves; player battler.moves replace the active row",
         "Move/type embeddings plus current_pp and max_pp / 64, clipped to [0,1]",
         "Unused active move slots are zero; bench rows retain their party moves."),
        ("opponent_moves", "present", "opponent_party[].moves; opponent battler.moves",
         "Same move/type/PP features; active row is cleared before its override",
         "All supplied opponent bench moves are exposed, including unrevealed moves."),
        ("explicit_move_power", "missing", "moves[].power (present in simulator reports)",
         "No power scalar; move ID embedding only",
         "mGBA legality can use power for Taunt; Showdown supplies its own mask."),
        ("explicit_move_accuracy", "missing", "Not emitted by the Showdown report adapter",
         "No accuracy scalar; move ID embedding only",
         "Adding moves[].accuracy to a report does not add a policy feature."),
        ("explicit_move_category", "missing", "Not emitted by the Showdown report adapter",
         "No physical/special/status field; move/type embeddings only",
         "Gen III type/category relationships can only be inferred from IDs."),
        ("battle_stats", "missing", "No attack/defense/speed/stat vector in encode",
         "None beyond HP, level, species and types",
         "Computed non-HP stats and IV/EV/nature effects are not explicit inputs."),
        ("stat_stages", "missing", "mGBA battlers[].stat_stages; not emitted by Showdown adapter",
         "None",
         "Attack/defense/speed/accuracy/evasion stage changes are not encoded."),
        ("weather", "missing", "Not emitted by the Showdown report adapter",
         "None",
         "Weather identity and duration are absent."),
        ("abilities", "missing", "party[] and battlers[].ability_id",
         "No ability ID embedding or scalar",
         "Ability effects may change legality or later battle values indirectly."),
        ("own_held_items", "present", "party[].held_item_id",
         "party[:,8] -> item embedding",
         "Active held-item fields are ignored; the party copy must be current."),
        ("opponent_held_items", "missing", "opponent_party[] and opponent battler held_item_id",
         "None",
         "The opponent's item identity is not exposed explicitly."),
        ("major_status", "partial", "party[].status_raw; battlers[].status_raw",
         "8 x 32 status bits expanded as integers before float32 conversion",
         "Own party and both actives are covered; opponent bench status is absent."),
        ("major_status_float_safety", "present", "policy_observation integer status bitfields",
         "Raw status columns are zeroed; exact individual bits reach _flatten",
         "High and low bits survive together; the aggregate float would lose low bits."),
        ("own_volatiles", "partial", "player battler.volatile_status_raw",
         "32 individual volatile bits expanded before float32 conversion",
         "Only flags/counters supplied by the adapter are available, not all simulator state."),
        ("opponent_volatiles", "missing", "opponent battler.volatile_status_raw",
         "Opponent active volatile field is deliberately zeroed by encode",
         "Opponent volatile_bits are consequently all zero."),
        ("bag", "partial", "bag pocket item_id/quantity",
         "Item embeddings and quantity / 999, masked by nonzero item ID",
         "The Showdown adapter supplies an empty bag."),
        ("action_mask", "present", "encode restrictions; Showdown report.action_mask override",
         "11 mask bits reach the features and policy action masking",
         "Mask changes reveal legality, not a complete explanation of the underlying effect."),
        ("turn_and_terminal_context", "present", "battle.turn, required switch, terminal argument, outcome_raw",
         "context scaled by [255,1,1,10]",
         "Outcome is encoded only when terminal=True; weather is not part of context."),
        ("opponent_active_slot", "present", "opponent battler.party_slot",
         "opponent_active_index / 6, using original 1-based party slots",
         "Opponent active's own active[:,8] field is zero; the index is separate."),
        ("all_moves_no_hidden_information_profile", "partial", "Complete supplied parties and move lists",
         "Both teams' species and all supplied moves/PP, including opponent bench",
         "Deliberate training profile, not a hidden-information leak test; this does not "
         "mean every battle field is available (opponent bench HP/items and other gaps remain)."),
    ]
    keys = ("name", "status", "source", "policy_representation", "limitation")
    return [dict(zip(keys, row)) for row in rows]


def _probe_report():
    """Small valid report with independent party and active copies."""
    def move(slot, move_id, type_id, pp, power):
        return dict(slot=slot, move_id=move_id, type_id=type_id,
                    base_type_id=type_id, current_pp=pp, max_pp=pp, power=power)

    def mon(slot, species, hp, types, moves):
        return dict(slot=slot, party_slot=slot, valid=True, occupied=True,
                    is_egg=False, species_id=species, level=50, current_hp=hp,
                    max_hp=hp, base_type_ids=types, type_ids=types[:], status_raw=0,
                    volatile_status_raw=0, held_item_id=0, ability_id=65, moves=moves)

    party = [mon(1, 9, 120, [11, 11], [move(1, 57, 11, 15, 95),
                                       move(2, 33, 0, 35, 35)]),
             mon(2, 6, 150, [10, 2], [move(1, 53, 10, 15, 95)])]
    opponents = [mon(1, 3, 160, [12, 3], [move(1, 75, 12, 25, 55),
                                          move(2, 33, 0, 35, 35)]),
                 mon(2, 131, 200, [11, 15], [move(1, 58, 15, 10, 95)])]
    battlers = [dict(deepcopy(party[0]), side="player", absent=False),
                dict(deepcopy(opponents[0]), side="opponent", absent=False)]
    return dict(frame=0, profile="firered_us_v1_0", backend="showdown", party=party,
                bag={}, action_mask=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                battle=dict(active=True, valid=True, turn=3, outcome_raw=0,
                            type_flags=10, opponent_party=opponents, battlers=battlers,
                            menus=[dict(state="command")]))


def inventory_checks():
    """Probe actual encoding and policy features; missing inputs are limitations.

    Training dependencies are imported only here and import errors propagate.
    Randomly initialized embeddings are deterministic, with caller RNG restored.
    Comparisons stop at _flatten, before the dense learned projection; they show
    representability, not whether any trained policy uses a feature well.
    Supplementary stats/boosts/weather/accuracy/category keys test encoder
    omission; they do not claim those keys are emitted by the current adapter.
    """
    import numpy as np
    import torch
    from stable_baselines3.common.preprocessing import preprocess_obs

    from .ppo_features import BattleFeatures, policy_observation, policy_space
    from .showdown_env import showdown_observation
    from .turn_env import encode, observation_space

    policy_schema = policy_space()
    with torch.random.fork_rng(devices=[]):
        # Seed only CPU initialization; do not alter a caller's CUDA generators.
        torch.random.default_generator.manual_seed(1729)
        extractor = BattleFeatures(policy_schema, features_dim=8).eval()

    def represent(report):
        raw = showdown_observation(report)
        policy = policy_observation(raw)
        tensors = {key: torch.as_tensor(value).unsqueeze(0) for key, value in policy.items()}
        with torch.no_grad():
            flat = extractor._flatten(preprocess_obs(tensors, policy_schema))
        return raw, policy, flat

    baseline = _probe_report()
    base_raw, base_policy, base_flat = represent(baseline)
    base_encoded = encode(baseline)
    checks = []

    def record(name, passed, details, classification="pass"):
        checks.append(dict(name=name, status=classification if passed else "fail", details=details))

    def compare(name, mutate, *, change=True, classification=None, explanation="", verify=None):
        changed = deepcopy(baseline)
        mutate(changed)
        raw, policy, flat = represent(changed)
        encoded = encode(changed)
        raw_keys = [key for key in raw if not np.array_equal(raw[key], base_raw[key])]
        policy_keys = [key for key in policy if not np.array_equal(policy[key], base_policy[key])]
        encoded_keys = [key for key in encoded
                        if not np.array_equal(encoded[key], base_encoded[key])]
        feature_count = int(torch.count_nonzero(flat != base_flat).item())
        passed = (bool(raw_keys) == change and bool(policy_keys) == change
                  and bool(feature_count) == change)
        verified = None
        if verify is not None:
            verified = bool(verify(raw, policy, flat))
            passed = passed and verified
        record(name, passed, dict(explanation=explanation,
                                 encoder_changed_keys=encoded_keys,
                                 observation_changed_keys=raw_keys,
                                 policy_changed_keys=policy_keys,
                                 changed_feature_values=feature_count,
                                 expected_change=change,
                                 value_assertion_passed=verified),
               classification or ("pass" if change else "limitation"))

    record("observation_schema", observation_space().contains(base_raw)
           and policy_space().contains(base_policy) and bool(torch.isfinite(base_flat).all()),
           dict(raw_shapes={key: list(value.shape) for key, value in base_raw.items()},
                policy_feature_width=int(base_flat.shape[1]),
                comparison="BattleFeatures._flatten before its dense projection"))
    compare("own_bench_hp", lambda r: r["party"][1].update(current_hp=75),
            verify=lambda raw, *_: raw["party"][1, 3] == 75)
    compare("own_active_hp", lambda r: r["battle"]["battlers"][0].update(current_hp=60))
    compare("own_held_items", lambda r: r["party"][1].update(held_item_id=200))
    compare("own_party_species", lambda r: r["party"][1].update(species_id=150))
    for side in (0, 1):
        compare(f"active_species_side_{side}",
                lambda r, side=side: r["battle"]["battlers"][side].update(species_id=151),
                verify=lambda raw, *_, side=side: raw["active"][side, 1] == 151
                and raw["party"][0, 1] == 9 and raw["opponent_party"][0, 1] == 3)
    compare("opponent_active_hp", lambda r: r["battle"]["battlers"][1].update(current_hp=80),
            verify=lambda raw, *_: raw["active"][1, 3:5].tolist() == [50, 100])
    compare("opponent_maximum_hp", lambda r: r["battle"]["battlers"][1].update(
        current_hp=320, max_hp=320), change=False,
        explanation="Equal HP percentages hide the opponent's true maximum HP.")
    compare("opponent_bench_hp", lambda r: r["battle"]["opponent_party"][1].update(current_hp=0),
            change=False, explanation="Fainting a bench opponent does not reach these features.")
    compare("opponent_bench_major_status",
            lambda r: r["battle"]["opponent_party"][1].update(status_raw=64), change=False)
    for side in ("party", "opponent_party"):
        def change_move(r, side=side):
            mons = r["party"] if side == "party" else r["battle"][side]
            mons[1]["moves"][0].update(move_id=85)
        compare(f"all_moves_{side}_bench", change_move,
                explanation="Unrevealed bench move IDs are deliberately exposed.")
    compare("move_pp", lambda r: r["party"][1]["moves"][0].update(current_pp=3))
    for side in (0, 1):
        compare(f"active_move_override_side_{side}",
                lambda r, side=side: r["battle"]["battlers"][side]["moves"][0].update(move_id=85),
                verify=lambda raw, *_, side=side:
                raw["moves" if side == 0 else "opponent_moves"][0, 0, 0] == 85)
    compare("opponent_active_party_move_copy_ignored",
            lambda r: r["battle"]["opponent_party"][0]["moves"][0].update(move_id=85),
            change=False, classification="pass",
            explanation="The active opponent move list overrides its party copy.")
    compare("opponent_active_removed_moves_zeroed",
            lambda r: r["battle"]["battlers"][1].update(
                moves=r["battle"]["battlers"][1]["moves"][:1]),
            verify=lambda raw, *_: not raw["opponent_moves"][0, 1:].any())
    def remove_own_move(r):
        r["battle"]["battlers"][0]["moves"] = r["battle"]["battlers"][0]["moves"][:1]
        r["action_mask"][1] = 0

    compare("own_active_removed_moves_zeroed", remove_own_move,
            verify=lambda raw, *_: not raw["moves"][0, 1:].any(),
            explanation="Active move removal should clear obsolete party-copy slots; "
            "the active row must fully replace its party copy. The mask is updated for the removed move; "
            "use live Transform coverage to establish reachability.")
    record("unused_slots_zero", not any(base_raw[key][2:].any() for key in
           ("party", "moves", "opponent_party", "opponent_moves"))
           and not base_raw["moves"][:, 2:].any()
           and not base_raw["opponent_moves"][:, 2:].any(),
           dict(explanation="Unused party and move slots are zero in the encoded fixture."))

    for field, value in (("power", 120), ("accuracy", 25), ("category", "Status")):
        compare(f"explicit_move_{field}",
                lambda r, field=field, value=value:
                r["battle"]["battlers"][0]["moves"][0].update({field: value}),
                change=False, explanation="No explicit feature; supplied action mask is held fixed.")
    for field, value in (("ability_id", 66), ("stats", {"atk": 300, "spe": 250}),
                         ("stat_stages", [6, 8, 6, 6, 6, 6, 6, 6]),
                         ("boosts", {"atk": 2, "accuracy": -1})):
        compare(field, lambda r, field=field, value=value:
                r["battle"]["battlers"][0].update({field: value}), change=False,
                explanation="No explicit feature; no battle simulation or mask change is implied.")
    compare("weather", lambda r: r["battle"].update(weather="raindance", weather_turns=5),
            change=False)
    compare("opponent_held_items", lambda r: [m.update(held_item_id=200) for m in
            [r["battle"]["opponent_party"][0], r["battle"]["battlers"][1]]], change=False)
    compare("opponent_volatiles", lambda r: r["battle"]["battlers"][1].update(
        volatile_status_raw=(1 << 24) | 1), change=False,
        verify=lambda raw, policy, _: not policy["volatile_bits"][1].any())
    compare("own_party_major_status", lambda r: r["party"][1].update(status_raw=64))
    compare("own_active_major_status", lambda r: r["battle"]["battlers"][0].update(status_raw=16))
    compare("opponent_active_major_status",
            lambda r: r["battle"]["battlers"][1].update(status_raw=64))
    compare("own_volatiles", lambda r: r["battle"]["battlers"][0].update(
        volatile_status_raw=1 << 24))

    def high_status(r):
        r["party"][0]["status_raw"] = (1 << 30) | 3
        r["battle"]["battlers"][1]["status_raw"] = 64
        r["battle"]["battlers"][0]["volatile_status_raw"] = (1 << 31) | 1

    def exact_bits(raw, policy, flat):
        return (policy["status_bits"][0, [0, 1, 30]].tolist() == [1, 1, 1]
                and policy["status_bits"][7, 6] == 1
                and policy["volatile_bits"][0, [0, 31]].tolist() == [1, 1]
                and not policy["party"][:, 7].any()
                and not policy["active"][:, [7, 9]].any()
                and int(np.float32(raw["party"][0, 7])) != int(raw["party"][0, 7]))

    compare("major_status_and_volatile_float_safety", high_status, verify=exact_bits,
            explanation="High and low bits survive integer expansion; direct float32 loses low bits.")
    for field, high, group in (("status_raw", 1 << 30, "major_status"),
                               ("volatile_status_raw", 1 << 31, "volatile_status")):
        first = deepcopy(baseline)
        target = first["party"][1] if field == "status_raw" else first["battle"]["battlers"][0]
        target[field] = high
        _, _, high_flat = represent(first)
        target[field] = high | 1
        _, _, low_flat = represent(first)
        count = int(torch.count_nonzero(high_flat != low_flat).item())
        record(f"{group}_low_bit_reaches_features", count == 1,
               dict(changed_feature_values=count,
                    explanation="Changing only bit zero next to a high bit changes one flat feature."))
    compare("showdown_authoritative_action_mask", lambda r: r.update(
        action_mask=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        verify=lambda raw, *_: raw["action_mask"].tolist() == [0, 1]+[0]*9,
        explanation="Only the supplied Showdown mask changes; encode's guessed mask is unchanged.")
    compare("turn_context", lambda r: r["battle"].update(turn=4))
    return checks
