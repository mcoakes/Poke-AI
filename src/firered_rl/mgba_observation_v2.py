"""FireRed US 1.0 linked inspection -> the existing Showdown v2 contract.

Native facts come from the paired adapter, never from species-stat estimates.
Move descriptors use the pinned training Dex; Hidden Power uses native IVs.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def move_catalog():
    return json.loads(Path(__file__).with_name('gen3_move_details.json').read_text())['moves']


def move_details(mon):
    result = [[0] * 6 for _ in range(4)]
    for move in mon['moves']:
        slot, move_id = move['slot'] - 1, move['move_id']
        if not 0 <= slot < 4:
            raise ValueError('native v2 move slot is invalid')
        if not move_id:
            continue
        if str(move_id) not in move_catalog():
            raise ValueError(f'native v2 unsupported move ID {move_id}')
        row = list(move_catalog()[str(move_id)])
        if move_id == 237:  # Hidden Power: hp/atk/def/spe/spa/spd packed IV order.
            ivs = [(mon['ivs_raw'] >> (5*i)) & 31 for i in range(6)]
            kind = sum((v & 1) << i for i, v in enumerate(ivs)) * 15 // 63 + 1
            if kind >= 9:  # Skip the cartridge's ??? type.
                kind += 1
            power = sum(((v >> 1) & 1) << i for i, v in enumerate(ivs)) * 40 // 63 + 30
            row[0], row[3], row[5] = power, int(kind >= 10), kind
        result[slot] = row
    return result


def native_fields(report):
    from .observation_v2 import v2_spaces, validate_v2_fields
    if report.get('native_v2') != 1 or report.get('backend') != 'mgba':
        raise ValueError('mGBA v2 requires paired native inspection from adapter 0.19.0; restart both emulators')
    if report.get('profile') != 'firered_us_v1_0':
        raise ValueError('native v2 supports the verified FireRed US v1.0 profile only')
    battle = report['battle']
    if 'simulation_source_port' not in battle or 'v2_opponent_party' not in battle:
        raise ValueError('mGBA v2 needs both linked players and the authoritative simulation')
    fields = {k: np.zeros(s.shape, dtype=s.dtype) for k, s in v2_spaces().items()}
    try:
        parties = [report['party'], battle['v2_opponent_party']]
        for side, party in enumerate(parties):
            for slot, mon in enumerate(party):
                if slot >= 6:
                    raise ValueError('native v2 party exceeds six slots')
                if not mon.get('occupied'):
                    continue
                if not mon.get('valid') or mon.get('is_egg'):
                    raise ValueError('native v2 party record is invalid or an Egg; retry inspection')
                fields['v2_party'][side, slot] = [1, mon['level'], mon['current_hp'], mon['max_hp'],
                    *mon['base_type_ids'], mon['held_item_id'], mon['ability_id'],
                    int(mon['current_hp'] == 0), mon['status_raw']]
                fields['v2_stats'][side, slot] = mon['stats']
                fields['v2_move_details'][side, slot] = move_details(mon)
        seen = set()
        for mon in battle['battlers']:
            side = {'player': 0, 'opponent': 1}[mon['side']]
            slot = mon['party_slot'] - 1
            if side in seen or not mon.get('valid') or not 0 <= slot < 6:
                raise ValueError('native v2 requires one valid active battler per side')
            seen.add(side)
            if not fields['v2_party'][side, slot, 0]:
                raise ValueError('native v2 active party slot is empty')
            # Party copies can lag battle fields; mirror the simulator contract
            # for live HP/status/items/ability while retaining base stats/types.
            p = fields['v2_party'][side, slot]
            p[2:4] = mon['current_hp'], mon['max_hp']
            p[6:10] = mon['held_item_id'], mon['ability_id'], int(mon['current_hp'] == 0), mon['status_raw']
            fields['v2_active'][side] = [mon['current_hp'], mon['max_hp'], mon['held_item_id'], mon['ability_id']]
            fields['v2_active_stats'][side] = mon['stats']
            stages = mon['stat_stages']
            fields['v2_boosts'][side] = [stages[i] - 6 for i in (1, 2, 4, 5, 3, 6, 7)]
            fields['v2_volatiles'][side] = mon['volatile_status_raw']
            fields['v2_move_details'][side, slot] = move_details(mon)
        if seen != {0, 1}:
            raise ValueError('native v2 active battlers are incomplete')
        weather, duration = battle['weather_raw'], battle['weather_duration']
        if not isinstance(weather, int) or not isinstance(duration, int):
            raise ValueError('native v2 weather fields are unavailable')
        kinds = [(1, 0x60, 0x40), (2, 0x07, 0x04), (3, 0x18, 0x10), (4, 0x80, 0)]
        current = [(kind, bool(weather & permanent)) for kind, mask, permanent in kinds if weather & mask]
        if weather & ~0xFF or len(current) > 1:
            raise ValueError('native v2 weather flags are inconsistent')
        if current:
            kind, permanent = current[0]
            fields['v2_weather'][:] = kind, 0 if permanent else duration, int(permanent)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f'native mGBA v2 data is incomplete: {error}; restart with adapter 0.19.0') from error
    return validate_v2_fields(fields)
