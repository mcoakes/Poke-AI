#!/usr/bin/env python3
"""Rebuild the bundled facts from the pinned, ROM-free pret/pokefirered source."""
import argparse
import hashlib
import json
import re
from pathlib import Path

COMMIT = "c75f352304d529f6ba92d4f74b9cf8b5c3810788"


def export(source, project):
    hashes = {}

    def read(relative):
        data = (source / relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def constants(relative, prefix):
        return {key: int(value) for key, value in re.findall(
            rf"^#define {prefix}_(\w+)\s+(\d+)\s*$", read(relative), re.M)}

    def blocks(text, prefix):
        return dict(re.findall(rf"\[{prefix}_(\w+)\]\s*=\s*(.*?)(?=\[{prefix}_|\Z)", text, re.S))

    names = json.loads((project / "mgba/profiles/firered_us_v1_0_names.json").read_text())
    species_ids = constants("include/constants/species.h", "SPECIES")
    move_ids = constants("include/constants/moves.h", "MOVE")
    item_ids = constants("include/constants/items.h", "ITEM")
    dex = re.findall(r"NATIONAL_DEX_(\w+)", read("include/constants/pokedex.h").split("};")[0])
    dex = {key: i for i, key in enumerate(dex) if 1 <= i <= 386}
    levels = read("src/data/pokemon/level_up_learnsets.h")
    # Keep FireRed branches, including its Deoxys form's learnset.
    levels = re.sub(r"#if defined\(FIRERED\)\n(.*?)#elif defined\(LEAFGREEN\).*?#endif", r"\1", levels, flags=re.S)
    levels = {key: re.findall(r"LEVEL_UP_MOVE\((\d+), MOVE_(\w+)\)", body)
              for key, body in re.findall(r"s(\w+)LevelUpLearnset\[\]\s*=\s*\{(.*?)\};", levels, re.S)}
    pointers = dict(re.findall(r"\[SPECIES_(\w+)\] = s(\w+)LevelUpLearnset", read("src/data/pokemon/level_up_learnset_pointers.h")))
    tm = blocks(read("src/data/pokemon/tmhm_learnsets.h"), "SPECIES")
    tutors = blocks(read("src/data/pokemon/tutor_learnsets.h"), "SPECIES")
    eggs = dict(re.findall(r"egg_moves\((\w+),(.*?)\)", read("src/data/pokemon/egg_moves.h"), re.S))
    evolutions = blocks(read("src/data/pokemon/evolution.h"), "SPECIES")
    parents = {}
    for key, body in evolutions.items():
        for child in re.findall(r"SPECIES_(\w+)", body):
            parents[child] = key
    info = blocks(read("src/data/pokemon/species_info.h"), "SPECIES")
    special = {"Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"}
    moves = {}
    for key, body in blocks(read("src/data/battle_moves.h"), "MOVE").items():
        if key == "NONE" or key not in move_ids:
            continue
        number = move_ids[key]
        move_type = re.search(r"\.type = TYPE_(\w+)", body)[1].title()
        power = int(re.search(r"\.power = (\d+)", body)[1])
        moves[names["moves"][str(number)]] = {
            "id": number, "type": move_type, "power": power,
            "category": "status" if power == 0 else "special" if move_type in special else "physical",
        }

    def learnset(key):
        result = {move for _, move in levels[pointers[key]]}
        result.update(re.findall(r"TMHM\((?:TM|HM)\d+_(\w+)\)", tm.get(key, "")))
        result.update(re.findall(r"TUTOR\(MOVE_(\w+)\)", tutors.get(key, "")))
        result.update(re.findall(r"MOVE_(\w+)", eggs.get(key, "")))
        if key in parents:
            result.update(learnset(parents[key]))
        return result

    species = {}
    for key, number in dex.items():
        name = names["species"][str(species_ids[key])]
        available = learnset(key)
        # Sketch can copy most moves; Struggle and Sketch itself are not copy targets.
        if key == "SMEARGLE":
            available.update(set(move_ids) - {"NONE", "STRUGGLE", "SKETCH"})
        types = re.search(r"\.types = \{TYPE_(\w+), TYPE_(\w+)\}", info[key])
        species[name] = {"id": number, "types": list(dict.fromkeys(t.title() for t in types.groups())),
                         "fully_evolved": key not in evolutions,
                         "moves": sorted(names["moves"][str(move_ids[move])] for move in available)}
    items = {}
    for item in json.loads(read("src/data/items.json"))["items"]:
        number = item_ids.get(item["itemId"].removeprefix("ITEM_"))
        if number is None or item["pocket"] == "POCKET_KEY_ITEMS" or item["importance"]:
            continue
        label = names["items"][str(number)]
        if label.startswith("?"):
            continue
        items[label] = {"id": number, "battle_effect": item["holdEffect"] != "HOLD_EFFECT_NONE"}
    items["None"] = {"id": 0, "battle_effect": False}
    return {"profile": "firered_us_v1_0", "source": f"https://github.com/pret/pokefirered/tree/{COMMIT}",
            "source_sha256": hashes, "species": species, "moves": moves, "items": items}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = export(args.source, Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{len(data['species'])} species, {len(data['moves'])} moves, {len(data['items'])} items: {args.output}")
