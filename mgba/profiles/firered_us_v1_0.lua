return {
    id = "firered_us_v1_0",
    game_codes = { "BPRE" },
    sha1 = "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc",
    crc32 = "dd88761c",
    reset_regions = { { 0x02000000, 0x40000 }, { 0x03000000, 0x8000 } },
    -- pret/pokefirered symbols branch; see docs/automatic_states.md.
    capture = {
        main = 0x030030F0,
        save_block1_ptr = 0x03005008,
        player_avatar = 0x02037078,
        object_events = 0x02036E38,
        palette_fade = 0x02037AB8,
        script_status = 0x03000EA8,
        field_locked = 0x03000F9C,
        tasks = 0x03005090,
        save_status = 0x030053A0,
        party_count = 0x02024029,
        cb1_overworld = 0x08056534,
        cb2_overworld = 0x080565B4,
        cb2_title = 0x08078B9C,
        cb2_menu = 0x0800C2D4,
        task_menu_input = 0x0800CA54,
        map_group = 5, map_number = 5, x = 10, y = 4,
    },

    -- Battle addresses and decoding rules are added and validated in Phase 2D.
    battle = {},
}
