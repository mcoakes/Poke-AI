-- Read-only cable state; independent of battle initialization and party decoding.
return function()
    local status = emu:read32(0x03003F20)
    local flags = emu:read32(0x02022B4C)
    local active = (emu:read8(0x03003529) & 2) ~= 0
    local result = {ok=true, frame=emu:currentFrame(), status_raw=status,
        local_id=status & 3, player_count=(status >> 2) & 7,
        connection_established=(status & 0x40) ~= 0,
        error_bits=status & 0x7F000, error_occurred=emu:read8(0x03003EAC) ~= 0,
        received_remote_players=emu:read8(0x03003F64) ~= 0,
        wireless=emu:read8(0x03003F3C) ~= 0, link_type=emu:read16(0x0202271A),
        battle_active=active, link_battle=active and (flags & 2) ~= 0,
        players={},
    }
    if result.received_remote_players and result.player_count >= 1 and result.player_count <= 4 then
        for i=0,result.player_count-1 do
            local base = 0x0202273C+i*0x1C
            result.players[#result.players+1] = {id=i, trainer_id=emu:read32(base+4),
                version=emu:read16(base), link_type=emu:read32(base+0x14)}
        end
    end
    return result
end
