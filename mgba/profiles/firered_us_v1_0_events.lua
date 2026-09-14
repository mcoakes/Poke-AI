-- Frame-sampled spectator events for BPRE v1.0. Addresses verified against
-- pret/pokefirered c75f352304d529f6ba92d4f74b9cf8b5c3810788 symbols.
-- Critical/effectiveness/miss messages come from the rendered battle string;
-- do not infer them from gMoveResultFlags, which can be stale or overwritten.
local chars = {[0]=" ", [0x1B]="é", [0x2D]="&", [0x2E]="+", [0x34]="Lv",
    [0x35]="=", [0x36]=";", [0x53]="POKé", [0x54]="MON", [0x5B]="%",
    [0x5C]="(", [0x5D]=")", [0xAB]="!", [0xAC]="?", [0xAD]=".", [0xAE]="-",
    [0xAF]="·", [0xB0]="…", [0xB1]='"', [0xB2]='"', [0xB3]="'", [0xB4]="'",
    [0xB5]="♂", [0xB6]="♀", [0xB7]="$", [0xB8]=",", [0xB9]="×", [0xBA]="/",
    [0xF0]=":"}
for i=0,9 do chars[0xA1+i]=tostring(i) end
for i=0,25 do chars[0xBB+i]=string.char(65+i); chars[0xD5+i]=string.char(97+i) end
local control_args = {[1]=1,[2]=1,[3]=1,[4]=3,[5]=1,[6]=1,[7]=0,[8]=1,[9]=0,
    [10]=0,[11]=2,[12]=1,[13]=1,[14]=1,[15]=0,[16]=2,[17]=1,[18]=1,[19]=1,
    [20]=1,[21]=0,[22]=0,[23]=0,[24]=0}
local function decode(raw)
    local out, i = {}, 1
    while i <= #raw do
        local c = raw:byte(i); i=i+1
        if c == 0xFC then
            local n = control_args[raw:byte(i)]
            if n == nil then out[#out+1]="[control]"; break end
            i=i+1+n
        elseif c == 0xFA or c == 0xFB or c == 0xFE then out[#out+1]=" "
        elseif c == 0xFD or c == 0xF7 or c == 0xF8 or c == 0xF9 then
            out[#out+1]="[symbol]"; i=i+1
        else out[#out+1]=chars[c] or "?" end
    end
    return table.concat(out):gsub("%s+", " "):match("^%s*(.-)%s*$")
end

return function()
    local self = {events={}, seq=0, previous={}, switching={}, initialized=false}
    function self:push(event)
        self.seq=self.seq+1
        event.seq, event.frame = self.seq, emu:currentFrame()
        self.events[#self.events+1]=event
        if #self.events > 256 then table.remove(self.events, 1) end
    end
    function self:sample()
        if self.error then return end
        if (emu:read8(0x03003529) & 2) == 0 then return end
        if (emu:read32(0x02022B4C) & 6) ~= 6 then return end -- one simulation master
        local count = emu:read8(0x02023BCC)
        if count ~= 2 then return end
        local printing = false
        for id=0,count-1 do
            local fn = emu:read32(0x03004FE0+id*4) & 0xFFFFFFFE
            -- Player and link-opponent CompleteOnInactiveTextPrinter handlers.
            if fn == 0x08030610 or fn == 0x0803ADD8 then printing=true end
        end
        local raw, cursor
        if printing then
            raw = emu:readRange(0x0202298C, 300):match("^(.-)\255")
            cursor = emu:read32(0x02020034)
            if raw and self.initialized and (not self.printing or raw ~= self.raw
                or (self.cursor and cursor < self.cursor)) then
                local text = decode(raw)
                if #text > 0 then self:push({kind="text", text=text}) end
            end
        end
        self.raw, self.cursor, self.printing = raw, cursor, printing
        local hp = {}
        for id=0,count-1 do
            local base=0x02023BE4+id*0x58
            local mon={battler_id=id, species_id=emu:read16(base),
                party_slot=emu:read16(0x02023BCE+id*2)+1,
                current_hp=emu:read16(base+0x28), max_hp=emu:read16(base+0x2C)}
            if mon.species_id > 0 and mon.species_id <= 411 and mon.party_slot <= 6
                and mon.max_hp > 0 and mon.current_hp <= mon.max_hp then
                local before=self.previous[id]
                local transition = self.switching[id] or (before and before.party_slot ~= mon.party_slot)
                if transition then
                    -- Party indexes may change before gBattleMons is copied.
                    -- Establish the incoming HP baseline before measuring losses.
                    local side=emu:read8(0x02023BD6+id) & 1
                    local party=(side == 0 and 0x02024284 or 0x0202402C)+(mon.party_slot-1)*100
                    self.switching[id] = mon.current_hp ~= emu:read16(party+86)
                        or mon.max_hp ~= emu:read16(party+88)
                end
                if not self.switching[id] then hp[#hp+1]=mon end
                if not transition and before and before.species_id == mon.species_id and before.party_slot == mon.party_slot
                    and before.max_hp == mon.max_hp and before.current_hp ~= mon.current_hp then
                    self:push({kind="hp", battler_id=id, species_id=mon.species_id,
                        before=before.current_hp, after=mon.current_hp, max_hp=mon.max_hp})
                end
                self.previous[id]=mon
            else self.previous[id]=nil end
        end
        -- Attach a contemporaneous HP display to the last event of this frame.
        local last=self.events[#self.events]
        if last and last.frame == emu:currentFrame() then last.hp=hp end
        self.initialized=true
    end
    return self
end
