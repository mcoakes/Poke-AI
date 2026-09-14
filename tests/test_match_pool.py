from pathlib import Path
import pytest

from firered_rl.match_pool import MatchPool, PoolMember


def test_late_join_pairs_once_and_preserves_fifo():
    pool = MatchPool()
    members = [PoolMember(f"agent-{i}", 55355+i, Path(f"team{i}.ss1")) for i in range(4)]
    pool.join(members[0])
    assert pool.take_pair() is None
    for member in members[1:]:
        pool.join(member)
    assert pool.take_pair() == tuple(members[:2])
    assert pool.take_pair() == tuple(members[2:])
    assert pool.take_pair() is None and not pool.waiting
    with pytest.raises(ValueError):
        pool.join(members[0])
    with pytest.raises(ValueError):
        pool.join(PoolMember("new", 55355, Path("new.ss1")))


def test_pool_cli_maps_states_to_distinct_cables(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "mgba"))
    from launch_pool import make_members
    states = [Path("fire.ss1"), Path("water.ss1")]
    members, pool = make_members(8, states)
    assert [m.state for m in members] == states*4
    assert [m.port for m in members] == list(range(55355, 55363))
    pairs = [pool.take_pair() for _ in range(4)]
    assert len({m.agent_id for pair in pairs for m in pair}) == 8
    for count, files, ports in [(1, states, None), (3, states, None), (4, states*3, None),
                                (4, states, [55355]*4), (4, states, [55355, 55356])]:
        with pytest.raises(ValueError):
            make_members(count, files, ports)
