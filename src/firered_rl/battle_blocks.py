"""Non-overlapping episode blocks shared by live and saved-run reports."""


class BattleBlocks:
    def __init__(self, size=500):
        if size < 1:
            raise ValueError('block size must be positive')
        self.size = size
        self.finished = 0
        self.rows = []
        self.last_complete = None

    def snapshot(self):
        count = len(self.rows)
        completed = sum(row['terminated'] for row in self.rows)
        wins = {side: sum(row['winner'] == side for row in self.rows)
                for side in ('player1', 'player2', 'draw')}
        return dict(block=self.finished + 1, first_episode=self.finished * self.size + 1,
                    last_episode=self.finished * self.size + count, episodes=count,
                    full=count == self.size, completed=completed,
                    truncated=sum(row['truncated'] for row in self.rows),
                    p1_win_rate=wins['player1'] / completed if completed else None,
                    p2_win_rate=wins['player2'] / completed if completed else None,
                    draw_rate=wins['draw'] / completed if completed else None,
                    p1_reward_mean=sum(row['returns'][0] for row in self.rows) / count if count else None,
                    p2_reward_mean=sum(row['returns'][1] for row in self.rows) / count if count else None,
                    decisions_mean=sum(row['decisions'] for row in self.rows) / count if count else None)

    def add(self, event):
        self.rows.append({key: event[key] for key in
                          ('terminated', 'truncated', 'winner', 'returns', 'decisions')})
        if len(self.rows) < self.size:
            return None
        result = self.snapshot()
        self.last_complete = result
        self.finished += 1
        self.rows.clear()
        return result


def format_block(row):
    def percent(key):
        return f"{row[key]:.1%}" if row[key] is not None else 'n/a'
    state = '' if row['full'] else ' (partial)'
    return (f"BLOCK {row['block']}{state} episodes={row['first_episode']}–{row['last_episode']} | "
            f"p1_win={percent('p1_win_rate')} p2_win={percent('p2_win_rate')} "
            f"draw={percent('draw_rate')} completed={row['completed']} truncated={row['truncated']} | "
            f"p1_mean={row['p1_reward_mean']:.3f} p2_mean={row['p2_reward_mean']:.3f} "
            f"decisions_mean={row['decisions_mean']:.1f}")
