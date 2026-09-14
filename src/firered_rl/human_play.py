"""Inference-only PPO opponent for the native human-versus-AI launcher."""
import numpy as np

from firered_rl.linked_env import linked_observation, local_mon


class PPOOpponent:
    def __init__(self, model, *, stochastic=False):
        from firered_rl.ppo_features import policy_space
        import gymnasium as gym
        if model.observation_space != policy_space() or model.action_space != gym.spaces.Discrete(11):
            raise ValueError("checkpoint does not match the linked PPO observation/action spaces")
        self.model, self.stochastic = model, stochastic
        self.model.policy.set_training_mode(False)

    def choose(self, report):
        from firered_rl.ppo_features import policy_observation
        observation = policy_observation(linked_observation(report))
        mask = observation["action_mask"].astype(bool)
        if not mask.any():
            raise ValueError("AI has no legal action at this decision boundary")
        action, _ = self.model.predict(observation, action_masks=mask,
                                       deterministic=not self.stochastic)
        action = int(np.asarray(action).item())
        if not 0 <= action < len(mask) or not mask[action]:
            raise ValueError("checkpoint selected an unavailable AI action")
        return action


def play_battle(session, opponent, human_player, announce=print):
    """The human uses the GUI; only the opponent submits an automated choice."""
    if session.human_player != human_player:
        raise ValueError("session human assignment does not match the opponent runner")
    session.reset_pair()
    ai = 1-human_player
    species = session.reports[human_player]["party"][local_mon(session.reports[human_player])["party_slot"]-1]
    name = species.get("species_name", str(species.get("species_id", "?")))
    announce(f"READY: you are player {human_player+1} (port {session.runtime.ports[human_player]}), "
             f"leading {name}. The AI is player {ai+1}.\n"
             "Click your mGBA window and use your configured GBA controls.\n"
             "Choose moves and switches normally; advance your own battle text with A.\n"
             "The AI waits while you choose. Ctrl+C in this terminal stops the match.")
    returns = [0.0, 0.0]
    while True:
        action = opponent.choose(session.reports[ai])
        actions = [None, None]
        actions[ai] = action
        result = session.step_pair(actions)
        for i, row in enumerate(result):
            returns[i] += row[1]
        # Do not reveal the AI's pending choice before the human commits.
        announce(f"decision={session.steps} turn={result[ai][4]['turn_end']} "
                 f"status={result[ai][4]['reason']}")
        if result[ai][2] or result[ai][3]:
            return dict(human_player=human_player+1, ai_player=ai+1,
                        human_outcome=result[human_player][4]["outcome"],
                        ai_outcome=result[ai][4]["outcome"],
                        human_reward=returns[human_player], ai_reward=returns[ai],
                        terminated=bool(result[ai][2]), truncated=bool(result[ai][3]),
                        reason=result[ai][4]["reason"], decisions=session.steps)
