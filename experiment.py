"""Shared setup and output helpers for training and experiment entry points."""

import csv

from agents import QLearningAgent, RandomAgent, RepetitionAgent
from four_rooms import FourRoomsEnv
from nine_rooms import NineRoomsEnv
from macro_actions import MacroActionAgent


AGENT_LABELS = {
    'q_learning': 'Q-learning',
    'repetition': 'Two-step options',
    'repetition_pruned': 'Two-step + pruning',
    'composed': 'Recursive options',
    'composed_pruned': 'Recursive + pruning',
}
AGENT_LABELS.update({
    kind + '_hindsight': label + ' + hindsight'
    for kind, label in list(AGENT_LABELS.items()) if kind != 'q_learning'
})
AGENT_LABELS.update({
    kind + '_softmax': label + ' + softmax'
    for kind, label in list(AGENT_LABELS.items()) if kind != 'q_learning'
})
AGENT_LABELS['composed_pruned_hindsight_support'] = 'Recursive + pruning + hindsight + initiation set'
AGENT_LABELS['macro'] = 'Recursive macros (no macro HAR)'
AGENT_LABELS['macro_hindsight'] = 'Recursive macros + HAR'
AGENT_LABELS['dqn'] = 'DQN'
AGENT_LABELS['outcome_dqn'] = 'DQN + displacement options + deep BC + HAR'


def variant_settings(config, kind):
    """Apply a named comparison variant without modifying the source config."""
    if kind not in AGENT_LABELS:
        raise ValueError(f'Unknown experiment agent: {kind!r}')
    if kind in ('dqn', 'outcome_dqn'):
        return dict(config, type=kind)
    if kind in ('macro', 'macro_hindsight'):
        return dict(config,type='macro',manager_hindsight=kind.endswith('_hindsight'))
    policy_kind = kind.removesuffix('_support').removesuffix('_softmax')
    base = policy_kind.removesuffix('_hindsight')
    return dict(
        config,
        type='q_learning' if base == 'q_learning' else 'repetition',
        compose_options=base in ('composed', 'composed_pruned'),
        prune_options=base in ('repetition_pruned', 'composed_pruned'),
        manager_hindsight=policy_kind.endswith('_hindsight'),
        option_action_selection=('softmax' if kind.endswith('_softmax')
                                 else config.get('option_action_selection', 'most_frequent')),
        option_initiation=('policy_support' if kind.endswith('_support')
                           else config.get('option_initiation', 'single_state')),
    )


def make_env(config):
    """Build a room environment from YAML settings."""
    options = config.copy()
    kind = options.pop('type', 'four_rooms')
    environments = {'four_rooms': FourRoomsEnv, 'nine_rooms': NineRoomsEnv}
    if kind not in environments:
        raise ValueError(f'Unknown environment type: {kind!r}')
    for name in ('start', 'goal'):
        if name in options:
            options[name] = tuple(options[name])
    return environments[kind](**options)


def make_agent(config, env, seed):
    """Build an agent, passing only the settings supported by its algorithm."""
    kind = config['type']
    if kind in ('dqn', 'outcome_dqn'):
        from deep_agents import DQNAgent, OutcomeDQNAgent
        cls = DQNAgent if kind == 'dqn' else OutcomeDQNAgent
        settings = {k: v for k, v in config.items() if k not in ('type', 'max_options')}
        return cls(env, seed=seed, max_options=config.get('max_options', 12) if kind == 'outcome_dqn' else 0,
                   **settings)
    if kind == 'random':
        return RandomAgent(n_actions=env.action_space.n, seed=seed)
    parameters = ('alpha', 'gamma', 'epsilon')
    if kind == 'macro':
        parameters += ('discovery_interval','min_occurrences','max_options','buffer_size',
                       'manager_hindsight','hindsight_replay_limit','macro_max_length')
        agent_class = MacroActionAgent
    elif kind == 'repetition':
        parameters += (
            'discovery_interval', 'min_occurrences', 'max_options', 'buffer_size',
            'compose_options', 'prune_options', 'prune_after_steps',
            'manager_hindsight', 'hindsight_replay_limit',
            'option_action_selection', 'option_temperature',
            'option_initiation',
        )
        agent_class = RepetitionAgent
    elif kind == 'q_learning':
        agent_class = QLearningAgent
    else:
        raise ValueError(f'Unknown agent type: {kind!r}')
    return agent_class(
        n_states=env.observation_space.n, n_actions=env.action_space.n, seed=seed,
        **{name: config[name] for name in parameters if name in config},
    )


def update_agent(agent, state, action, reward, next_state, terminated, truncated):
    """Learn from one primitive step; time limits bootstrap but reset options."""
    from deep_agents import DQNAgent
    if isinstance(agent, (RepetitionAgent, MacroActionAgent, DQNAgent)):
        agent.update(state, action, reward, next_state, terminated, truncated)
    elif isinstance(agent, QLearningAgent):
        agent.update(state, action, reward, next_state, terminated)


def write_csv(path, rows):
    """Write nonempty row dictionaries, creating the output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
