"""Recursive action grammar with tabular semi-Markov learning and action replay.

AG-RL-inspired adaptation, not a reproduction of its Atari implementation.
https://arxiv.org/abs/1910.02876 (Christodoulou et al., 2019).
"""

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from agents import QLearningAgent, Transition


@dataclass
class MacroAction:
    rule: tuple[int, int]
    primitive_actions: tuple[int, ...]
    depth: int
    created_step: int
    selections: int = 0
    expansions: int = 0
    execution_lengths: Counter = field(default_factory=Counter)
    start_states: Counter = field(default_factory=Counter)

    @property
    def max_steps(self):
        return len(self.primitive_actions)


def replace_pair(tokens, pair, symbol):
    """Replace nonoverlapping occurrences, never joining separate episodes."""
    result = []
    position = matches = 0
    while position < len(tokens):
        if position+1 < len(tokens) and (tokens[position], tokens[position+1]) == pair:
            result.append(symbol)
            position += 2
            matches += 1
        else:
            result.append(tokens[position])
            position += 1
    return result, matches


class MacroActionAgent(QLearningAgent):
    """Learn globally available, deterministic open-loop action sequences.

    Grammar induction compresses successful primitive histories using existing
    rules, then adds one frequent nonoverlapping digram per discovery interval.
    Rules only refer to primitives or older rules, so recursion is acyclic.
    The manager updates a selected macro over its actual primitive duration.
    Primitive Q-values learn on every real step, including steps inside macros.
    Optional HAR also updates matching macros online and replays a new macro
    against bounded stored primitive trajectories. Rewards/goals are never changed.
    """

    def __init__(self, n_states, n_actions, alpha=.1, gamma=.99, epsilon=.1,
                 seed=None, discovery_interval=10, min_occurrences=3,
                 max_options=20, buffer_size=100, manager_hindsight=True,
                 hindsight_replay_limit=100, macro_max_length=32):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, seed)
        for name, value in dict(discovery_interval=discovery_interval,
                                min_occurrences=min_occurrences, buffer_size=buffer_size,
                                hindsight_replay_limit=hindsight_replay_limit,
                                macro_max_length=macro_max_length).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(max_options, int) or isinstance(max_options, bool) or max_options < 0:
            raise ValueError('max_options must be a nonnegative integer')
        self.n_actions = int(n_actions)
        self.discovery_interval = discovery_interval
        self.min_occurrences = min_occurrences
        self.max_options = max_options
        self.macro_max_length = macro_max_length
        self.manager_hindsight = manager_hindsight
        self.hindsight_replay_limit = hindsight_replay_limit
        self.macros = []
        self.successful_trajectories = deque(maxlen=buffer_size)
        self.replay_trajectories = deque(maxlen=buffer_size)
        self._trajectory = []
        self._recent = deque(maxlen=macro_max_length)
        self._pending = []
        self._choice = None
        self._start_state = None
        self._return = 0.
        self._duration = 0
        self.environment_steps = self.episodes = 0
        self.hindsight_updates = self.online_hindsight_updates = self.discovery_hindsight_updates = 0
        self.primitive_updates = self.manager_macro_updates = 0
        self.longest_option_execution = 0
        self.learning_enabled = True

    @property
    def options(self):
        """Shared experiment metric: library size (these are macros, not policies)."""
        return self.macros

    @property
    def _next_option_id(self):
        return len(self.macros)+1

    def select_action(self, state, greedy=False):
        if self._choice is None:
            self._choice = super().select_action(state, greedy=greedy)
            self._start_state = state
            self._pending = [self._choice]
            if self._choice >= self.n_actions:
                macro = self.macros[self._choice-self.n_actions]
                macro.selections += 1
                macro.start_states[int(state)] += 1
        while self._pending:
            symbol = self._pending.pop()
            if symbol < self.n_actions:
                return symbol
            macro = self.macros[symbol-self.n_actions]
            macro.expansions += 1
            self._pending.extend(reversed(macro.rule))
        raise RuntimeError('Call update after each selected primitive action')

    def _macro_update(self, state, index, reward, duration, next_state, terminated):
        bootstrap = 0. if terminated else self.q[next_state].max()
        key = state, self.n_actions+index
        self.q[key] += self.alpha * (reward + self.gamma**duration * bootstrap - self.q[key])

    def update(self, state, action, reward, next_state, terminated, truncated=False):
        if self._choice is None:
            raise RuntimeError('Call select_action before update')
        self.environment_steps += 1
        done = terminated or truncated
        transition = Transition(state, action, reward, next_state, terminated=terminated)
        self._return += self.gamma**self._duration * reward
        self._duration += 1
        finished = not self._pending or done
        selected = self._choice
        selected_duration = self._duration
        if self.learning_enabled:
            self._trajectory.append(transition)
            self._recent.append(transition)
            # HAR direction 1: every primitive is learned, including inside macros.
            super().update(state, action, reward, next_state, terminated)
            self.primitive_updates += 1
        if finished and selected >= self.n_actions:
            index = selected-self.n_actions
            self.macros[index].execution_lengths[self._duration] += 1
            self.longest_option_execution = max(self.longest_option_execution,self._duration)
            if self.learning_enabled:
                self._macro_update(self._start_state,index,self._return,self._duration,next_state,terminated)
                self.manager_macro_updates += 1
        if self.learning_enabled and self.manager_hindsight:
            # HAR direction 2: action-sequence matches from any observed state/goal.
            self._replay_suffixes(terminated, selected if finished else None, selected_duration)
        if finished:
            self._choice = None
            self._pending = []
            self._return = 0.
            self._duration = 0
        if done:
            self.episodes += 1
            if self.learning_enabled:
                trajectory = tuple(self._trajectory)
                self.replay_trajectories.append(trajectory)
                if terminated:
                    self.successful_trajectories.append(trajectory)
            self._trajectory = []
            self._recent.clear()
            if self.learning_enabled and self.episodes % self.discovery_interval == 0:
                self.discover_macros()

    def _replay_suffixes(self, terminated, selected, selected_duration):
        history = tuple(self._recent)
        actions = tuple(t.action for t in history)
        for index, macro in enumerate(self.macros):
            maximum = min(len(actions), macro.max_steps)
            lengths = range(maximum,0,-1) if terminated else (macro.max_steps,)
            for duration in lengths:
                if duration > len(actions) or actions[-duration:] != macro.primitive_actions[:duration]:
                    continue
                # Do not update the exact selected-macro transition twice.
                if selected != self.n_actions+index or selected_duration != duration:
                    segment = history[-duration:]
                    total = sum(self.gamma**i*t.reward for i,t in enumerate(segment))
                    self._macro_update(segment[0].state,index,total,duration,
                                       segment[-1].next_state,segment[-1].terminated)
                    self.hindsight_updates += 1
                    self.online_hindsight_updates += 1
                break  # At most the longest compatible suffix per macro/step.

    def _expanded(self, symbol):
        return (symbol,) if symbol < self.n_actions else self.macros[symbol-self.n_actions].primitive_actions

    def discover_macros(self):
        if not self.learning_enabled or len(self.macros) >= self.max_options:
            return None
        corpus = [[t.action for t in trajectory] for trajectory in self.successful_trajectories]
        for index, macro in enumerate(self.macros):
            corpus = [replace_pair(tokens,macro.rule,self.n_actions+index)[0] for tokens in corpus]
        counts = Counter(pair for tokens in corpus for pair in zip(tokens,tokens[1:]))
        existing = {macro.primitive_actions for macro in self.macros}
        for pair, count in counts.most_common():
            if count < self.min_occurrences:
                break
            expanded = self._expanded(pair[0]) + self._expanded(pair[1])
            if len(expanded) > self.macro_max_length or expanded in existing:
                continue
            occurrences = sum(replace_pair(tokens,pair,-1)[1] for tokens in corpus)
            # Each replacement saves one token; a binary rule costs two RHS tokens.
            if occurrences < self.min_occurrences or occurrences <= 2:
                continue
            depth = 1+max((self.macros[s-self.n_actions].depth
                          for s in pair if s >= self.n_actions),default=0)
            macro = MacroAction(pair,expanded,depth,self.environment_steps)
            self.macros.append(macro)
            # Paper-inspired first-primitive value transfer; preserve existing columns.
            self.q = np.column_stack((self.q,self.q[:,expanded[0]].copy()))
            index = len(self.macros)-1
            if self.manager_hindsight:
                self.replay_new_macro(index)
            return index
        return None

    def replay_new_macro(self, index):
        """One bounded pass, matching full expansions or goal-interrupted prefixes."""
        macro = self.macros[index]
        updates = 0
        for trajectory in reversed(self.replay_trajectories):
            for start in range(len(trajectory)):
                segment = trajectory[start:start+macro.max_steps]
                actions = tuple(t.action for t in segment)
                if (not segment or actions != macro.primitive_actions[:len(segment)]
                        or (len(segment) < macro.max_steps and not segment[-1].terminated)):
                    continue
                total = sum(self.gamma**i*t.reward for i,t in enumerate(segment))
                self._macro_update(segment[0].state,index,total,len(segment),
                                   segment[-1].next_state,segment[-1].terminated)
                self.hindsight_updates += 1
                self.discovery_hindsight_updates += 1
                updates += 1
                if updates >= self.hindsight_replay_limit:
                    return updates
        return updates

    def snapshot(self):
        return dict(environment_steps=self.environment_steps,episodes=self.episodes,
                    hindsight_updates=self.hindsight_updates,
                    online_hindsight_updates=self.online_hindsight_updates,
                    discovery_hindsight_updates=self.discovery_hindsight_updates,
                    primitive_updates=self.primitive_updates,manager_macro_updates=self.manager_macro_updates,
                    macros=[dict(action=self.n_actions+i,rule=m.rule,primitive_actions=m.primitive_actions,
                                 depth=m.depth,created_step=m.created_step,selections=m.selections,
                                 expansions=m.expansions,execution_lengths=dict(m.execution_lengths),
                                 start_states=dict(m.start_states)) for i,m in enumerate(self.macros)])
