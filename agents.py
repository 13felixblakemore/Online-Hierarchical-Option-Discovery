"""Tabular agents and online option discovery for discrete room environments.

Requires: numpy
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np


class RandomAgent:
    """Choose each action with equal probability."""

    def __init__(self, n_actions, seed=None):
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state):
        return int(self.rng.integers(self.n_actions))


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(
        self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1, seed=None
    ):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))

        # Break ties randomly so unvisited actions are treated equally.
        values = self.q[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, terminated):
        """Update Q(s, a) after a transition."""
        best_next_q = 0.0 if terminated else self.q[next_state].max()
        target = reward + self.gamma * best_next_q
        self.q[state, action] += self.alpha * (target - self.q[state, action])


@dataclass(frozen=True)
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    duration: int = 1
    terminated: bool = False


@dataclass(frozen=True)
class OptionCandidate:
    """Demonstrations sharing an initiation state and a target state."""

    start_state: int
    goal_state: int
    segments: tuple[tuple[Transition, ...], ...]


class StatePairDiscovery:
    """Rank two-step start/end state pairs by occurrence count.

    Different action sequences with the same endpoints belong to one group.
    Self-returning segments are excluded from this first version.
    """

    def __init__(self, min_occurrences=3):
        if min_occurrences < 1:
            raise ValueError("min_occurrences must be positive")
        self.min_occurrences = min_occurrences

    def discover(self, trajectories):
        groups = defaultdict(list)
        for trajectory in trajectories:
            for first, second in zip(trajectory, trajectory[1:]):
                if first.state != second.next_state:
                    groups[first.state, second.next_state].append((first, second))
        ranked = sorted(groups.items(), key=lambda item: -len(item[1]))
        return [
            OptionCandidate(start, goal, tuple(segments))
            for (start, goal), segments in ranked
            if len(segments) >= self.min_occurrences
        ]


class TabularOption:
    """Closed-loop behavioural cloning using state/action occurrence counts.

    Choose modal actions by default, or softmax over raw counts / temperature.
    Stop at the demonstrated goal, an unseen state, or the longest demo length.
    """

    def __init__(self, candidate, n_actions=None, options=(),
                 action_selection='most_frequent', temperature=1.0,
                 initiation='single_state'):
        self.validate_policy(action_selection, temperature)
        self.validate_initiation(initiation)
        self.initiation_mode = initiation
        self.action_selection = action_selection
        self.temperature = float(temperature)
        self.start_state = candidate.start_state
        self.goal_state = candidate.goal_state
        self.max_decisions = max(map(len, candidate.segments))
        self.max_steps = max(sum(t.duration for t in segment) for segment in candidate.segments)
        self.children = sorted({t.action - n_actions
                                for segment in candidate.segments for t in segment
                                if n_actions is not None and t.action >= n_actions})
        self.depth = 1 + max((options[i].depth for i in self.children), default=0)
        # A safe primitive-step bound, including paths mixed by cloning.
        if self.children:
            largest = max([1] + [options[i].max_steps for i in self.children])
            self.max_steps = max(self.max_steps, self.max_decisions * largest)
        self.action_counts = defaultdict(Counter)
        self.execution_lengths = Counter()
        for segment in candidate.segments:
            for transition in segment:
                self.action_counts[transition.state][transition.action] += 1
        self.initiation_states = frozenset(
            {self.start_state} if initiation == 'single_state'
            else set(self.action_counts) - {self.goal_state}
        )

    @staticmethod
    def validate_initiation(initiation):
        if initiation not in ('single_state', 'policy_support'):
            raise ValueError("option_initiation must be 'single_state' or 'policy_support'")

    def can_initiate(self, state):
        """Allow only configured starts with a defined first policy action."""
        return state in self.initiation_states

    @staticmethod
    def validate_policy(action_selection, temperature):
        if action_selection not in ('most_frequent', 'softmax'):
            raise ValueError("option_action_selection must be 'most_frequent' or 'softmax'")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError('option_temperature must be finite and positive')

    def action_probabilities(self, state):
        """Distribution over demonstrated actions; softmax uses raw counts / T."""
        counts = self.action_counts.get(state)
        if not counts:
            return {}
        actions = list(counts)
        values = np.array([counts[a] for a in actions], dtype=float)
        if self.action_selection == 'softmax':
            # Subtract the maximum before scaling to avoid exponential overflow.
            with np.errstate(over='ignore', under='ignore'):
                weights = np.exp((values - values.max()) / self.temperature)
        else:
            weights = (values == values.max()).astype(float)
        return dict(zip(actions, weights / weights.sum()))

    def select_action(self, state, rng):
        if self.action_selection == 'softmax':
            probabilities = self.action_probabilities(state)
            return int(rng.choice(list(probabilities), p=list(probabilities.values())))
        # Preserve the default's existing RNG sequence, including modal ties.
        counts = self.action_counts[state]
        best = max(counts.values())
        return int(rng.choice([a for a, count in counts.items() if count == best]))

    def finished(self, state, steps):
        return (state == self.goal_state or steps >= self.max_steps
                or state not in self.action_counts)


class RepetitionAgent(QLearningAgent):
    """Minimal online option discovery for discrete states and actions.

    The manager uses semi-Markov Q-learning over primitives and learned options.
    Discovery receives successful primitive trajectories, including transitions
    executed inside options when compose_options=False. With compose_options=True,
    discovery uses manager decisions, and option policies can call older options.
    Child options run to termination; a parent's target or step limit can interrupt
    them. References only point to older options, preventing recursive cycles.

    Call update after every environment step, passing truncated as well as
    terminated so option execution and episode buffers reset at time limits.
    A custom discovery object only needs discover(trajectories), returning
    ranked OptionCandidate objects. At most one new option is added per interval.

    Set prune_options=True to remove options unused for prune_after_steps
    primitive environment steps, checked at episode-end discovery intervals.
    Nested calls count as use. Dependencies of recently used options are kept.
    New options receive the same inactivity grace period from creation time.

    Set manager_hindsight=True to warm-start new manager option values with up
    to hindsight_replay_limit policy-compatible segments from a separate buffer
    of successful primitive trajectories. Rewards keep their original task meaning.
    Hindsight does not count as option use or consume environment steps.

    option_action_selection is 'most_frequent' (default) or 'softmax'.
    option_temperature scales raw action counts for softmax and must be positive.
    option_initiation is 'single_state' (default) or 'policy_support', which allows
    every demonstrated policy state except the option target. This does not supply
    actions at unseen states or remove task-goal context from state IDs.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, seed=None, discovery=None, discovery_interval=10,
                 min_occurrences=3, max_options=20, buffer_size=100,
                 compose_options=False, prune_options=False, prune_after_steps=10000,
                 manager_hindsight=False, hindsight_replay_limit=100,
                 option_action_selection='most_frequent', option_temperature=1.0,
                 option_initiation='single_state'):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, seed)
        TabularOption.validate_policy(option_action_selection, option_temperature)
        self.option_action_selection = option_action_selection
        self.option_temperature = option_temperature
        TabularOption.validate_initiation(option_initiation)
        self.option_initiation = option_initiation
        if discovery_interval < 1 or buffer_size < 1 or max_options < 0:
            raise ValueError("Invalid discovery interval, buffer size, or option limit")
        self.n_actions = int(n_actions)
        self.compose_options = compose_options
        if not isinstance(hindsight_replay_limit, int) or isinstance(hindsight_replay_limit, bool) or hindsight_replay_limit < 1:
            raise ValueError("hindsight_replay_limit must be a positive integer")
        self.manager_hindsight = manager_hindsight
        self.hindsight_replay_limit = hindsight_replay_limit
        self.hindsight_updates = 0
        self.successful_primitive_trajectories = deque(maxlen=buffer_size)
        self._primitive_trajectory = []
        if not isinstance(prune_after_steps, int) or isinstance(prune_after_steps, bool) or prune_after_steps < 1:
            raise ValueError("prune_after_steps must be a positive integer")
        self.prune_options = prune_options
        self.prune_after_steps = prune_after_steps
        self.environment_steps = 0
        self.pruning_history = []
        self.longest_option_execution = 0
        self._next_option_id = 1
        self._stack = []
        self.discovery = discovery if discovery is not None else StatePairDiscovery(min_occurrences)
        self.discovery_interval = discovery_interval
        self.max_options = max_options
        self.options = []
        self.successful_trajectories = deque(maxlen=buffer_size)
        self.episodes = 0
        self._trajectory = []
        self._choice = None
        self._start_state = None
        self._return = 0.0
        self._duration = 0

    def select_action(self, state, greedy=False):
        if self._choice is None:
            available = self._available_actions(state)
            if not greedy and self.rng.random() < self.epsilon:
                self._choice = int(self.rng.choice(available))
            else:
                values = self.q[state, available]
                best = np.flatnonzero(values == values.max())
                self._choice = available[int(self.rng.choice(best))]
            self._start_state = state
            if self._choice >= self.n_actions:
                self._touch_option(self._choice-self.n_actions)
        if self._choice < self.n_actions:
            return self._choice
        if self.compose_options:
            if not self._stack:
                self._stack.append(dict(index=self._choice-self.n_actions, steps=0, decisions=0))
            while True:
                frame = self._stack[-1]
                option = self.options[frame['index']]
                action = option.select_action(state, self.rng)
                frame['decisions'] += 1
                if action < self.n_actions:
                    return action
                self._touch_option(action-self.n_actions)
                self._stack.append(dict(index=action-self.n_actions, steps=0, decisions=0))
        return self.options[self._choice - self.n_actions].select_action(state, self.rng)

    def _available_actions(self, state):
        return list(range(self.n_actions)) + [
            self.n_actions + i for i, option in enumerate(self.options)
            if option.can_initiate(state)
        ]

    def _best_value(self, state):
        return self.q[state, self._available_actions(state)].max()

    def update(self, state, action, reward, next_state, terminated, truncated=False):
        if self._choice is None:
            raise RuntimeError("Call select_action before update")
        self.environment_steps += 1
        if self.manager_hindsight:
            self._primitive_trajectory.append(Transition(state, action, reward, next_state,
                                                         terminated=terminated))
        for index in {frame['index'] for frame in self._stack} | (
                {self._choice-self.n_actions} if self._choice >= self.n_actions else set()):
            self.options[index].last_used_step = self.environment_steps
        if not self.compose_options:
            self._trajectory.append(Transition(state, action, reward, next_state,
                                               terminated=terminated))
        self._return += self.gamma ** self._duration * reward
        self._duration += 1
        done = terminated or truncated
        finished = self._choice < self.n_actions or done
        if self.compose_options and self._stack:
            self._advance_option_stack(self._stack, next_state)
            finished = finished or not self._stack
        elif not finished:
            finished = self.options[self._choice - self.n_actions].finished(next_state, self._duration)
        if finished:
            if self._choice >= self.n_actions:
                self.options[self._choice-self.n_actions].execution_lengths[self._duration] += 1
                self.longest_option_execution = max(self.longest_option_execution, self._duration)
            if self.compose_options:
                self._trajectory.append(Transition(self._start_state, self._choice,
                                                    self._return, next_state, self._duration, terminated))
            self._stack = []
            # Time limits still bootstrap; only true terminal states have value zero.
            bootstrap = 0.0 if terminated else self._best_value(next_state)
            target = self._return + self.gamma ** self._duration * bootstrap
            index = self._start_state, self._choice
            self.q[index] += self.alpha * (target - self.q[index])
            self._choice = None
            self._return = 0.0
            self._duration = 0
        if done:
            # In Four Rooms, reaching a terminal state means reaching the goal.
            if terminated:
                self.successful_trajectories.append(tuple(self._trajectory))
                if self.manager_hindsight:
                    self.successful_primitive_trajectories.append(tuple(self._primitive_trajectory))
            self._trajectory = []
            self._primitive_trajectory = []
            self.episodes += 1
            if self.episodes % self.discovery_interval == 0:
                if self.prune_options:
                    self.prune_inactive_options()
                self.discover_options()

    def _advance_option_stack(self, stack, next_state):
        """Advance runtime or retrospective execution by one primitive step."""
        for frame in stack:
            frame['steps'] += 1
        for i, frame in enumerate(stack):
            option = self.options[frame['index']]
            if next_state == option.goal_state or frame['steps'] >= option.max_steps:
                del stack[i:]
                break
        # Coverage is checked at child boundaries, never during a child call.
        while stack:
            frame = stack[-1]
            option = self.options[frame['index']]
            if (frame['decisions'] >= option.max_decisions
                    or next_state not in option.action_counts):
                stack.pop()
            else:
                break

    def _hindsight_actions(self, state, frames):
        """Enumerate supported policy choices without sampling or changing usage."""
        index, steps, decisions = frames[-1]
        probabilities = self.options[index].action_probabilities(state)
        advanced = frames[:-1] + ((index, steps, decisions+1),)
        for action, probability in probabilities.items():
            if probability <= 0:
                continue
            if action < self.n_actions:
                yield action, advanced
            else:
                yield from self._hindsight_actions(
                    state, advanced + ((action-self.n_actions, 0, 0),))

    def _match_hindsight(self, index, trajectory, start):
        """Match a policy-supported primitive prefix through option termination.

        Modal ties are supported by default. Softmax supports every demonstrated
        action with nonzero numerical probability; replay is not importance weighted.
        Matching never assumes that a two-decision demonstration takes two
        primitive steps, and stops at the option's actual first termination.
        """
        option = self.options[index]
        configurations = {((index, 0, 0),)}
        total = 0.0
        for offset, transition in enumerate(trajectory[start:start+option.max_steps]):
            following = set()
            total += self.gamma ** offset * transition.reward
            for frames in configurations:
                for action, advanced in self._hindsight_actions(transition.state, frames):
                    if action != transition.action:
                        continue
                    stack = [dict(index=i, steps=s, decisions=d) for i, s, d in advanced]
                    self._advance_option_stack(stack, transition.next_state)
                    if not stack or transition.terminated:
                        return total, offset+1, transition.next_state, transition.terminated
                    following.add(tuple((f['index'], f['steps'], f['decisions']) for f in stack))
            if not following:
                return None
            configurations = following
        return None

    def replay_manager_hindsight(self, index):
        """Warm-start a new option using one bounded pass over successful replay.

        Only matching primitive prefixes receive semi-Markov Q updates. This is
        action relabelling, not goal relabelling. It performs no environment steps
        and does not count as option use. Existing primitive/child values stay fixed.
        """
        option = self.options[index]
        updates = 0
        for trajectory in reversed(self.successful_primitive_trajectories):
            for start, transition in enumerate(trajectory):
                if not option.can_initiate(transition.state):
                    continue
                match = self._match_hindsight(index, trajectory, start)
                if match is None:
                    continue
                reward, duration, next_state, terminated = match
                bootstrap = 0.0 if terminated else self._best_value(next_state)
                target = reward + self.gamma ** duration * bootstrap
                key = transition.state, self.n_actions+index
                self.q[key] += self.alpha * (target-self.q[key])
                updates += 1
                self.hindsight_updates += 1
                if updates >= self.hindsight_replay_limit:
                    return updates
        return updates

    def _touch_option(self, index):
        option = self.options[index]
        option.last_used_step = self.environment_steps
        option.selections += 1

    def prune_inactive_options(self):
        """Remove inactive options at episode boundaries and free capacity.

        Usage includes nested calls. Retain all dependencies of recently used
        options, even if a dependency has not itself run recently. Compact Q
        columns and remap policies/replay together; stable option_id values are
        separate from action-space indices. Replay is split at removed actions
        so discovery never joins transitions across a deleted temporal gap.
        """
        if self._choice is not None or self._trajectory:
            raise RuntimeError("Prune only at an episode boundary")
        keep = {i for i, o in enumerate(self.options)
                if self.environment_steps-o.last_used_step < self.prune_after_steps}
        pending = list(keep)
        while pending:
            for child in self.options[pending.pop()].children:
                if child not in keep:
                    keep.add(child)
                    pending.append(child)
        removed = set(range(len(self.options))) - keep
        if not removed:
            return []
        removed_ids = [self.options[i].option_id for i in sorted(removed)]
        ordered = sorted(keep)
        action_map = {a: a for a in range(self.n_actions)}
        action_map.update({self.n_actions+old: self.n_actions+new
                           for new, old in enumerate(ordered)})
        # Preserve only contiguous surviving fragments of successful replay.
        replay = deque(maxlen=self.successful_trajectories.maxlen)
        for trajectory in self.successful_trajectories:
            fragment = []
            for t in trajectory:
                if t.action not in action_map:
                    if fragment:
                        replay.append(tuple(fragment))
                    fragment = []
                else:
                    fragment.append(Transition(t.state, action_map[t.action],
                                               t.reward, t.next_state, t.duration, t.terminated))
            if fragment:
                replay.append(tuple(fragment))
        self.successful_trajectories = replay
        self.q = self.q[:, list(range(self.n_actions))
                        + [self.n_actions+i for i in ordered]]
        self.options = [self.options[i] for i in ordered]
        for option in self.options:
            option.children = [action_map[self.n_actions+i]-self.n_actions for i in option.children]
            option.action_counts = defaultdict(Counter, {
                state: Counter({action_map[a]: count for a, count in counts.items()})
                for state, counts in option.action_counts.items()
            })
        self.pruning_history.append(dict(step=self.environment_steps, option_ids=removed_ids))
        return removed_ids

    def discover_options(self):
        """Add the highest-ranked unseen candidate; optionally replay its value."""
        if len(self.options) >= self.max_options:
            return
        existing = {(o.start_state, o.goal_state) for o in self.options}
        for candidate in self.discovery.discover(self.successful_trajectories):
            if (candidate.start_state, candidate.goal_state) in existing:
                continue
            option = TabularOption(
                candidate, self.n_actions, self.options,
                action_selection=self.option_action_selection,
                temperature=self.option_temperature,
                initiation=self.option_initiation,
            )
            option.option_id = self._next_option_id
            self._next_option_id += 1
            option.created_step = option.last_used_step = self.environment_steps
            option.selections = 0
            self.options.append(option)
            self.q = np.column_stack((self.q, np.zeros(self.q.shape[0])))
            option.hindsight_updates = (self.replay_manager_hindsight(len(self.options)-1)
                                        if self.manager_hindsight else 0)
            break
