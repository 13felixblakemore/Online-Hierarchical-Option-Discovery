"""NumPy deep Q networks and displacement options with neural cloning and HAR."""
from collections import defaultdict, deque, Counter
import copy
import numpy as np
from agents import RepetitionAgent, Transition


class MLP:
    """Two ReLU hidden layers, Adam, and either Huber or cross-entropy loss."""
    def __init__(self, inputs, outputs, rng, width=64):
        sizes = [inputs, width, width, outputs]
        self.p = []
        for a, b in zip(sizes, sizes[1:]):
            self.p.extend([rng.normal(0, np.sqrt(2/a), (a, b)).astype(np.float32),
                           np.zeros(b, np.float32)])
        self.m = [np.zeros_like(p) for p in self.p]
        self.v = [np.zeros_like(p) for p in self.p]
        self.t = 0

    def forward(self, x):
        h = [np.asarray(x, np.float32)]
        for i in (0, 2, 4):
            y = h[-1] @ self.p[i] + self.p[i+1]
            h.append(np.maximum(y, 0) if i < 4 else y)
        return h

    def predict(self, x):
        return self.forward(x)[-1]

    def fit(self, x, actions, targets=None, lr=.001):
        h = self.forward(x)
        y = h[-1]
        if targets is None:
            exp = np.exp(y-y.max(axis=1, keepdims=True))
            grad = exp/exp.sum(axis=1, keepdims=True)
            grad[np.arange(len(x)), actions] -= 1
        else:
            grad = np.zeros_like(y)
            grad[np.arange(len(x)), actions] = np.clip(
                y[np.arange(len(x)), actions]-targets, -1, 1)
        grad /= len(x)
        grads = [None]*6
        for layer in (2, 1, 0):
            i = 2*layer
            grads[i] = h[layer].T @ grad
            grads[i+1] = grad.sum(axis=0)
            if layer:
                grad = (grad @ self.p[i].T) * (h[layer] > 0)
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = .9*self.m[i]+.1*g
            self.v[i] = .999*self.v[i]+.001*g*g
            self.p[i] -= lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)


class Features:
    def __init__(self, env):
        self.positions = np.asarray(env.positions, dtype=np.float32)
        self.n = len(self.positions)
        self.observe_goal = env.observe_goal
        self.goal = env.position_to_state[env.goal]
        self.scale = env.size-1

    def cells(self, state):
        return int(state)//self.n if self.observe_goal else int(state)

    def xy(self, state):
        return self.positions[self.cells(state)]

    def __call__(self, states):
        states = np.asarray(states, dtype=int).reshape(-1)
        cells = states//self.n if self.observe_goal else states
        goals = states % self.n if self.observe_goal else np.full_like(states, self.goal)
        return np.concatenate([self.positions[cells], self.positions[goals]], axis=1)/self.scale


class DQNAgent:
    def __init__(self, env, seed=None, gamma=.99, epsilon=.2, learning_rate=.001,
                 max_options=0, batch_size=64, replay_size=50000, train_every=4,
                 target_interval=250, **kwargs):
        self.rng = np.random.default_rng(seed)
        self.features = Features(env)
        self.n_actions = env.action_space.n
        self.network = MLP(4, self.n_actions+max_options, self.rng)
        self.target = copy.deepcopy(self.network)
        self.replay = deque(maxlen=replay_size)
        self.gamma, self.epsilon, self.learning_rate = gamma, epsilon, learning_rate
        self.batch_size, self.train_every = batch_size, train_every
        self.target_interval = target_interval
        self.learning_enabled = True
        self.learn_steps = self.updates = 0

    def active_actions(self):
        return self.n_actions

    def select_action(self, state, greedy=False):
        n = self.active_actions()
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(n))
        values = self.network.predict(self.features([state]))[0, :n]
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def remember(self, state, action, reward, next_state, terminated, duration=1):
        if not self.learning_enabled:
            return
        self.replay.append((state, action, reward, next_state, terminated, duration))
        self.learn_steps += 1
        if len(self.replay) < self.batch_size or self.learn_steps % self.train_every:
            return
        batch = [self.replay[i] for i in self.rng.choice(len(self.replay), self.batch_size, replace=False)]
        s, a, r, ns, done, duration = map(np.asarray, zip(*batch))
        bootstrap = self.target.predict(self.features(ns))[:, :self.active_actions()].max(axis=1)
        targets = r + self.gamma**duration * (1-done)*bootstrap
        self.network.fit(self.features(s), a, targets, self.learning_rate)
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target = copy.deepcopy(self.network)

    def update(self, state, action, reward, next_state, terminated, truncated=False):
        self.remember(state, action, reward, next_state, terminated)


class NeuralOption:
    def __init__(self, outcome, segments, agent, epochs):
        self.outcome = tuple(outcome)
        self.features = agent.features
        self.children = sorted({t.action-agent.n_actions for seg in segments for t in seg
                                if t.action >= agent.n_actions})
        self.depth = 1+max((agent.options[i].depth for i in self.children), default=0)
        self.max_decisions = max(map(len, segments))
        self.max_steps = self.max_decisions*max([1]+[agent.options[i].max_steps for i in self.children])
        self.network = MLP(4, agent.n_actions+len(agent.options), agent.rng)
        transitions = [t for seg in segments for t in seg]
        x = self.features([t.state for t in transitions])
        a = np.array([t.action for t in transitions])
        for _ in range(epochs):
            ids = agent.rng.choice(len(a), min(256, len(a)), replace=False)
            self.network.fit(x[ids], a[ids])
        self.demonstrations = len(segments)
        self.clone_accuracy = float(np.mean(self.network.predict(x).argmax(axis=1) == a))
        self.execution_lengths = Counter()
        self.selections = 0
        self.last_used_step = self.created_step = agent.environment_steps

    def can_initiate(self, state):
        return True

    def select_action(self, state, rng):
        return int(self.network.predict(self.features([state]))[0].argmax())

    def action_probabilities(self, state):
        return {self.select_action(state, None): 1.0}


class OutcomeDQNAgent(RepetitionAgent, DQNAgent):
    """Reuse episode buffering, option execution and policy-compatible HAR.

    Outcomes are physical (delta row, delta column), never differences of IDs.
    Policies observe position and task goal; termination is relative to invocation.
    """
    def __init__(self, env, seed=None, max_options=12, bc_epochs=150, **kwargs):
        repetition_keys = ('gamma', 'epsilon', 'discovery_interval', 'min_occurrences',
                           'buffer_size', 'hindsight_replay_limit')
        RepetitionAgent.__init__(self, env.observation_space.n, env.action_space.n,
                                 seed=seed, max_options=max_options, compose_options=True,
                                 manager_hindsight=True,
                                 **{k:v for k,v in kwargs.items() if k in repetition_keys})
        DQNAgent.__init__(self, env, seed, max_options=max_options, **kwargs)
        self.bc_epochs = bc_epochs
        self.min_occurrences = kwargs.get('min_occurrences', 3)

    def active_actions(self):
        return self.n_actions+len(self.options)

    def _best_value(self, state):
        return float(self.target.predict(self.features([state]))[0, :self.active_actions()].max())

    def select_action(self, state, greedy=False):
        self.q[state, :self.active_actions()] = self.network.predict(self.features([state]))[0, :self.active_actions()]
        action = RepetitionAgent.select_action(self, state, greedy)
        for frame in self._stack:
            frame.setdefault('origin', self.features.xy(state).copy())
        return action

    def _advance_option_stack(self, stack, next_state):
        for frame in stack:
            frame['steps'] += 1
        for i, frame in enumerate(stack):
            option = self.options[frame['index']]
            displacement = self.features.xy(next_state)-frame['origin']
            if np.array_equal(displacement, option.outcome) or frame['steps'] >= option.max_steps:
                del stack[i:]
                break
        while stack and stack[-1]['decisions'] >= self.options[stack[-1]['index']].max_decisions:
            stack.pop()

    def update(self, state, action, reward, next_state, terminated, truncated=False):
        start, choice = self._start_state, self._choice
        total = self._return + self.gamma**self._duration*reward
        duration = self._duration+1
        RepetitionAgent.update(self, state, action, reward, next_state, terminated, truncated)
        if self._choice is None:
            self.remember(start, choice, total, next_state, terminated, duration)

    def _match_hindsight(self, index, trajectory, start):
        stack = [dict(index=index, steps=0, decisions=0,
                      origin=self.features.xy(trajectory[start].state).copy())]
        total = 0.
        for offset, t in enumerate(trajectory[start:start+self.options[index].max_steps]):
            while True:
                frame = stack[-1]
                a = self.options[frame['index']].select_action(t.state, None)
                frame['decisions'] += 1
                if a < self.n_actions:
                    break
                stack.append(dict(index=a-self.n_actions, steps=0, decisions=0,
                                  origin=self.features.xy(t.state).copy()))
            if a != t.action:
                return None
            total += self.gamma**offset*t.reward
            self._advance_option_stack(stack, t.next_state)
            if not stack or t.terminated:
                return total, offset+1, t.next_state, t.terminated
        return None

    def replay_manager_hindsight(self, index):
        updates = 0
        for trajectory in reversed(self.successful_primitive_trajectories):
            for start, t in enumerate(trajectory):
                match = self._match_hindsight(index, trajectory, start)
                if match is not None:
                    reward, duration, ns, done = match
                    self.remember(t.state, self.n_actions+index, reward, ns, done, duration)
                    updates += 1
                    self.hindsight_updates += 1
                    if updates >= self.hindsight_replay_limit:
                        return updates
        return updates

    def discover_options(self):
        if not self.learning_enabled or len(self.options) >= self.max_options:
            return
        groups = defaultdict(list)
        existing = {o.outcome for o in self.options}
        for trajectory in self.successful_trajectories:
            for a, b in zip(trajectory, trajectory[1:]):
                outcome = tuple(self.features.xy(b.next_state)-self.features.xy(a.state))
                if outcome != (0, 0) and outcome not in existing:
                    groups[outcome].append((a, b))
        if not groups:
            return
        outcome = max(groups, key=lambda k: len(groups[k]))
        if len(groups[outcome]) < self.min_occurrences:
            return
        option = NeuralOption(outcome, groups[outcome], self, self.bc_epochs)
        option.option_id = self._next_option_id
        self._next_option_id += 1
        self.options.append(option)
        self.q = np.column_stack([self.q, np.zeros(len(self.q))])
        option.hindsight_updates = self.replay_manager_hindsight(len(self.options)-1)
