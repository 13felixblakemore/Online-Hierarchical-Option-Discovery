import copy
import json
import unittest

import numpy as np

from agents import Transition
from experiment import make_agent, make_env
from macro_actions import MacroActionAgent, replace_pair


class MacroActionTests(unittest.TestCase):
    def add(self, agent, actions):
        path = tuple(Transition(i,a,float(i == len(actions)-1),i+1,
                                terminated=i == len(actions)-1) for i,a in enumerate(actions))
        agent.successful_trajectories.clear()
        agent.successful_trajectories.extend([path]*3)
        return agent.discover_macros()

    def test_nonoverlapping_replacement_and_episode_boundaries(self):
        self.assertEqual(replace_pair([0,0,0,0],(0,0),2),([2,2],2))
        agent = MacroActionAgent(10,2)
        agent.successful_trajectories.extend([(Transition(0,0,0,1),)]*10)
        self.assertIsNone(agent.discover_macros())
        agent.successful_trajectories.clear()
        agent.successful_trajectories.append(tuple(Transition(i,0,0,i+1) for i in range(4)))
        # Three overlapping digrams, but only two nonoverlapping replacements.
        self.assertIsNone(agent.discover_macros())

    def test_recursive_rules_and_global_open_loop_execution(self):
        agent = MacroActionAgent(20,2,epsilon=0,manager_hindsight=False)
        self.assertEqual(self.add(agent,[0,1,0,1]),0)
        self.assertEqual(self.add(agent,[0,1,0,1]),1)
        self.assertEqual(agent.macros[0].rule,(0,1))
        self.assertEqual(agent.macros[1].rule,(2,2))
        self.assertEqual(agent.macros[1].primitive_actions,(0,1,0,1))
        self.assertEqual(agent.macros[1].depth,2)
        agent.q[10,3] = 10
        agent.learning_enabled = False
        actions = []
        for state in range(10,14):
            action = agent.select_action(state)
            actions.append(action)
            agent.update(state,action,0,state+1,False)
        self.assertEqual(actions,[0,1,0,1])
        self.assertEqual(agent.macros[1].execution_lengths[4],1)
        self.assertEqual(agent.macros[0].expansions,2)
        self.assertIsNone(agent._choice)

    def test_macro_return_and_primitive_learning_inside_macro(self):
        agent = MacroActionAgent(5,2,epsilon=0,alpha=1,gamma=.5,manager_hindsight=False)
        self.add(agent,[0,1])
        agent.q[0,2] = 10
        agent.q[2,:] = 100
        for state,action,reward,terminal in [(0,0,2,False),(1,1,4,True)]:
            self.assertEqual(agent.select_action(state),action)
            agent.update(state,action,reward,state+1,terminal)
        self.assertEqual(agent.q[0,2],4)
        self.assertEqual(agent.q[1,1],4)
        self.assertEqual(agent.primitive_updates,2)
        self.assertEqual(agent.manager_macro_updates,1)

    def test_time_limit_bootstraps_and_clears_execution(self):
        agent = MacroActionAgent(5,2,epsilon=0,alpha=1,gamma=.5,manager_hindsight=False)
        self.add(agent,[0,1])
        agent.q[0,2] = 10
        agent.q[1,:] = 6
        agent.select_action(0)
        agent.update(0,0,1,1,False,truncated=True)
        self.assertEqual(agent.q[0,2],4)
        self.assertEqual(agent._pending,[])
        self.assertEqual(len(agent.successful_trajectories),3)
        self.assertEqual(len(agent.replay_trajectories),1)

    def test_har_matches_nonselected_macro_and_terminal_prefix(self):
        agent = MacroActionAgent(5,2,epsilon=0,alpha=1,gamma=.5,manager_hindsight=True)
        self.add(agent,[0,1])
        agent.q[0,0] = 10
        agent.q[1,1] = 10
        for state,action,reward,terminal in [(0,0,2,False),(1,1,4,True)]:
            self.assertEqual(agent.select_action(state),action)
            agent.update(state,action,reward,state+1,terminal)
        self.assertEqual(agent.q[0,2],4)
        self.assertEqual(agent.online_hindsight_updates,1)
        # A goal interrupts a macro after a matching first primitive.
        agent.q[3,0] = 10
        agent.select_action(3)
        agent.update(3,0,7,4,True)
        self.assertEqual(agent.q[3,2],7)
        self.assertEqual(agent.online_hindsight_updates,2)

    def test_discovery_replay_limits_reward_context_and_rng(self):
        agent = MacroActionAgent(10,2,alpha=1,gamma=.5,manager_hindsight=True,hindsight_replay_limit=1)
        agent.replay_trajectories.extend([
            (Transition(0,0,2,1),Transition(1,1,4,2,terminated=True)),
            (Transition(5,0,6,6),Transition(6,1,8,7,terminated=True)),
        ])
        state = copy.deepcopy(agent.rng.bit_generator.state)
        self.add(agent,[0,1])
        self.assertEqual(agent.discovery_hindsight_updates,1)
        self.assertEqual(agent.q[5,2],10)
        self.assertEqual(agent.q[0,2],0)
        self.assertEqual(agent.environment_steps,0)
        self.assertEqual(agent.macros[0].selections,0)
        self.assertEqual(agent.rng.bit_generator.state,state)

    def test_selected_macro_is_not_double_updated_by_har(self):
        agent = MacroActionAgent(5,2,epsilon=0,manager_hindsight=True)
        self.add(agent,[0,1])
        agent.q[0,2] = 10
        for state,action in ((0,0),(1,1)):
            self.assertEqual(agent.select_action(state),action)
            agent.update(state,action,float(state == 1),state+1,state == 1)
        self.assertEqual(agent.manager_macro_updates,1)
        self.assertEqual(agent.online_hindsight_updates,0)

    def test_replay_rejects_wrong_actions_and_nonterminal_incomplete_prefixes(self):
        agent = MacroActionAgent(5,2,alpha=1,manager_hindsight=False)
        self.add(agent,[0,1])
        agent.replay_trajectories.append((Transition(0,1,2,1),Transition(1,1,4,2,terminated=True)))
        agent.replay_trajectories.append((Transition(0,0,3,1),))
        self.assertEqual(agent.replay_new_macro(0),0)
        self.assertEqual(agent.q[0,2],0)
        agent.replay_trajectories.append((Transition(0,0,7,1,terminated=True),))
        self.assertEqual(agent.replay_new_macro(0),1)
        self.assertEqual(agent.q[0,2],7)

    def test_online_har_never_matches_across_episode_boundaries(self):
        agent = MacroActionAgent(5,2,alpha=1,epsilon=0,manager_hindsight=True)
        self.add(agent,[0,1])
        agent.q[0,0] = 10
        self.assertEqual(agent.select_action(0),0)
        agent.update(0,0,0,1,False,truncated=True)
        agent.q[1,1] = 10
        self.assertEqual(agent.select_action(1),1)
        agent.update(1,1,1,2,True)
        self.assertEqual(agent.online_hindsight_updates,0)

    def test_frozen_evaluation_and_factory(self):
        env = make_env(dict(random_start=True,random_goal=True,observe_goal=True))
        agent = make_agent(dict(type='macro',manager_hindsight=True),env,0)
        self.assertEqual(agent.q.shape,(10816,4))
        self.add(agent,[0,1,0,1])
        self.add(agent,[0,1,0,1])
        json.dumps(agent.snapshot())  # Gymnasium exposes a NumPy action-space size.
        agent.macros.clear()
        agent.q = agent.q[:,:4]
        agent.learning_enabled = False
        before = agent.q.copy()
        state,_ = env.reset(seed=0)
        action = agent.select_action(state)
        next_state,reward,terminal,truncated,_ = env.step(action)
        agent.update(state,action,reward,next_state,terminal,True)
        np.testing.assert_array_equal(agent.q,before)
        self.assertEqual(agent.hindsight_updates,0)
        self.assertFalse(agent.replay_trajectories)
        self.assertFalse(agent.macros)
        env.close()

    def test_length_cap_and_invalid_parameters(self):
        agent = MacroActionAgent(10,2,macro_max_length=2)
        self.add(agent,[0,1,0,1])
        self.assertIsNone(self.add(agent,[0,1,0,1]))
        self.assertTrue(all(m.max_steps <= 2 for m in agent.macros))
        for settings in ({'max_options':-1},{'macro_max_length':0},{'buffer_size':True}):
            with self.assertRaises(ValueError):
                MacroActionAgent(10,2,**settings)
