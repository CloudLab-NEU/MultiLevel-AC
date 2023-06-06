import numpy as np
# from malib.spaces import Discrete, Box, MASpace, MAEnvSpec
# from malib.environments.base_game import BaseGame
# from malib.error import EnvironmentNotFound, WrongNumberOfAgent, WrongNumberOfAction, WrongActionInputLength

from multilevel_pg.multilevelpg.spaces import Discrete, Box, MASpace, MAEnvSpec
from multilevel_pg.multilevelpg.environment.base_game import  BaseGame
from multilevel_pg.multilevelpg.error import EnvironmentNotFound, WrongNumberOfAgent, WrongNumberOfAction, WrongActionInputLength
import jpype
import os



class My_env(BaseGame):
    def __init__(self, env_name, agent_num, action_num, payoff=None, repeated=False, max_step=25, memory=0, discrete_action=True, tuple_obs=False):
        self.env_name = env_name
        self.agent_num = agent_num
        self.action_num = action_num
        self.discrete_action = discrete_action
        self.tuple_obs = tuple_obs
        self.num_state = 1

        env_list = My_env.get_env_list()

        if not self.env_name in env_list:
            raise EnvironmentNotFound(f"The env {self.env_name} doesn't exists")

        expt_num_agent = env_list[self.env_name]['agent_num']
        expt_num_action = env_list[self.env_name]['action_num']

        if expt_num_agent != self.agent_num:
            raise WrongNumberOfAgent(f"The number of agent \
                required for {self.env_name} is {expt_num_agent}")

        if expt_num_action != self.action_num:
            raise WrongNumberOfAction(f"The number of action \
                required for {self.env_name} is {expt_num_action}")


        self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
        self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))

        if self.discrete_action:
            self.action_spaces = MASpace(tuple(Discrete(action_num) for _ in range(self.agent_num)))
            if memory == 0:
                self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))
            elif memory == 1:
                self.observation_spaces = MASpace(tuple(Discrete(5) for _ in range(self.agent_num)))
        else:
            self.action_range = [-1., 1.]
            self.action_spaces = MASpace(tuple(Box(low=-1., high=1., shape=(1,)) for _ in range(self.agent_num)))
            if memory == 0:
                self.observation_spaces = MASpace(tuple(Discrete(1) for _ in range(self.agent_num)))
            elif memory == 1:
                self.observation_spaces =  MASpace(tuple(Box(low=-1., high=1., shape=(12,)) for _ in range(self.agent_num)))

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

        self.t = 0
        self.repeated = repeated
        self.max_step = max_step
        self.memory = memory
        self.previous_action = 0
        self.previous_actions = []
        self.ep_rewards = np.zeros(2)
        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_env_list():
        return {
            'my_env': {'agent_num': 5, 'action_num': 2},
        }
    
    def get_payoff(self):
        jar_path = os.path.abspath('.') + '/getPayoff.jar'
        jpype.startJVM('usr/local/java/jdk.../jre/lib/amd64/server/libjvm.so', '-ea', '-Djava.class.path=%s' % jar_path)
        java_class = jpype.JClass('getPayoffClass')
        payoff = java_class.getPayoffFunc()
        self.payoff = payoff

    def get_rewards(self, actions):
        jar_path = os.path.abspath('.') + '/getReward.jar'
        jpype.startJVM('usr/local/java/jdk.../jre/lib/amd64/server/libjvm.so', '-ea', '-Djava.class.path=%s' % jar_path)
        java_class = jpype.JClass('getRewardClass')
        reward_n = java_class.getRewardFunc(actions)
        return reward_n

    def step(self, actions):
        if len(actions) != self.agent_num:
            raise WrongActionInputLength(f"Expected number of actions is {self.agent_num}")

        actions = np.array(actions).reshape((self.agent_num,))
        reward_n = self.get_rewards(actions)
        self.rewards = reward_n
        info = {}
        done_n = np.array([True] * self.agent_num)
        if self.repeated:
            done_n = np.array([False] * self.agent_num)
        self.t += 1
        if self.t >= self.max_step:
            done_n = np.array([True] * self.agent_num)

        state = [0] * (self.action_num * self.agent_num * (self.memory) + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0 and self.t > 0:
            # print('actions', actions)
            if self.discrete_action:
                state[actions[1] + 2 * actions[0] + 1] = 1
            else:
                state = actions

        # tuple for tublar case, which need a hashabe obersrvation
        if self.tuple_obs:
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])

        # for i in range(self.agent_num):
        #     state_n[i] = tuple(state_n[i][:])

        self.previous_actions.append(tuple(actions))
        self.ep_rewards += np.array(reward_n)
        # print(state_n, reward_n, done_n, info)
        return state_n, reward_n, done_n, info

    def reset(self):
        # print('reward,', self.ep_rewards / self.t)
        self.ep_rewards = np.zeros(2)
        self.t = 0
        self.previous_action = 0
        self.previous_actions = []
        state = [0] * (self.action_num * self.agent_num * (self.memory)  + 1)
        # state_n = [tuple(state) for _ in range(self.agent_num)]
        if self.memory > 0:
            state = [0., 0.]
        if self.tuple_obs:
            # print(self.agent_num)
            state_n = [tuple(state) for _ in range(self.agent_num)]
        else:
            state_n = np.array([state for _ in range(self.agent_num)])
        # print(state_n)

        return state_n

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def terminate(self):
        pass

    def get_joint_reward(self):
        return self.rewards

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Number of Action \n'.format(self.env_name, self.agent_num, self.action_num)
        content += 'Payoff Matrixs:\n\n'
        for i in range(self.agent_num):
            content += 'Agent {}, Payoff:\n {} \n\n'.format(i+1, str(self.payoff[i]))
        return content



  