


from multilevel_pg.multilevelpg.environment.my_env import MatrixGame
from multilevel_pg.multilevelpg.environment.grid_game import GridEnv
from multilevel_pg.multilevelpg.trainer.bilevel_trainer import Bilevel_Trainer
from multilevel_pg.multilevelpg.utils.random import set_seed
from multilevel_pg.multilevelpg.logger.utils import set_logger
from multilevel_pg.multilevelpg.samplers.sampler import MASampler
from multilevel_pg.multilevelpg.agents.bi_follower_pg import FollowerAgent
from multilevel_pg.multilevelpg.agents.bi_leader_pg import LeaderAgent
from multilevel_pg.multilevelpg.agents.maddpg import MADDPGAgent
from multilevel_pg.multilevelpg.policy.base_policy import StochasticMLPPolicy
from multilevel_pg.multilevelpg.value_functions import MLPValueFunction
from multilevel_pg.multilevelpg.replay_buffers import IndexedReplayBuffer
from multilevel_pg.multilevelpg.explorations.ou_exploration import OUExploration
from multilevel_pg.multilevelpg.policy import DeterministicMLPPolicy

import tensorflow as tf


def gambel_softmax(x):
    u = tf.random.uniform(tf.shape(x))
    return tf.nn.softmax(x - tf.math.log(-tf.math.log(u)), axis=-1)


def get_leader_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]

    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return LeaderAgent(
        env_specs=env.env_specs,
        policy=StochasticMLPPolicy(
            input_shapes=(env.num_state, ),
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            output_activation=gambel_softmax,
            # preprocessor='LSTM',
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(num_sample * 2 + 1, (env.env_specs.action_space.flat_dim,)),
            # input_shapes=(num_sample * 2 + 1, ),
            input_shapes=(env.num_state + env.action_num * 2,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.num_state,
                                          action_dim=env.action_num,
                                          opponent_action_dim=env.action_num,
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_follower_stochasitc_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return FollowerAgent(
        env_specs=env.env_specs,
        policy=StochasticMLPPolicy(
            input_shapes=(env.num_state + env.action_num, ), # 1 for action1, 1 for state
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            output_activation=gambel_softmax,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(1 + 1, (env.env_specs.action_space.flat_dim,)),
            input_shapes=(env.num_state + env.action_num * 2,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.num_state + env.action_num,
                                          action_dim=env.action_num,
                                          opponent_action_dim=env.action_num,
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_follower_deterministic_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return FollowerAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(env.action_num + 1, ), # 1 for action1, 1 for state
            output_shape=(env.action_num, ),
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            # input_shapes=(1 + 1, (env.env_specs.action_space.flat_dim,)),
            input_shapes=(2,),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=env.action_num + 1,
                                          action_dim=1,
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


def get_maddpg_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    # print(action_space.shape)
    # print(env.env_specs.action_space.shape)
    return MADDPGAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            # input_shapes=(observation_space.shape, ),
            input_shapes=(1,),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim,)),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )


set_seed(0)

agent_setting = 'bilevel'
game_name = 'climbing'
suffix = f'{game_name}/{agent_setting}'

set_logger(suffix)

agent_num = 2
action_num = 4
batch_size = 128
training_steps = 50000
exploration_step = 1000
hidden_layer_sizes = (30, 30, 30)
max_replay_buffer_size = 1e5

# env = MatrixGame(game_name, agent_num, action_num)
env = GridEnv()

agents = []

# for i in range(agent_num):
#     agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
#     agents.append(agent)

agent_0 = get_leader_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_1 = get_follower_stochasitc_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agents.append(agent_0)
agents.append(agent_1)

sampler = MASampler(agent_num)
sampler.initialize(env, agents)

trainer = Bilevel_Trainer(
    env=env, agents=agents, sampler=sampler,
    steps=training_steps, exploration_steps=exploration_step,
    extra_experiences=['target_actions'], batch_size=batch_size
)

trainer.run()
