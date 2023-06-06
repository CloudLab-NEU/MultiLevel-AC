
from multilevel_pg.multilevelpg.environment.my_env import My_env
from multilevel_pg.multilevelpg.trainer.bilevel_trainer import Bilevel_Trainer
from multilevel_pg.multilevelpg.logger.utils import set_logger
from multilevel_pg.multilevelpg.samplers.bilevel_q_pg_sampler import BiSampler
from multilevel_pg.multilevelpg.agents.agent_factory import *




# set_seed(0)

agent_setting = 'multilevel'
env_name = 'my_env'
suffix = f'{env_name}/{agent_setting}'

set_logger(suffix)

agent_num = 5
action_num = 2
batch_size = 512
training_steps = 30000
exploration_step = 500
hidden_layer_sizes = (20, 20)
max_replay_buffer_size = 10000


env = My_env(env_name, agent_num, action_num)

# for round in range(100):

agents = []

# for i in range(agent_num):
#     agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
#     agents.append(agent)

# agent_0 = get_leader_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_leader_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_0 = get_leader_q_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_follower_q_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_1 = get_follower_stochasitc_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_2 = get_follower_stochasitc_agent(env, 2, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_3 = get_follower_stochasitc_agent(env, 3, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
agent_4 = get_follower_stochasitc_agent(env, 4, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_0 = get_independent_q_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_independent_q_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)

# agent_0 = get_maddpg_agent(env, 0, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
# agent_1 = get_follower_deterministic_agent(env, 1, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)



agents.append(agent_0)
agents.append(agent_1)
agents.append(agent_2)
agents.append(agent_3)
agents.append(agent_4)


# sampler = MASampler(agent_num)
# sampler = BiPGSampler(agent_num)
# sampler = Bi_continuous_Sampler(agent_num)
sampler = BiSampler(agent_num)
sampler.initialize(env, agents)

trainer = Bilevel_Trainer(
    env=env, agents=agents, sampler=sampler,
    steps=training_steps, exploration_steps=exploration_step,
    extra_experiences=['target_actions_q_pg'], batch_size=batch_size
)

trainer.run()
