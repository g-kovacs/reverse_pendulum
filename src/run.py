from doubleCartPoleEnv import DoubleCartPoleEnv
import agent as ag
import gym


def registerEnv():
    gym.envs.register(
        id='DoubleCartPoleEnv-v0',
        entry_point='gym.envs.classic_control:DoubleCartPoleEnv',
        max_episode_steps=1000,
    )


def run():
    env = DoubleCartPoleEnv()
    model = ag.Model(num_actions=env.action_space.n)
    agent = ag.A2CAgent(model)
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    for i in range(200):
        print("%d out of 200" % agent.test(env, True))


if __name__ == "__main__":
    # registerEnv()
    run()
