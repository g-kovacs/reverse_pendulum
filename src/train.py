from doubleCartPoleEnv import DCPEnv
import agent as ag


def run():
    env = DCPEnv()
    model = ag.Model(num_actions=env.action_space.n)
    agent = ag.A2CAgent(model)
    rewards_history = agent.train(env, 32, 125)
    print("Finished training, testing...")
    agent.save('saves/model_1.0')


if __name__ == "__main__":
    run()
