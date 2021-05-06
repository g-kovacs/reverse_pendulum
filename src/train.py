from matplotlib import pyplot as plt

from DCPEnv import DCPEnv
from SimpleModel import SimpleModel
from agent import A2CAgent


def run():
    env = DCPEnv()
    model = SimpleModel(num_actions=env.action_space.n)
    agent = A2CAgent(model)
    rewards_history = agent.train(env, 32, 128)
    plt.plot(rewards_history)
    plt.show()
    print("Finished training, testing...")
    print(f'Test result: {env.test(model, False) / 10.0}')
    model.save_weights('saves/simpleModel_1.0')

if __name__ == "__main__":
    run()
