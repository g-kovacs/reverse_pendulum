from matplotlib import pyplot as plt

from DCPEnv import DCPEnv
from models import SimpleAC, SimpleAC2, CNNModel
from agent import A2CAgent

def run():
    env = DCPEnv()
    models = []
    models.append(SimpleAC(num_actions=env.action_space.n))
    models.append(SimpleAC2(num_actions=env.action_space.n))
    models.append(CNNModel(num_actions=env.action_space.n))
    agent = A2CAgent()
    for model in models:
        rewards_history = agent.train(env, model, 128, 500)
        plt.plot(rewards_history, label = model.label)
        print(f'Finished training, {model.label}')
    plt.legend()
    plt.show()
    print("Finished training, testing...")
    for model in models:
        print(f'Test result of {model.label}: {env.test(model, False) / 10.0}')
        model.save_weights(f'saves/{model.label}')

if __name__ == "__main__":
    run()
