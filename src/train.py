from matplotlib import pyplot as plt
from DCPEnv import DCPEnv
import models as m
from agent import A2CAgent
import sys, getopt
from os import environ
from timeit import default_timer as timer

helpMSG = """
train.py usage
    -h:     Prints this message
    -g:     Use GPU for calculations (default is CPU)
"""


def run():
    env = DCPEnv()
    for gamma in [0.95, 0.99, 0.999]:
        models = []
        #models.append(m.SimpleAC(num_actions=env.action_space.n))
        models.append(m.SimpleAC2(num_actions=env.action_space.n,name=f'{gamma}'))
        #models.append(m.CNNModel(num_actions=env.action_space.n))
        #models.append(m.LSTMModel(num_actions=env.action_space.n,memory_size=4))
        agent = A2CAgent(gamma=gamma)
        for model in models:
            starttime = timer()
            rewards_history = agent.train(env, model, 64, 100)
            dt = timer() - starttime
            plt.plot(rewards_history, label=model.label)
            print(f'Finished training {model.label} in {int(dt)} seconds')
    plt.legend()
    plt.draw()
    print("Finished training, testing...")
    for model in models:
        print(f'Test result of {model.label}: {env.test(model, False) / 10.0}')
        model.save_weights(f'saves/{model.label}')
    env.close()
    plt.show()


def main(argv):
    environ['CUDA_VISIBLE_DEVICES'] = '-1'

    try:
        opts, args = getopt.getopt(argv, 'hg')
    except getopt.GetoptError:
        print(helpMSG)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(helpMSG)
            sys.exit()
        elif opt == "-g":
            environ.pop('CUDA_VISIBLE_DEVICES')
    run()


if __name__ == "__main__":
    main(sys.argv[1:])
