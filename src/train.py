from matplotlib import pyplot as plt
from DCPEnv import DCPEnv
from models import SimpleAC, SimpleAC2, CNNModel
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
    models = []
    models.append(SimpleAC(num_actions=env.action_space.n))
    models.append(SimpleAC2(num_actions=env.action_space.n))
    models.append(CNNModel(num_actions=env.action_space.n))
    agent = A2CAgent()
    for model in models:
    	starttime = timer()
        rewards_history = agent.train(env, model, 128, 500)
        dt = timer() - starttime
        plt.plot(rewards_history, label = model.label)
        print(f'Finished training {model.label} in {int(dt)} seconds')
    plt.legend()
    plt.draw()
    print("Finished training, testing...")
    for model in models:
        print(f'Test result of {model.label}: {env.test(model, False) / 10.0}')
        model.save_weights(f'saves/{model.label}')
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
