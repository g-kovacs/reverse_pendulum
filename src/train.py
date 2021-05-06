from matplotlib import pyplot as plt
from DCPEnv import DCPEnv
from SimpleModel import SimpleModel
from agent import A2CAgent
import sys, getopt
from os import environ

helpMSG = """
train.py usage
    -h:     Prints this message
    -g:     Use GPU for calculations (default is CPU)
"""

def run():
    env = DCPEnv()
    model = SimpleModel(num_actions=env.action_space.n)
    agent = A2CAgent(model)
    rewards_history = agent.train(env, 32, 128)
    plt.plot(rewards_history)
    plt.ion()
    plt.show()
    print("Finished training, testing...")
    print(f'Test result: {env.test(model, False) / 10.0}')
    model.save_weights('saves/simpleModel_1.0')
    input("Press Enter to end session...")

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
