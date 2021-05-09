from matplotlib import pyplot as plt
from DCPEnv import DCPEnv
import Models as Models
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
    models = Models.ModelConfiguration([
        Models.SimpleAC2(num_actions=DCPEnv.actions_size),
    ])
    env = DCPEnv(numCars=models.num, buffer_size=models.window_size)
    agent = A2CAgent()
    starttime = timer()
    rewards_history = agent.train(env, models, 64, 100)
    dt = timer() - starttime
    plt.plot(list(rewards_history.values()), label=list(rewards_history.keys()))
    plt.draw()
    print(f"Finished training in {dt} seconds, testing...")
    seconds, death_list = env.test(models, False)
    print(f'Alive for {seconds} seconds')
    print('Died:')
    print(death_list)
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
