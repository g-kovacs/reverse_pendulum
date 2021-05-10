from matplotlib import pyplot as plt
import matplotlib as mpl
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
    config = Models.ModelConfiguration([
        #Models.SimpleAC(num_actions=DCPEnv.actions_size),
        Models.SimpleAC2(num_actions=DCPEnv.actions_size),
        Models.LSTMModel(num_actions=DCPEnv.actions_size),
    ])
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size)
    agent = A2CAgent()
    starttime = timer()
    episodes, deaths = agent.train(env, config, 4, 6)
    config.save()
    dt = timer() - starttime
    
    fig = plt.figure()
    if(len(deaths)>1):
        ax = fig.add_subplot(121)
        ax.set_title("Death Counts")
        ax.pie(list(deaths.values()),
                    labels=list(deaths.keys()),
                    explode=[0.1]*config.num,
                    shadow=True,
                    autopct=lambda p : f'{int(p * sum(deaths.values())/100)}')
        ax = fig.add_subplot(122)
    else:
        ax = fig.add_subplot(111)
    ax.set_title("Training history")
    ax.plot(episodes,'bo', markersize=2)
    ax.set(xlabel='episodes', ylabel='seconds')

    plt.draw()
    print(f"Finished training in {int(dt+1)} seconds, testing...")
    seconds, death_list = env.test(config.get(), True, 'media/AC2vsLSTM.gif')
    print(f'Alive for {int(seconds)} seconds')
    print('Died:')
    print(death_list)
    env.close()
    plt.savefig('media/training_history.png', bbox_inches='tight')
    plt.show()


def main(argv):
    mpl.rcParams['toolbar'] = 'None'

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
