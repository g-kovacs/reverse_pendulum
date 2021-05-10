from matplotlib import pyplot as plt
import matplotlib as mpl
from DCPEnv import DCPEnv
import Models
from agent import A2CAgent
import sys, getopt
import os
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
        Models.SimpleAC2(num_actions=DCPEnv.actions_size),
        #Models.LSTMModel(num_actions=DCPEnv.actions_size,memory_size=2),
    ], 'AC2vsAC2_64_0.01x2')
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size)
    agent = A2CAgent(lr=1e-2)
    starttime = timer()
    batch_size = 64
    sample_count = 128000
    episodes, deaths = agent.train(env, config, batch_size, sample_count//batch_size)
    config.save()
    dt = timer() - starttime
    
    fig = plt.figure()
    if(len(deaths)>1):
        spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1,4])
        ax = fig.add_subplot(spec[0])
        ax.set_title("Death Counts")
        ax.pie(list(deaths.values()),
                    labels=list(deaths.keys()),
                    explode=[0.1]*config.num,
                    shadow=True,
                    autopct=lambda p : f'{int(p * sum(deaths.values())/100)}')
        ax = fig.add_subplot(spec[1])
    else:
        ax = fig.add_subplot(111)
    ax.set_title("Training history")
    ax.plot(episodes,'bo', markersize=2)
    ax.set(xlabel='episodes', ylabel='seconds')

    plt.draw()
    print(f"Finished training in {int(dt+1)} seconds, testing...")
    if not os.path.exists('media'):
            os.makedirs('media')
    seconds, death_list = env.test(config.get(), True, f'media/{config.label}.gif')
    print(f'Alive for {int(seconds)} seconds')
    print('Died:')
    print(death_list)
    env.close()
    plt.savefig(f'media/{config.label}.png', bbox_inches='tight')
    plt.show()


def main(argv):
    mpl.rcParams['toolbar'] = 'None'

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
            os.environ.pop('CUDA_VISIBLE_DEVICES')
    run()


if __name__ == "__main__":
    main(sys.argv[1:])
