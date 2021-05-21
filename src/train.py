from matplotlib import pyplot as plt
import matplotlib as mpl
from DCPEnv import DCPEnv
import Models
from agent import A2CAgent
import sys
import getopt
import os

helpMSG = """
train.py usage
    -h:     Prints this message
    -g:     Use GPU for calculations (default is CPU)
"""
batch_update = (512, 300)
cfg_name = '.'.join(
    ['AC2vsLSTM', 'b'+str(batch_update[0]), 'u'+str(batch_update[1])])


def run():
    config = Models.ModelConfiguration([
        # Models.SimpleAC(num_actions=DCPEnv.actions_size),
        #Models.SimpleAC2(num_actions=DCPEnv.actions_size),
        #Models.GRUModel(num_actions=DCPEnv.actions_size,memory_size=64),
        Models.GRUModel(num_actions=DCPEnv.actions_size,memory_size=4),
        #Models.GRUModel(num_actions=DCPEnv.actions_size,memory_size=2),
        #Models.CNNModel(num_actions=DCPEnv.actions_size,memory_size=4),
        #Models.LSTMModel(num_actions=DCPEnv.actions_size,memory_size=4),
    ], 'testdt')
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size, time_step=1e-2)
    agent = A2CAgent(lr=1e-2)
    batch_size = 16
    sample_count = 4000
    episodes, deaths = agent.train(env, config, batch_size, sample_count//batch_size)
    config.save()

    fig = plt.figure()
    if(len(deaths) > 1):
        spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 4])
        ax = fig.add_subplot(spec[0])
        ax.set_title("Death Counts")
        ax.pie(list(deaths.values()),
               labels=list(deaths.keys()),
               explode=[0.1]*config.num,
               shadow=True,
               autopct=lambda p: f'{int(p * sum(deaths.values())/100)}')
        ax = fig.add_subplot(spec[1])
    else:
        ax = fig.add_subplot(111)
    ax.set_title("Training history")
    ax.plot(episodes, 'bo', markersize=2)
    ax.set(xlabel='episodes', ylabel='seconds')

    plt.draw()
    if not os.path.exists('media'):
        os.makedirs('media')
    seconds, death_list = env.test(config.get(), False, 'media/test.gif')
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
