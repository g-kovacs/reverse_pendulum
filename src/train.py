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
    -b:     Batch size (power of 2)
    -s:     Total sample count

Models, registered from left to right in order of calling:

    --lstm [mem_size]:  LSTMModel with specified window size
    --cnn [mem_size]:   CNNModel with specified window size
    --simple:           SimpleAC model
    --simple2:          SimpleAC2 model
"""


def run(models, batch, sample):
    config = Models.ModelConfiguration(models, (batch, sample/batch))
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size)
    agent = A2CAgent(lr=1e-2)
    episodes, deaths = agent.train(env, config, batch, sample//batch)
    config.save()

    fig = plt.figure(figsize=(16.0, 8.0))
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
    if not os.path.exists(f'media/{config.label}'):
        os.makedirs(f'media/{config.label}')
    seconds, death_list = env.test(
        config.get(), False, f'media/{config.label}/test.gif')
    print(f'Alive for {int(seconds)} seconds')
    print('Died:')
    print(death_list)
    env.close()
    plt.savefig(f'media/{config.label}/train.png', bbox_inches='tight')
    plt.show()


def main(argv):
    mpl.rcParams['toolbar'] = 'None'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    batch_size = 128
    sample_count = 32000
    models = []

    try:
        opts, args = getopt.getopt(
            argv, 'hgb:s:', ['lstm=', 'cnn=', 'simple', 'simple2', 'rnn=', 'gru='])
    except getopt.GetoptError:
        print(helpMSG)
        sys.exit(2)
    try:
        for opt, arg in opts:
            if opt == "-h":
                print(helpMSG)
                sys.exit()
            elif opt == "-g":
                os.environ.pop('CUDA_VISIBLE_DEVICES')
            elif opt == "-b":
                batch_size = int(arg)
            elif opt == '-s':
                sample_count = int(arg)
            elif opt == '--lstm':
                models.append(Models.LSTMModel(
                    DCPEnv.actions_size, memory_size=(int(arg) if arg else 8)))
            elif opt == '--cnn':
                models.append(Models.CNNModel(
                    DCPEnv.actions_size, memory_size=(int(arg) if arg else 8)))
            elif opt == '--simple':
                models.append(Models.SimpleAC(DCPEnv.actions_size))
            elif opt == '--simple2':
                models.append(Models.SimpleAC2(DCPEnv.actions_size))
            elif opt == '--gru':
                models.append(Models.GRUModel(DCPEnv.actions_size,
                              memory_size=(int(arg) if arg else 8)))
            elif opt == '--rnn':
                models.append(Models.RNNModel(DCPEnv.actions_size,
                              memory_size=(int(arg) if arg else 8)))
    except Exception:
        print(helpMSG)
        sys.exit(2)

    if len(models) == 0:
        print('No models given!')
        print(helpMSG)
        sys.exit(2)

    run(models, batch_size, sample_count)


if __name__ == "__main__":
    main(sys.argv[1:])
