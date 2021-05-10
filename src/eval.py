from DCPEnv import DCPEnv
from Models import ModelConfiguration


def main():
    config = ModelConfiguration.load('AC2vsLSTM')
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size)
    for i in range(5):
        print("Alive for %.1f seconds" % (env.test(config.get())[0] / 10.0))
    env.close()
    config.clean()


if __name__ == "__main__":
    main()
