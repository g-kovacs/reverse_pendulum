from DCPEnv import DCPEnv
from Models import ModelConfiguration


def main():
    config = ModelConfiguration.load('GRUvsCNN_8_4_0.001x2')
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size)
    for i in range(1):
        print("Alive for %.1f seconds" % (env.test(config.get(), True, f'media/{config.label}.gif')[0] / 10.0))
    env.close()
    config.clean()


if __name__ == "__main__":
    main()
