from DCPEnv import DCPEnv
from Models import ModelConfiguration
import os


def main():
    config = ModelConfiguration.load('GRUModel_1_mem8vsGRUModel_2_mem8.b8.u8000')
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size)
    save_path = f'media\{config.label}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(10):
        print("Alive for %.1f seconds" % (env.test(config.get(), gif_path=f'{save_path}\\run_{i}.gif')[0] / 10.0))
    env.close()
    config.clean()


if __name__ == "__main__":
    main()
