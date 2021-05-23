from DCPEnv import DCPEnv
from Models import ModelConfiguration
import os

cfgName = 'SimpleAC-1vsSimpleAC2-1_b16_u16000_t1.0E-1'

def main():
    config = ModelConfiguration.load(cfgName)
    env = DCPEnv(num_cars=config.num, buffer_size=config.window_size, time_step=config.timestep)
    save_path = f'media\{config.label}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(5):
        print("Alive for %.1f seconds" % (env.test(config.get(), False, gif_path=f'{save_path}\\run_{i}.gif', max_seconds=30)[0]))
    env.close()
    config.clean()


if __name__ == "__main__":
    main()
