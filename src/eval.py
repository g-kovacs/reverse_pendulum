from DCPEnv import DCPEnv
from Models import ModelConfiguration

def main():
    env = DCPEnv()
    config = ModelConfiguration.load()
    for i in range(5):
        print("Alive for %.1f seconds" % (env.test(config.get())[0] / 10.0))
    env.close()
    config.clean()

if __name__ == "__main__":
    main()
