from DCPEnv import DCPEnv
from SimpleModel import SimpleModel

def main():
    env = DCPEnv()
    model = SimpleModel(num_actions=env.action_space.n)
    model.load_weights('saves/simpleModel_1.0').expect_partial()
    for i in range(4):
        print("Alive for %.1f seconds" % (env.test(model, True) / 10.0))
    env.close()

if __name__ == "__main__":
    main()