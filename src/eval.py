from DCPEnv import DCPEnv
from models import SimpleAC2 as Policy

def main():
    env = DCPEnv()
    model = Policy(num_actions=env.action_space.n)
    model.load_weights('saves/simpleModel_1.0')
    for i in range(4):
        print("Alive for %.1f seconds" % (env.test(model, True) / 10.0))

if __name__ == "__main__":
    main()