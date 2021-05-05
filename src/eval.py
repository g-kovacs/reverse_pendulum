from doubleCartPoleEnv import DCPEnv
import agent as ag
from tensorflow.keras import models

def main():
    env = DCPEnv()
    model = ag.Model(num_actions=env.action_space.n)
    agent = ag.A2CAgent(model)
    model.load_weights('saves/model_1.0')
    for i in range(4):
        print("Alive for %.1f seconds" % (agent.test(env, True) / 10.0))


if __name__ == "__main__":
    main()