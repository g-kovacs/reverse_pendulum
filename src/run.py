from doubleCartPoleEnv import DoubleCartPoleEnv
import agent as ag

def run():
    env = DoubleCartPoleEnv()
    model = ag.Model(num_actions=env.action_space.n)
    agent = ag.A2CAgent(model)
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    for i in range(30):
        print("%d out of 200" % agent.test(env, True))


if __name__ == "__main__":
    run()