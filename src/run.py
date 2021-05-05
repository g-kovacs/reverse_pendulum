from doubleCartPoleEnv import DCPEnv
import agent as ag

def run():
    env = DCPEnv()
    model = ag.Model(num_actions=env.action_space.n)
    agent = ag.A2CAgent(model)
    rewards_history = agent.train(env)
    print("Finished training, testing...")
    for i in range(30):
        print("Alive for %.1f seconds" % (agent.test(env, True) / 10.0))


if __name__ == "__main__":
    run()