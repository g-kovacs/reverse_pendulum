import env as environment
import agent as ag

def main(train = False):
    env = environment.DoubleCartPoleEnv()
    model = ag.Model(num_actions=env.action_space.n)
    agent = ag.A2CAgent(model)
    if train:
        rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("Starting infinite loop...")
    while True:
        agent.test(env)

if __name__ == "__main__":
    main(True)