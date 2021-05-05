from tkinter import *

from flappy_agent import FlappyAgent
from flappy_env import Environment


class FlappyGUI:
    def __init__(self, agent: FlappyAgent, n_iterations, env, env_scaler=10, train_update_interval=50,
                 test_update_interval=200):
        self.env = env
        self.state = self.env.reset()
        self.observation = None

        self.env_scaler = env_scaler
        self.train_update_interval = train_update_interval
        self.test_update_interval = test_update_interval
        self.draw_epoch_interval = 500
        self.print_epoch_interval = 100

        self.agent = agent

        self.test = False
        self.n_iterations = n_iterations
        self.iteration = 0
        self.epoch = 0
        self.epoch_iteration = 0
        self.epoch_reward_sum = 0
        self.best_epoch_reward_sum = -1

        self.window = Tk()
        self.window.title("Flappy birds")

        self.canvas = Canvas(self.window, width=self.env.map_size.x * env_scaler,
                             height=self.env.map_size.y * env_scaler,
                             borderwidth=0,
                             highlightthickness=0, bg="gray")
        self.canvas.grid(column=0, row=1, columnspan=40)

        self.window.after(1000, self.loop())
        self.window.mainloop()

    def draw(self):
        self.canvas.delete('all')
        info = self.observation[3]
        bird = info['bird']
        bird_x = bird['pos'].x * self.env_scaler
        bird_y = bird['pos'].y * self.env_scaler
        bird_size = bird['size'] * self.env_scaler
        tubes = info['tubes']

        self.canvas.create_rectangle(bird_x, bird_y, bird_x + bird_size, bird_y + bird_size, fill='red')
        for tube in tubes['list']:
            self.canvas.create_rectangle(
                tube[0] * self.env_scaler,
                (tube[1] + tubes['gapsize']) * self.env_scaler,
                (tube[0] + tubes['width']) * self.env_scaler,
                self.env.map_size.y * self.env_scaler,
                fill='green')
            self.canvas.create_rectangle(
                tube[0] * self.env_scaler,
                0,
                (tube[0] + tubes['width']) * self.env_scaler,
                tube[1] * self.env_scaler,
                fill='yellow')

    def loop(self):
        if not self.test:
            action = self.agent.step(self.state)
            self.observation = self.env.step(action)
            next_state, reward, done, info = self.observation
            self.agent.learn(self.state, action, next_state, reward)

            self.state = next_state

            self.epoch_iteration += 1
            self.epoch_reward_sum += reward
            self.iteration += 1

            if (self.epoch % self.draw_epoch_interval) == 0:
                self.draw()
                call_latency = self.train_update_interval
            elif self.iteration < self.n_iterations:
                call_latency = 0
            else:
                self.test = True
                self.agent.train_end()
                call_latency = self.test_update_interval

            if done or self.iteration >= self.n_iterations:
                if (self.epoch % self.print_epoch_interval) == 0 or self.iteration >= self.n_iterations:
                    print('\nEpoch:', self.epoch)
                    print('Iteration: {} ({:.02f}%)'.format(self.iteration, 100 * self.iteration / self.n_iterations))
                    print('Aggregate reward:', self.epoch_reward_sum)

                self.agent.epoch_end(self.epoch_reward_sum)
                self.state = self.env.reset()
                self.epoch_reward_sum = 0
                self.epoch_iteration = 0
                self.epoch += 1

        else:
            action = self.agent.step(self.state)
            self.observation = self.env.step(action)
            next_state, reward, done, info = self.observation
            self.state = next_state

            self.epoch_iteration += 1
            self.epoch_reward_sum += reward

            if done:
                print('\nAggregate test reward:', self.epoch_reward_sum)
                self.state = self.env.reset()
                self.epoch_reward_sum = 0
                self.epoch_iteration = 0

            self.draw()
            call_latency = self.test_update_interval

        self.window.after(call_latency, self.loop)


if __name__ == '__main__':
    env_scaler = 10
    env = Environment()

    n_iterations = int(2e5)

    agent = FlappyAgent(observation_space_size=env.observation_space_size,
                        action_space=env.action_space,
                        n_iterations=n_iterations)
    gui = FlappyGUI(agent=agent,
                    n_iterations=n_iterations,
                    env=env,
                    env_scaler=10,
                    train_update_interval=25,
                    test_update_interval=50)
