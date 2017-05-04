import sys

import pylab as plb
import numpy as np
import mountaincar


class RandomAgent():
    def __init__(self):
        """
        Initialize your internal state
        """
        pass

    def act(self):
        """
        Choose action depending on your internal state
        """
        return np.random.randint(-1, 2)

    def update(self, next_state, reward):
        """
        Update your internal state
        """
        pass

# implement your own agent here


# test class, you do not need to modify this class
class Tester:

    def __init__(self, agent):
        self.mountain_car = mountaincar.MountainCar()
        self.agent = agent

    def visualize_trial(self, n_steps=100):
        """
        Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            print("Enter to continue...")
            raw_input()

            sys.stdout.flush()

            reward = self.mountain_car.act(self.agent.act())
            self.agent.state = [self.mountain_car.x, self.mountain_car.vx]

            # update the visualization
            mv.update_figure()
            plb.draw()

            # check for rewards
            if reward > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, n_episodes, max_episode):
        """
        params:
            n_episodes: number of episodes to perform
            max_episode: maximum number of steps on one episode, 0 if unbounded
        """

        rewards = np.zeros(n_episodes)
        for c_episodes in range(1, n_episodes):
            self.mountain_car.reset()
            step = 1
            while step <= max_episode or max_episode <= 0:
                reward = self.mountain_car.act(self.agent.act())
                self.agent.update([self.mountain_car.x, self.mountain_car.vx],
                                  reward)
                rewards[c_episodes] += reward
                if reward > 0.:
                    break
                step += 1
            formating = "end of episode after {0:3.0f} steps,\
                           cumulative reward obtained: {1:1.2f}"
            print(formating.format(step-1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    # modify RandomAgent by your own agent with the parameters you want
    agent = RandomAgent()
    test = Tester(agent)
    # you can (and probably will) change these values, to make your system
    # learn longer
    test.learn(10, 100)

    print("End of learning, press Enter to visualize...")
    raw_input()
    test.visualize_trial()
    plb.show()
