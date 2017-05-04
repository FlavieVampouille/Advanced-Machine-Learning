import sys

import pylab as plb
import numpy as np
import mountaincar


def update_phi (x,vx,s):
    return np.exp(-(x-s[0])**2) * np.exp(-(vx-s[1])**2)
    
def update_Q (Q,W,phi):
    for a in xrange(3):
        Q[a] = sum(W[a]*phi)
    return Q

class Agent():
    
    def __init__(self):
        
        self.reset()
        
        self.p = 150         # 150 -> move of unit 1 on x axis
        self.k = 40          # 40 -> move of unit 1 on vx axis

        self.eps = 0.01       # set epsilon for epsilon greedy action
        self.gamma = 1       # discount factor
        self.alpha = 0.1     # learning rate
        self.e = 0
        self.l = 0.8         # lambda
 
        self.s = np.zeros((2,self.p+1,self.k+1))
        for i in xrange(self.p+1):
            for j in xrange(self.k+1):
                self.s[0][i,j] , self.s[1][i,j] = -150 + i*150/self.p , -20 + j*40/self.k
        self.s = np.reshape(self.s,(2,(self.p+1)*(self.k+1)))  # linear form speed up calculations

        self.phi = np.zeros((self.p+1)*(self.k+1))
        self.Q = np.zeros(3)                            # 3 actions possibles
        self.W = np.zeros((3,(self.p+1)*(self.k+1)))    # 3 actions possibles

    def reset(self):
        self.first_visit = True
        self.action = np.random.randint(-1, 2)      # reset first action 

    def state(self):
        return self.state
                      
    def act(self):
        self.previous_action = self.action
        # explore randomly with probability epsilon
        if np.random.uniform() < self.eps :
            self.action = np.random.randint(-1, 2)    # 3 actions possibles -1, 0 , 1
        # exploitation of knowledge with probability 1 - epsilon
        else :
            self.action = np.argmax(self.Q)-1
        return self.action            
        
    def update(self, next_state, reward):
        # update phi, Q, W
        if self.first_visit == False :
            self.previous_phi = self.phi
            self.phi = update_phi(next_state[0],next_state[1],self.s)
            self.previous_Q = self.Q
            self.Q = update_Q(self.Q,self.W,self.phi)
            self.delta = reward + self.gamma * max(self.Q) - self.previous_Q[self.previous_action+1]
            self.e = self.gamma * self.l * self.e + self.previous_phi
            self.W[self.action+1] += self.alpha * self.delta * self.e
        else :
            self.first_visit = False
        self.state = next_state
        
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

        for n in xrange(n_steps):
            
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
        for c_episodes in xrange(1, n_episodes):
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
            formating = "end of episode {0:3.0f} after {1:3.0f} steps,\
                           cumulative reward obtained: {2:1.2f}"
            print(formating.format(c_episodes, step-1, rewards[c_episodes]))
            sys.stdout.flush()
        return rewards


if __name__ == "__main__":
    # modify RandomAgent by your own agent with the parameters you want
    agent = Agent()
    test = Tester(agent)
    # you can (and probably will) change these values, to make your system
    # learn longer
    test.learn(2000, 5000)

    #print("End of learning, press Enter to visualize...")
    #raw_input()
    #test.visualize_trial()
    #plb.show()
