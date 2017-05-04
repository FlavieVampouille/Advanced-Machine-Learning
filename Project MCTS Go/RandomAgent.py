from __future__ import division
import gym
import pachi_py
import logging
import matplotlib.pyplot as plt
import pylab as pl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

""" import environment (OpenAI Gym interraction) """
env = gym.make("Go9x9-v0")
env.reset()

""" set the size of the board (OpenAI Gym interraction) """
env.board_size = 4

""" set opponent type from OpenAI (OpenAI Gym interraction)
comment the line below to play against the UCT agent of OpenAI Gym """
env.opponent = "random"

""" store rewards at each episode """
reward_vect = []
lost = 0
win = 0
tie = 0


class RandomAgent:
	"""
	brute random agent
	"""
	def __init__(self):
		pass

	def getAction(self):
		action = env.action_space.sample()
		return action
		

agent = RandomAgent()

for i_episode in range(2000):
	env.illegal_move_mode = "lose"
	observation = env.reset()
	
	for t in range(100):
		action = agent.getAction()
		observation, reward, done, info = env.step(action)
		
		if done:
			
			reward_vect.append(reward)
			if reward == 1:
				win += 1
			elif reward == -1:
				lost += 1
			else:
				tie += 1
				
			print("Episode finished after {} timesteps".format(t+1))
			break
		

print "win : {0} , lost : {1} , tie : {2}".format(win,lost,tie)
pl.ylim([-1.2,1.2])
plt.plot(range(len(reward_vect)),reward_vect,'ro')
plt.title('Reward over time for {3} episodes \n win : {0} , lost : {1} , tie : {2}'.format(win,lost,tie,win+lost+tie))
plt.show()
