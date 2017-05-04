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


""" To play only on legal moves (OpenAI Gym interraction) """
def _pass_action(board_size):
	return board_size**2
def _resign_action(board_size):
	return board_size**2 + 1
def _coord_to_action(board, c):
	'''Converts Pachi coordinates to actions'''
	if c == pachi_py.PASS_COORD: return _pass_action(board.size)
	if c == pachi_py.RESIGN_COORD: return _resign_action(board.size)
	i, j = board.coord_to_ij(c)
	return i*board.size + j
def _action_to_coord(board, a):
	'''Converts actions to Pachi coordinates'''
	if a == _pass_action(board.size): return pachi_py.PASS_COORD
	if a == _resign_action(board.size): return pachi_py.RESIGN_COORD
	return board.ij_to_coord(a // board.size, a % board.size)
def str_to_action(board, s):
	return _coord_to_action(board, board.str_to_coord(s.encode()))
	

class RandomAgent:
	"""
	Wise random agent who only play legal moves
	"""
	def __init__(self):
		pass

	def getAction(self):
		b = env.state.board
		legal_coords = b.get_legal_coords(env.state.color)
		action = _coord_to_action(b, env.np_random.choice(legal_coords))
		return action
		

agent = RandomAgent()

for i_episode in range(2000):
	
	env.illegal_move_mode = "lose"
	observation = env.reset()
	
	for t in range(200):
		
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
