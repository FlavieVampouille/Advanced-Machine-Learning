from __future__ import division
import gym
import pachi_py
import logging
import matplotlib.pyplot as plt
import pylab as pl
import random
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

""" import environment (OpenAI Gym interraction) """
env = gym.make("Go9x9-v0")
env.reset()

""" set the size of the board (OpenAI Gym interraction) """
env.board_size = 3

""" set that illegal moves ends in lost """
env.illegal_move_mode = "lose"

""" set opponent type from OpenAI (OpenAI Gym interraction)
comment the line below to play against the UCT agent of OpenAI Gym """
env.opponent = "random"

""" store rewards at each episode """
reward_vect = []
lost = 0
win = 0
tie = 0

""" maximum computational budget """
budget = 10000

""" Constant in exploration term
large c increase exploration while small c increase exploitation """
# TO BE DEFINED
c = 0.5

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
		
		
class Node():
	
	def __init__(self, state=env.reset(), parent=None, terminal=False, fullyExpanded=False):
		self.visits = 10**(-100)
		self.reward = 0
		self.state = state
		self.children = {}
		self.parent = parent
		self.action = 0
		self.terminal = terminal
		self.fullyExpanded = fullyExpanded
		self.player = 0
		
	def add_children(self, action, next_state, done):
		"""
		Create a new node, if it already exist in the tree do nothing,
		else create a non terminal node with reward zero and visit 1
		"""
		NewNode = Node(next_state,self)
		key = np.array_str(NewNode.state.board.encode())
		if key in self.children.keys():
			""" should nerver happen (except one time for root node), used to debug """
			#print "Error: this child already exist"
			pass
		else:
			NewNode.fullyExpanded = done
			NewNode.terminal = True
			NewNode.action = action
			NewNode.state = next_state
			NewNode.reward = 0
			NewNode.visits = 10**(-100)
			self.children[key] = NewNode
			
RootNodeReward = []
def UCTSearch(agent,RootNode,budget):
	"""
	The UCT Search combine both the TreePolicy to select/expand a node to explore
	and the DefaultPolicy to run a simulated game from this node
	the result of the game is then backed up through the selected nodes
	"""
	i = 0
	while True:
		env.reset()
		NewNode = TreePolicy(RootNode)
		NewNode.terminal = False
		""" use the line below to check the actions taken on the board """
		#print NewNode.state.board
		delta = DefaultPolicy(agent,NewNode)
		BACKUP(NewNode,delta)
		plot_reward = []
		i += 1
		if i % 100 == 0:
			RootNodeReward.append(RootNode.reward / RootNode.visits)
		if i % 1000 == 0:
			print RootNode.reward / RootNode.visits
		if i == budget:
			print RootNode.reward / RootNode.visits
			break
	return RootNode
	
	
def TreePolicy(node):
	"""
	Select nodes from the nodes already contained within the search tree
	and expand one new node to explore
	"""
	while node.terminal == False and node.fullyExpanded == False:
		if node.visits < 2:
			Expand(node)
			node = random.choice(node.children.values())
		else:
			node = BestChild(node,c)
	return node


def Expand(node):
	"""
	all the children nodes are added to expand the tree,
	according to available actions at the current node
	"""
	env_save = node.state
	legal_coords = env_save.board.get_legal_coords(env_save.color)
	for legal_action in legal_coords:
		action = _coord_to_action(env_save.board,legal_action)
		node.add_children(action=action, next_state=env_save.act(action), done=env_save.act(action).board.is_terminal)


def BestChild(node,c):
	"""
	In the selection process choose the next child with a UCB policy
	which balanced eploitation and exploration
	"""
	bestscore = -1000
	bestchildren = []
	for child in node.children.values():
		exploit = child.reward / child.visits
		explore = np.sqrt (2*np.log(node.visits) / child.visits)
		score = exploit + c*explore
		if score == bestscore:
			bestchildren.append(child)
		if score > bestscore:
			bestchildren = [child]
			bestscore = score
	if len(bestchildren) == 0:
		""" Should never happen, used to debug """
		print "Error: no more child"
	elif len(bestchildren) == 1:
		""" if one best child pick it """
		return bestchildren[0]
	else:
		""" if several good chidren choose one randomly """
		return random.choice(bestchildren)


def DefaultPolicy(agent,node):
	"""
	Play out the domain from a given non-terminal state
	to produce a value estimate of the reward (simulation)
	"""
	done = node.fullyExpanded
	reward = node.reward
	env.state = node.state
	if done != True and node.state.color == 2:
		legal_coords = node.state.board.get_legal_coords(node.state.color)
		action = _coord_to_action(node.state.board,env.np_random.choice(legal_coords))
		env.state = node.state.act(action)
	while done != True:
		action = agent.getAction()
		observation, reward, done, info = env.step(action)
	return reward


def BACKUP(node,delta):
	"""
	The simulation result is backed up (ie backpropagated)
	through the selected nodes to update their statistics 
	"""
	while node != None:
		"""
		Each node holds two values:
		- the number of times it has been visited
		- the total reward of all playout that pass through this state
		reward/visits is an approximation of the node's game theoretic value 
		"""
		node.visits += 1
		node.reward += delta
		node = node.parent


if __name__ == "__main__":
	agent = RandomAgent()
	env.reset()
	RootNode = Node(env.state)
	UCT_Tree = UCTSearch(agent,RootNode,budget)
	
	"""
	plt.plot(range(len(RootNodeReward)),RootNodeReward,'ro')
	plt.title('Reward/Visits ratio on Root Node over iterations with c = 0.5')
	plt.show()
	"""
	
	""" Real games against OpenAI Gym """
	
	for i_episode in range(2000):
		env.reset()
		CurrentNode = UCT_Tree
	
		for t in range(500):
			
			CurrentChildren = [child for child in CurrentNode.children.values()]
			theoreticValue = np.copy(CurrentChildren)
			theoreticValue[:] = [child.reward/child.visits for child in theoreticValue]
			
			action = CurrentChildren[np.argmax(theoreticValue)].action
			observation, reward, done, info = env.step(action)
			
			CurrentNode = CurrentChildren[np.argmax(theoreticValue)].children[np.array_str(env.state.board.encode())]
			env.state = CurrentNode.state
		
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
	
