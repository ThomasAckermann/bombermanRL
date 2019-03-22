import numpy as np
from time import sleep
# from tempfile import * 
from scipy.optimize import curve_fit
# import random
import scipy as sp
import scipy.optimize as opt
import sys 
import traceback

moves = np.array([[]])
last_len_q_data = 0

def gauss(x):
	return np.exp(-x**2)


def func(theta, data, y):
	q = theta[-1]
	for i in range(np.shape(data)[1]):
		q += theta[3*i]*sigmoid(theta[3*i + 1] * (data[:,i] - theta[3*i + 2]))
	f = y - q
	return f

def func_lin(theta, data, y):
	q = theta[-1]
	for i in range(np.shape(data)[1]):
		q += theta[i] * data[:,i]
	f = y - q
	return f


# def func(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15):
#	 (x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14) = X
#	 return a_1 + a_2*sigmoid(x_1) + a_3*sigmoid(x_2) + a_4*sigmoid(x_3) + a_5*sigmoid(x_4) + a_6*sigmoid(x_5) + a_7*sigmoid(x_6) + a_8*sigmoid(x_7) + a_9*sigmoid(x_8) + a_10*sigmoid(x_9) + a_11*sigmoid(x_10) + a_12*sigmoid(x_11) + a_13*sigmoid(x_12) + a_14*sigmoid(x_13) + a_15*sigmoid(x_14)



def func_curve(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15):
	(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14) = X
	return a_15 + a_1*x_1 + a_2*x_2 + a_3*x_3 + a_4*x_4 + a_5*x_5 + a_6*x_6 + a_7*x_7 + a_8*x_8 + a_9*x_9 + a_10*x_10 + a_11*x_11 + a_12*x_12 + a_13*x_13 + a_14*x_14


def positions(self):
	agent = self.game_state['self']
	agent_pos = np.array([agent[0], agent[1]])
	coins = np.array(self.game_state['coins'])
	if len(coins) == 0:
		return agent_pos, np.array([0,0]), np.array([0,0]) 
	else:
		dist = np.array([])   
		for i in range(len(coins)):
			dist = np.append(dist, np.linalg.norm(agent_pos - coins[i]))
		coin = coins[np.argmin(dist)]
		mean_coin = []
		
		for i in range(len(coins)):
			if dist[i] != 0:
				mean_coin.append((agent_pos - coins[i])/dist[i]**(3/2))
		mean_coin = np.sum(mean_coin, axis=0)
		if np.linalg.norm(mean_coin) != 0: 
			mean_coin = (mean_coin)/ np.linalg.norm(mean_coin) 
		# mean_coin[1] = -1 * mean_coin[1]
		# print(mean_coin)
		return agent_pos, np.array(coin), mean_coin

def difference(self):
	agent_pos, coin, mean_coin = positions(self) 
	arena = self.game_state['arena']
	diff = np.linalg.norm(agent_pos - coin)
	if diff != 0 :
		direction = (agent_pos - coin) / diff
	else:
		direction = np.array([0,0])
	
	return np.array([diff, *direction, *mean_coin])

def crate_positions(self):
	agent = self.game_state['self']
	agent_pos = np.array([agent[0], agent[1]])
	arena = np.array(self.game_state['arena'])
	crates = []
	for i in range(np.shape(arena)[0]):
		for j in range(np.shape(arena)[1]):
			if arena[i][j] == 1:
				crates.append([i,j])#, axis=0)
	crates = np.array(crates)   
	dist = np.array([])   
	for i in range(len(crates)):
		dist = np.append(dist, np.linalg.norm(agent_pos - crates[i]))
	crate = crates[np.argmin(dist)]
	mean_crate = []
	for i in range(len(crates)):
		if dist[i] != 0:
			mean_crate.append((agent_pos - crates[i])/dist[i]**(3/2))
	mean_crate = np.sum(mean_crate, axis=0)
	if np.linalg.norm(mean_crate) != 0: 
		mean_crate = (mean_crate)/ np.linalg.norm(mean_crate)   
	return agent_pos, np.array(crate), mean_crate

def crate_difference(self):
	agent_pos, crate, mean_crate = crate_positions(self) 
	arena = self.game_state['arena']
	diff = np.linalg.norm(agent_pos - crate)
	if diff != 0 :
		direction = (agent_pos - crate) / diff
	else:
		direction = np.array([0,0]) 
	return np.array([diff, *direction])# , mean_crate])

def bomb_difference(self):
	agent = self.game_state['self']
	agent_pos = np.array([agent[0], agent[1]])
	arena = np.array(self.game_state['arena'])
	bombs = np.array(self.game_state['bombs']) 
	if len(bombs)==0: 
		bomb = agent_pos
	else:
		bombs = bombs[:,:2]
		dist = np.array([])   
		for i in range(len(bombs)):
			dist = np.append(dist, np.linalg.norm(agent_pos - bombs[i])) 
		bomb = bombs[np.argmin(dist)]  
	diff = np.linalg.norm(agent_pos - bomb)
	if diff != 0 :
		direction = (agent_pos - bomb) / diff
	else:
		direction = np.array([0,0]) 
	return np.array([diff, *direction])

def explosion_radius(self):
	agent = self.game_state['self']
	agent_pos = np.array([agent[0], agent[1]])
	arena = np.array(self.game_state['arena'])
	bombs = np.array(self.game_state['bombs']) 
	danger = []
	if len(bombs)==0: 
		pass 
	else:
		bombs = bombs[:,:2]

		for bomb in bombs:
			# danger.append(i)
			wall = 0 
			j = 0  
			while (wall == 0) and (j <= 3):
				if arena[bomb[0]+j, bomb[1]] in [1, -1]:
					wall = wall + 1		  
				else:
					danger.append([bomb[0]+j , bomb[1]])
				j += 1 

			wall = 0
			j = -1
			while (wall == 0) and (j >= -3):
				if arena[bomb[0]+j, bomb[1]] in [1, -1]:
					wall = wall + 1
				else:
					danger.append([bomb[0]+j , bomb[1]])
				j -= 1

			wall = 0 
			j = 1 
			while (wall == 0) and (j <= 3):
				if arena[bomb[0], bomb[1]+j] in [1, -1]:
					wall = wall + 1
				else:
					danger.append([bomb[0] , bomb[1]+j])
				j += 1

			wall = 0
			j = -1
			while (wall == 0) and (j >= -3):
				if arena[bomb[0], bomb[1]+j] in [1, -1]:
					wall = wall + 1
				else:
					danger.append([bomb[0] , bomb[1]+j])
				j -= 1

	if np.array(agent_pos) in np.array(danger):
		#print('danger', len(danger), ';', danger)
		return 1
	else:
		return 0
		


def check_even_odd (self):
	agent = self.game_state['self']
	return (np.array([agent[0], agent[1]]) + 1) % 2
   
def q_function(theta_q, features):
	f = theta_q[:,-1]
	for i in range(len(features)):
		f = f + theta_q[:, 3*i] * sigmoid( theta_q[:, 3*i+1] * (features[i] - theta_q[:, 3*i + 2])) 
	return f

def q_function_lin(theta_q, features):
	f = theta_q[:,-1]
	for i in range(len(features)):
		f = f + theta_q[:, i] * features[i] 
	return f

def build_features (self):
	features = difference(self) # [diff, *direction, *mean_coin] also 5 Werte, Indizes 0, 1, 2, 3, 4
	features = np.append(features, check_even_odd(self)) # 2 Werte, Indizes 5, 6
	try:
		features = np.append(features, crate_difference(self)) # [diff, *direction] also 3 Werte, Indizes 7, 8, 9
	except:
		features = np.append(features, np.array([0,0,0]))
	features = np.append(features, bomb_difference(self)) # [diff, *direction] also 3 Werte, Indizes 10, 11, 12
	features = np.append(features, explosion_radius(self)) # 1 (Gefahr) oder 0 (sicher) also 1 Wert, Index 13
	# currently own bomb ticking?
	return features

def setup(self):
	np.random.seed()
	q_data = np.load('agent_code/qn_agent/q_data.npy')
	# print(len(q_data))
	if len(q_data) > 6000:
		q_data = q_data[-6000:]
		np.save('agent_code/qn_agent/q_data.npy', q_data)
	pass

def sigmoid(x):
	return 1/(1+ np.exp(-1 * x))

def act(self):  
	try:
		q_data = np.load('agent_code/qn_agent/q_data.npy')

		features = build_features(self) 

		theta_q = np.load('agent_code/qn_agent/theta_q.npy')

		action = {0:'UP', 1:'DOWN', 2:'RIGHT', 3:'LEFT', 4:'WAIT', 5:'BOMB'}

		q_value = q_function_lin(theta_q, features)
		p_value = sigmoid(8*q_value) # 0.5 + q_value / (np.sum(q_value)*2)
		p_value = p_value / np.sum(p_value)

		#''' 
		eps =  2/(1+ np.log(1+len(q_data)))**1 + 0.15# 0.5# 1 / (1 + len(q_data))**0.3
		#eps = 0.1
		e = np.random.uniform(0,1)
		if e < eps:
			chosen_action = int(np.random.choice([0,1,2,3,4,5]))
		else:
			chosen_action = int(np.argmax(q_value))
		#'''
		
		#indices = np.arange(len(q_value))#[q_value == np.max(q_value)]
		#chosen_action = int(np.random.choice(indices, p=p_value))
		# print(action[chosen_action])
		# print('qn: ', p_value)



		q_data = np.append(q_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)

		np.save('agent_code/qn_agent/q_data.npy', q_data)
		self.next_action = action[chosen_action] 
		# if len(self.game_state['coins'])==1:
		#	 print(self.game_state['step']) 
		#print(action[chosen_action])

	except Exception as e:
		print('Exception as e:')
		print('line: ', sys.exc_info()[2].tb_lineno)
		print(type(e), e) 
		print('Traceback:')
		traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
		print('end of traceback.')
	return None

def reward_update(self):
	global moves
	features = build_features(self)
	history = np.load('agent_code/qn_agent/q_data.npy')
	last_event = history[-1]
	last_move = last_event[0]
	spacken = 0
	if len(history) >= 1:
		second_to_last_move = history[-2][0]
		if (second_to_last_move == 0 and last_move == 1) or (second_to_last_move == 1 and last_move == 0) or (second_to_last_move == 2 and last_move == 3) or (second_to_last_move == 3 and last_move == 2):
			spacken = 2
			#print('spacken')
		
		
	delta_dist = features[0] - last_event[2]
	delta_bd = features[10] - last_event[12]
	bomb_reward = 0
	if features[7] <= 1:
		 bomb_reward = 4
	rewards = {0:-4.5 - spacken - 1.5*features[1]*delta_dist + 0.2*features[3] + 0.5*features[8] + 2*delta_bd - 3*features[13] + 1.5*features[11],
			1:-4.5 - spacken + 1.5*features[1]*delta_dist - 0.2*features[3] - 0.5*features[8] + 2*delta_bd - 3*features[13] - 1.5*features[11],
			2:-4.5 - spacken - 1.5*features[2]*delta_dist + 0.2*features[4] + 0.5*features[9] + 2*delta_bd - 3*features[13] + 1.5*features[12],
			3:-4.5 -spacken + 1.5*features[2]*delta_dist - 0.2*features[4] - 0.5*features[9] + 2*delta_bd - 3*features[13] - 1.5*features[12],
			4:-20 - 20*features[13],
			5:-5,
			6:-10 - 1.5*features[5] - 1.5*features[6] -25*features[13] ,# - 1.5*features[8] - 1.5*features[9], #sinnvoll, dass beitrag bei >1?
			
			7:-5 + bomb_reward,
			8:0, 
			
			9:10,
			10:15,
			11:100,

			12:0,
			13:-300,

			14:-30,
			15:0,
			16:100 #das schafft er eh nicht :D
			}


	reward = np.sum([rewards[item] for item in self.events])
	# print('test 1')
	# print(moves)
	moves = np.append(moves, [last_event[0], reward, *features]) #*last_event[2:]])
	# print('test 2')
	# try:
	#	 with open('test_reward_q.txt', 'a') as f:
	#		 #print(q_value)
	#		 f.write(str(reward)[:] + '\n')
	#		 #f.write(str(p_value)[1:-1] + '\n')
	#		 pass
	# except:
	#	 pass
	moves = np.reshape(moves, (int(len(moves.flat)/(2+len(features))), 2+len(features)))

	return None


def end_of_episode(self):

	try:
		global moves
		theta_q = np.load('agent_code/qn_agent/theta_q.npy')	 
		history = np.load('agent_code/qn_agent/q_data.npy') 
		history = history[:-1]
		y_array = np.load('agent_code/qn_agent/y_data.npy')
		alpha = 0.1 
		gamma = 0.1
		n = 10	
		for t in range(len(moves)):
			if (t >= len(moves) - n):
				n = len(moves) - t - 1
			q_next = np.max(q_function_lin(theta_q, moves[t + n, 2:])) 
			y_t = np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]) + gamma**n * q_next
			y_array = np.append(y_array, y_t)
			# print(moves[t_, 1] for t_ in range(t+1, t+n+1)) 
			# print(moves[:,1])
			# try:
			#	 with open('test_reward_qnn.txt', 'a') as f:
			#		 #print(q_value)
			#		 f.write(str(np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]))[:] + '\n')
			#		 #f.write(str(p_value)[1:-1] + '\n')
			#		 pass
			# except:
			#	 pass

			q_update = history[t, 1] + alpha * (y_t - history[t, 1])   
			# print('Q_n: ', y_t)
			history[t, 1] = q_update
			chosen_action = int(history[t, 0])
			regression_data = history[history[:,0]==chosen_action][:,1:]   
			# theta = opt.least_squares(func, x0=start, method='lm', args=[regression_data, y_array])['x']
			# popt,cov = curve_fit(func, (regression_data[:, 1:].T), regression_data[:, 0])#, p0=theta_q[chosen_action])
		

			# if len(history) <= 800:
				# pass
			# else: 
				# theta_q[chosen_action] = np.array([*popt])
		


		with open('round_number.txt', 'r') as f:
			round_number = int(f.read())
			# print(round_number)
		with open('round_number.txt', 'w') as f:
			f.write(str(round_number + 1))
		if (round_number % 10 == 0):
			# print(theta_q)
			try: 
				# print('hallo')
				theta = []
				for i in range(6):
					if len(history) != len(y_array):
						print('h:', len(history), '; y:', len(y_array))
						history = history[:len(y_array)]
					mask = history[:,0]==i
					# print(self.game_state['step'])
					# print('history: ', np.shape(history))
					# print('y: ', np.shape(y_array))
					# print('theta_q: ', np.shape(theta_q[i]))
					regression_data = history[history[:,0] == i][:,1:]
					popt, pcov = curve_fit(func_curve, (regression_data[:, 1:].T), regression_data[:, 0])#, p0=theta_q[chosen_action])
					theta.append(popt)
					#theta.append(opt.least_squares(func_lin, x0=theta_q[i], method='lm', args=[history[mask][:,2:], y_array[mask]])['x'])



				np.save('agent_code/qn_agent/theta_q.npy', theta)
			except Exception as e:
				print(e)
				print('theta unverändert')
				print('Exception as e:')
				print('line: ', sys.exc_info()[2].tb_lineno)
				print(type(e), e) 
				print('Traceback:')
				traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
				print('end of traceback.')
				pass
			
		# for i in range(4):
		#	 regression_data = history[history[:,0]==int(i)][:,1:]
		#	 popt,cov = curve_fit(func, (regression_data[:, 1:].T), regression_data[:, 0])
		#	 theta_q[chosen_action] = np.array([*popt])
		
		np.save('agent_code/qn_agent/q_data.npy', history)
		np.save('agent_code/qn_agent/y_data.npy', y_array)
		q_data = np.load('agent_code/qn_agent/q_data.npy')

		global last_len_q_data
		new = len(q_data)- last_len_q_data
		#print(len(q_data), 'also +', new, '|')
		if new < 410:
			try:
				with open('neue_q_data.txt', 'a') as f:
					#print(q_value)
					f.write(str(new)[:] + '\n')
					#f.write(str(p_value)[1:-1] + '\n')
					pass
			except:
				pass
		if len(q_data) > 6000:
			q_data = q_data[-6000:]
			y_array = y_array[-6000:]
			np.save('agent_code/qn_agent/y_data.npy', y_array)
			np.save('agent_code/qn_agent/q_data.npy', q_data)
		last_len_q_data = len(q_data)

		# print(self.events)	
		# print('END')
		moves = np.array([[]])	
	except Exception as e:
		print('Exception as e:')
		print('line: ', sys.exc_info()[2].tb_lineno)
		print(type(e), e) 
		print('Traceback:')
		traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
		print('end of traceback.')
		

	return None


