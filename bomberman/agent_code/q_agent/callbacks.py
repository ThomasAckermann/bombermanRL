import numpy as np
from time import sleep
# from tempfile import * 
from scipy.optimize import curve_fit

def func(X, a, b, c, d):
    (x,y,z) = X
    return a + b*x + c*y + d*z


def positions(self):
    agent = self.game_state['self']
    agent_pos = (agent[0], agent[1])
    coins = self.game_state['coins']
    dist = np.array([])   
    for i in range(len(coins)):
        dist = np.append(dist, np.linalg.norm(np.array(coins[i]) - np.array(agent_pos)))
    coin = coins[np.argmin(dist)] 
    return agent_pos, coin

def difference(self):
    agent_pos, coin = positions(self) 
    arena = self.game_state['arena']
    diff = np.linalg.norm(np.array(agent_pos) - np.array(coin))
    direction = (np.array(agent_pos) - np.array(coin)) / diff
    return np.array([diff, *direction])

def q_function(theta_q, features):
    f = theta_q[:,0]
    for i in range(len(features)):
        f = f + theta_q[:, i+1] * features[i] 
    return f

def setup(self):
    np.random.seed()
    pass

def sigmoid(x):
    return 1/(1+ np.exp(-1 * x))

def act(self): 
    # diff = difference(self)
    features = difference(self)
    theta_q = np.load('agent_code/q_agent/theta_q.npy')
    action = {0:'UP', 1:'DOWN', 2:'RIGHT', 3:'LEFT'}
    q_value = q_function(theta_q, features)
    p_value = sigmoid(4*q_value) # 0.5 + q_value / (np.sum(q_value)*2)
    p_value = p_value / np.sum(p_value)
    print(p_value)
    indices = np.arange(len(q_value))# [q_value == np.max(q_value)]
    chosen_action = int(np.random.choice(indices, p=p_value))
    print(chosen_action)
    q_data = np.load('agent_code/q_agent/q_data.npy')
    q_data = np.append(q_data, [np.array([q_value[chosen_action], *features, chosen_action])], axis=0)
    np.save('agent_code/q_agent/q_data.npy', q_data)
    self.next_action = action[chosen_action] 
    return None

def reward_update(self):
    alpha = 0.05
    gamma = 0.03
    features = difference(self)
    
    history = np.load('agent_code/q_agent/q_data.npy')
    theta_q = np.load('agent_code/q_agent/theta_q.npy')    
  
    last_event = history[-1]
    r = features[0] - last_event[1]
      
   
    rewards = {0:-1+features[1], 
            1:-1-features[1], 
            2:-1+features[2], 
            3:-1-features[2], 
            4:-1, 
            5:-1,
            6:-10,
            
            7:0,
            8:0, 
            
            9:0,
            10:0,
            11:20,

            12:0,
            13:0,

            14:0,
            15:0,
            16:20
            }
    
    reward = np.sum([rewards[item] for item in self.events])
    q_next = np.max(q_function(theta_q, features)) 
    q_update = last_event[0] + alpha * (reward + gamma * q_next - last_event[0]) 
    history[-1][0] = q_update
    np.save('agent_code/q_agent/q_data.npy', history)
    chosen_action = int(last_event[-1])
    regression_data = history[history[:,-1]==chosen_action][:,:-1]
    popt,cov = curve_fit(func, (regression_data[:, 1:].T), regression_data[:, 0])  
    # print(len(history))
    if len(history) <= 200:
        pass
    else:
        theta_q[chosen_action] = np.array([*popt])
    # print(chosen_action)
    # print(theta_q)
    
    np.save('agent_code/q_agent/theta_q.npy', theta_q)
    return None


def end_of_episode(self):
    print('END')
    return None


