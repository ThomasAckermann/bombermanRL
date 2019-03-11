import numpy as np
from time import sleep
# from tempfile import * 
from scipy.optimize import curve_fit
# import random


moves = np.array([[]])
eps = 0.1

def func(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13):
    (x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12) = X
    return a_1 + a_2*x_1 + a_3*x_2 + a_4*x_3 + a_5*x_4 + a_6*x_5 + a_7*x_6 + a_8*x_7 + a_9*x_8 + a_10*x_9 + a_11*x_10 + a_12*x_11 + a_13*x_12

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
    return np.array([diff, *direction])

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
    return np.array([diff])

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
        return 1
    else:
        return 0
        


def check_even_odd (self):
    agent = self.game_state['self']
    return (np.array([agent[0], agent[1]]) + 1) % 2
   
def q_function(theta_q, features):
    f = theta_q[:,0]
    for i in range(len(features)):
        f = f + theta_q[:, i+1] * features[i] 
    return f


def build_features (self):
    features = difference(self)
    features = np.append(features, check_even_odd(self))
    features = np.append(features, crate_difference(self)) 
    features = np.append(features, bomb_difference(self))
    features = np.append(features, explosion_radius(self))
    return features

def setup(self):
    np.random.seed()
    q_value = np.load('agent_code/q_bomb_agent/q_data.npy')
    print(len(q_value))
    if len(q_value) > 1000:
        q_value = q_value[200:]
        np.save('agent_code/q_bomb_agent/q_data.npy', q_data)
    pass

def sigmoid(x):
    return 1/(1+ np.exp(-1 * x))

def act(self):  
    q_data = np.load('agent_code/q_bomb_agent/q_data.npy')

    features = build_features(self) 

    theta_q = np.load('agent_code/q_bomb_agent/theta_q.npy')
    action = {0:'UP', 1:'DOWN', 2:'RIGHT', 3:'LEFT', 4:'WAIT', 5:'BOMB'}
    q_value = q_function(theta_q, features)
    p_value = sigmoid(8*q_value) # 0.5 + q_value / (np.sum(q_value)*2)
    p_value = p_value / np.sum(p_value)
    
    #''' 
    eps =  0.3# 1 / (1 + len(q_data))**0.3
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
    np.save('agent_code/q_bomb_agent/q_data.npy', q_data)
    self.next_action = action[chosen_action] 
    # if len(self.game_state['coins'])==1:
    #     print(self.game_state['step']) 

    return None

def reward_update(self):
  

    alpha = 0.04
    gamma = 0.1
    features = build_features(self)

    history = np.load('agent_code/q_bomb_agent/q_data.npy')
    theta_q = np.load('agent_code/q_bomb_agent/theta_q.npy')    
  
    last_event = history[-1] 

    delta_dist = features[0] - last_event[2]
    delta_bd = features[10] - last_event[12]

    rewards = {0:-3 - 1.5*features[1]*delta_dist + 0.2*features[3] + 0.5*features[8] + 1*delta_bd - 3*features[11],
            1:-3 + 1.5*features[1]*delta_dist - 0.2*features[3] - 0.5*features[8] + 1*delta_bd - 3*features[11],
            2:-3 - 1.5*features[2]*delta_dist + 0.2*features[4] + 0.5*features[9] + 1*delta_bd - 3*features[11],
            3:-3 + 1.5*features[2]*delta_dist - 0.2*features[4] - 0.5*features[9] + 1*delta_bd - 3*features[11],
            4:-7 - 10*features[11], 
            5:-1,
            6:-5 - 1.5*features[5] - 1.5*features[6] - 1.5*features[8] - 1.5*features[9],
            
            7:-5,
            8:0, 
            
            9:5,
            10:10,
            11:20,

            12:0,
            13:-50,

            14:-30,
            15:0,
            16:20
            }

    
    
    reward = np.sum([rewards[item] for item in self.events])

    q_next = np.max(q_function(theta_q, features)) 
    y_t = reward + gamma * q_next
    q_update = last_event[0] + alpha * (y_t - last_event[0]) 
    # print('Q: ', y_t)
    history[-1][0] = q_update
    np.save('agent_code/q_bomb_agent/q_data.npy', history)
    chosen_action = int(last_event[-1])
    regression_data = history[history[:,-1]==chosen_action][:,:-1]
    popt,cov = curve_fit(func, (regression_data[:, 1:].T), regression_data[:, 0])  
    
    if len(history) <= 200:
        pass

    return None



def end_of_episode(self):
    # q_data = np.open('agent_code/q_bomb_agent/q_data.npy')
    # if len(q_data) > 10000:
    #     np.save('agent_code/q_bomb_agent/q_data.npy', q_data[50:])
    print('END')
    return None



