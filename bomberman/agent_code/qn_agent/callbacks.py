import numpy as np
from time import sleep
# from tempfile import * 
from scipy.optimize import curve_fit

moves = np.array([[]])


def func(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8):
    (x_1, x_2, x_3, x_4, x_5, x_6, x_7) = X
    return a_1 + a_2*x_1 + a_3*x_2 + a_4*x_3 + a_5*x_4 + a_6*x_5 + a_7*x_6 + a_8*x_7 

def positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    coins = np.array(self.game_state['coins'])
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
    direction = (agent_pos - coin) / diff
    return np.array([diff, *direction, *mean_coin])


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
    return features

def setup(self):
    np.random.seed()
    pass

def sigmoid(x):
    return 1/(1+ np.exp(-1 * x))

def act(self):  
    # diff = difference(self)
    features = build_features(self) 
    
    # print(features)
    theta_q = np.load('agent_code/qn_agent/theta_q.npy')
    action = {0:'UP', 1:'DOWN', 2:'RIGHT', 3:'LEFT'}
    q_value = q_function(theta_q, features)
    p_value = sigmoid(8*q_value) # 0.5 + q_value / (np.sum(q_value)*2)
    p_value = p_value / np.sum(p_value)
    # print(p_value)
    indices = np.arange(len(q_value))#[q_value == np.max(q_value)]
    chosen_action = int(np.random.choice(indices, p=p_value))
    # print(chosen_action)
    q_data = np.load('agent_code/qn_agent/q_data.npy')
    q_data = np.append(q_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)
    np.save('agent_code/qn_agent/q_data.npy', q_data)
    self.next_action = action[chosen_action] 
    if len(self.game_state['coins'])==1:
        print(self.game_state['step']) 
    return None

def reward_update(self):
    global moves
    features = build_features(self)
    history = np.load('agent_code/qn_agent/q_data.npy') 
    last_event = history[0] 

    rewards = {0:-2 + features[1] + 0.2*features[3] + 0.2*features[5]  , 
            1:-2 - features[1] - 0.2*features[3] + 0.2*features[5], 
            2:-2 + features[2] + 0.2*features[4] + 0.2*features[6], 
            3:-2 - features[2] - 0.2*features[4] + 0.2*features[6], 
            4:-1, 
            5:-1,
            6:-5 - 5*features[5] - 5*features[6],
            
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
    moves = np.append(moves, [last_event[0], reward, *features])
    moves = np.reshape(moves, (int(len(moves.flat)/(2+len(features))), 2+len(features)))
    return None


def end_of_episode(self):
    global moves
    theta_q = np.load('agent_code/qn_agent/theta_q.npy')     
    history = np.load('agent_code/qn_agent/q_data.npy') 
   
    alpha = 0.04 
    gamma = 0.1
    n = 5
    print('Begin Calculations')
    for t in range(len(moves)):
        if (t >= len(moves) - n):
            n = len(moves) - t - 1
        q_next = np.max(q_function(theta_q, moves[t + n, 2:])) 
        y_t = np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]) + gamma**n *q_next
        q_update = history[t, 1] + alpha * (y_t - history[t, 1])   
        history[t, 1] = q_update
        chosen_action = int(history[t, 0])
        regression_data = history[history[:,0]==chosen_action][:,1:]
        popt,coyv = curve_fit(func, (regression_data[:, 1:].T), regression_data[:, 0])  
        theta_q[chosen_action] = np.array([*popt])

    np.save('agent_code/qn_agent/theta_q.npy', theta_q)
    np.save('agent_code/qn_agent/q_data.npy', history)
    
 
    #if len(history) <= 200:
    #    pass
    #else:
     
    # print(self.events)    
    print('END')
    
    return None


