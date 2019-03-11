import numpy as np
from time import sleep
# from tempfile import * 
from scipy.optimize import curve_fit
# import random


moves = np.array([[]])
eps = 0.1

def get_data(self):
    data = np.array([])
    for key in ['step', 'arena', 'explosions']:
        data = np.append(data, self.game_state[key])
    player_mask = [True, True, False, True, True] 
    data = np.append(data, np.array(self.game_state['self'])[player_mask])
    n = len(self.game_state['others'])
    for i in range(3):
        if i < n:
            data = np.append(data, np.array(self.game_state['others'])[i][player_mask])
        else:
            data = np.append(data, [0,0,0,0]) 
    n = len(self.game_state['bombs'])
    for i in range(4):
        if i < n:
            data = np.append(data, np.array(self.game_state['bombs'])[i])
        else:
            data = np.append(data, [0,0,0])
    n = len(self.game_state['coins'])
    for i in range(9):
        if i < n:
            data = np.append(data, np.array(self.game_state['coins'])[i])
        else:
            data = np.append(data, [0,0])
    data = np.array(data, dtype=int)
    return data 

def get_neuron_values(data, weight, bias):
    return sigmoid((weight @ data) + bias)


def nn(self, weights, biases, lin_data_1, lin_data_2):
    data = get_data(self)
    neurons = get_neuron_values(data, weights, biases)
    return neurons @ lin_data_1 + lin_data_2


def setup(self):  
    pass

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def act(self):
    # print(self.game_state['self'])
    print('length: ', len(get_data(self)), 'data: ', get_data(self))
    self.next_action = 'BOMB'
    
    '''
    q_data = np.load('agent_code/qn_agent/q_data.npy')

    features = build_features(self) 

    theta_q = np.load('agent_code/qn_agent/theta_q.npy')
    action = {0:'UP', 1:'DOWN', 2:'RIGHT', 3:'LEFT', 4:'WAIT', 5:'BOMB'}
    q_value = q_function(theta_q, features)
    p_value = sigmoid(8*q_value) # 0.5 + q_value / (np.sum(q_value)*2)
    p_value = p_value / np.sum(p_value)
    
    # 
    eps =  0.2# 1 / (1 + len(q_data))**0.3
    e = np.random.uniform(0,1)
    if e < eps:
        chosen_action = int(np.random.choice([0,1,2,3,4,5]))
    else:
        chosen_action = int(np.argmax(q_value))
    #
    
    #indices = np.arange(len(q_value))#[q_value == np.max(q_value)]
    #chosen_action = int(np.random.choice(indices, p=p_value))
    # print(action[chosen_action])
    # print('qn: ', p_value)
    q_data = np.append(q_data, [np.array([chosen_action, q_value[chosen_action], *features])], axis=0)
    np.save('agent_code/qn_agent/q_data.npy', q_data)
    self.next_action = action[chosen_action] 
    # if len(self.game_state['coins'])==1:
    #     print(self.game_state['step']) 
    '''
    return None

def reward_update(self):
    '''
    global moves
    features = build_features(self)
    history = np.load('agent_code/qn_agent/q_data.npy')
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
    moves = np.append(moves, [last_event[0], reward, last_event[2:]])
    moves = np.reshape(moves, (int(len(moves.flat)/(2+len(features))), 2+len(features)))
    '''
    return None


def end_of_episode(self):
  
    global moves
 
    theta_q = np.load('agent_code/qn_agent/theta_q.npy')     
    history = np.load('agent_code/qn_agent/q_data.npy') 
   
    alpha = 0.1 
    gamma = 0.1
    n = 10
    # print('Begin Calculations')
    for t in range(len(moves)):
        if (t >= len(moves) - n):
            n = len(moves) - t - 1
        q_next = np.max(q_function(theta_q, moves[t + n, 2:])) 
        y_t = np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]) + gamma**n * q_next
        q_update = history[t, 1] + alpha * (y_t - history[t, 1])   
        # print('Q_n: ', y_t)
        history[t, 1] = q_update
        chosen_action = int(history[t, 0])
        regression_data = history[history[:,0]==chosen_action][:,1:]
        popt,cov = curve_fit(func, (regression_data[:, 1:].T), regression_data[:, 0])
        if len(history) <= 800:
            pass
        else: 
            theta_q[chosen_action] = np.array([*popt])
   

    np.save('agent_code/qn_agent/theta_q.npy', theta_q)
    np.save('agent_code/qn_agent/q_data.npy', history)

    q_value = np.load('agent_code/qn_agent/q_data.npy')
    print(len(q_value))
    if len(q_value) > 20000:
        q_value = q_value[100:]
        np.save('agent_code/qn_agent/q_data.npy', q_value) 
    # print(self.events)    
    print('END')
    moves = np.array([[]])

    '''
    return None


