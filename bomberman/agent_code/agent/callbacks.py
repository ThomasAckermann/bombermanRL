import numpy as np
from time import sleep


def positions(self):
    agent = self.game_state['self']
    agent_pos = (agent[0], agent[1])
    coins = self.game_state['coins']
    dist = np.array([])   
    for i in range(len(coins)):
        dist = np.append(dist, np.linalg.norm(np.array(coins[i]) - np.array(agent_pos)))
    coin = coins[np.argmin(dist)]
    print(coin)
    return agent_pos, coin

def collect_coins(self):
    agent_pos, coin = positions(self)
    temp = (np.array(coin) + 1) % 2

    arena = self.game_state['arena']
    diff = np.array(agent_pos) - np.array(coin) 

    if (diff[0] > temp[0]) and (arena[agent_pos[0] - 1, agent_pos[1]] != -1):
        return 'LEFT'
    elif (diff[0] < temp[0]) and (arena[agent_pos[0] + 1, agent_pos[1]] != -1):
        return 'RIGHT'
    elif (diff[1] > 0) and (arena[agent_pos[0], agent_pos[1] - 1] != -1):
        return 'UP'
    elif (diff[1] < 0) and (arena[agent_pos[0], agent_pos[1] + 1] != -1):
        return 'DOWN'
    else: 
        if temp[0] == 1:
            if diff[0] == 1:
                return 'LEFT'
            else:
                return 'RIGHT'
        
        else:
            return 'BOMB'
        # if  (arena[agent_pos[0] - 1, agent_pos[1]] = 1): 
            

def setup(self):
    pass

def act(self): 
    self.next_action = collect_coins(self)
    return None

def reward_update(self):
    pass

def learn(self):
    pass



