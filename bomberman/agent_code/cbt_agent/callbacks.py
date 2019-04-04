import numpy as np
from time import sleep
from scipy.optimize import curve_fit
import sys 
import traceback
import random
import copy

moves = np.array([[]])
path = '' #'./agent_code/cbt_agent/'
crate_counter = 0
round_number = 1

def func_curve(X, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, a_16, a_17, a_18, a_19, a_20, a_21, a_22, a_23, a_24, a_25, a_26, a_27, a_28, a_29, a_30, a_31, a_32, a_33, a_34):
    (x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30, x_31, x_32, x_33) = X
    return a_1*x_1 + a_2*x_2 + a_3*x_3 + a_4*x_4 + a_5*x_5 + a_6*x_6 + a_7*x_7 + a_8*x_8 + a_9*x_9 + a_10*x_10 + a_11*x_11 + a_12*x_12 + a_13*x_13 + a_14*x_14 + a_15*x_15 + a_16*x_16 + a_17*x_17 + a_18*x_18 + a_19*x_19 + a_20*x_20 + a_21*x_21 + a_22*x_22 + a_23*x_23 + a_24*x_24 + a_25*x_25 + a_26*x_26 + a_27*x_27 + a_28*x_28 + a_29*x_29 + a_30*x_30 + a_31*x_31 + a_32*x_32 + a_33*x_33 + a_34


def positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    coins = np.array(self.game_state['coins'])
    if len(coins) == 0:
        # return agent_pos three times, otherwise the agent would try to go to the top left corner, this way he "tries to stay where he is"
        return agent_pos, agent_pos, np.zeros(2)
    else:
        dist = np.array([])   
        for i in range(len(coins)):
            dist = np.append(dist, np.linalg.norm(agent_pos - coins[i]))
        coin = coins[np.argmin(dist)]
        mean_coin = []
        
        for i in range(len(coins)):
            if dist[i] != 0:
                mean_coin.append((agent_pos - coins[i])/dist[i]**(3/2))
        if (len(np.shape(mean_coin)) > 1):
            mean_coin = np.sum(mean_coin, axis=0)
        if np.linalg.norm(mean_coin) != 0: 
            mean_coin = (mean_coin)/ np.linalg.norm(mean_coin) 
        if (mean_coin == []):
            mean_coin = np.zeros(2)
        return agent_pos, np.array(coin), mean_coin # agent_pos, absolute pos naechster coin, richtung mean coin

def difference(self):
    agent_pos, coin, mean_coin = positions(self) 
    arena = self.game_state['arena']
    diff = np.linalg.norm(agent_pos - coin)
    if diff != 0 :
        direction = (agent_pos - coin) / diff
    else:
        direction = np.zeros(2)

    return np.array([diff, *direction, *mean_coin])


def crate_positions(self):
    agent = self.game_state['self']
    agent_pos = np.array([agent[0], agent[1]])
    arena = self.game_state['arena']
    crates = []
    for i in range(np.shape(arena)[0]):
        for j in range(np.shape(arena)[1]):
            if arena[i][j] == 1:
                crates.append([i,j])
    crates = np.array(crates) 
    if len(crates) == 0:
        # return agent_pos three times, otherwise the agent would try to go to the top left corner, this way he "tries to stay where he is"
        return agent_pos, agent_pos, np.zeros(2)
    else:
        dist = np.array([])   
        for i in range(len(crates)):
            dist = np.append(dist, np.linalg.norm(agent_pos - crates[i]))
        crate = crates[np.argmin(dist)]
        mean_crate = []
        
        for i in range(len(crates)):
            if dist[i] != 0:
                mean_crate.append((agent_pos - crates[i])/dist[i]**(3/2))
        if (len(np.shape(mean_crate)) > 1):
            mean_crate = np.sum(mean_crate, axis=0)
        if np.linalg.norm(mean_crate) != 0: 
            mean_crate = (mean_crate)/ np.linalg.norm(mean_crate) 
        if (mean_crate == []):
            mean_crate = np.zeros(2)
        return agent_pos, np.array(crate), mean_crate # agent_pos, absolute pos naechster coin, richtung mean coin

def crate_diff(self):
    agent_pos, crate, mean_crate = crate_positions(self)     
    diff = np.linalg.norm(agent_pos - crate)
    if diff != 0 :
        direction = (agent_pos - crate) / diff
        if diff != 1:
            diff = 0
    else:
        direction = np.zeros(2)

    return np.array([*direction, *mean_crate, diff])

def check_even_odd(position):
    # 1 means, that the row/column is free
    # x and y are swapped, because the y direction is free if the x value is odd
    position = np.array(position)[np.r_[1, 0]] #np.r_ does the swap of x and y
    return position % 2
   
def last_move(self):
    forbidden = np.zeros(4)
    try:
        move = [self.game_state['self'][:2][i] - self.last_coords[i] for i in range(2)]
        moves = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        last_move_ = moves.index(move)
        forbidden[int(last_move_ + 1 - 2 * (last_move_ % 2))] = 1
    except:
        pass
    return forbidden


def own_bomb_ticking(self):
    #returns 1 if own bomb is ticking, else 0
    bomb_possible = self.game_state['self'][3]
    return (bomb_possible + 1) % 2

def explosion_radius_single_bomb(coordinates):
    check_free = check_even_odd(coordinates)
    x, y, x_, y_ = *coordinates, *check_free
    row = [[item, y] for item in np.unique(np.clip(np.r_[x-3:x+4], 0, 16))] if x_ else [[x, y]]
    column = [[x, item] for item in np.unique(np.clip(np.r_[y-3:y+4], 0, 16))] if y_ else [[x, y]] 
    return row + column

def position_in_danger(position, self):
    danger = 0
    for bomb in self.game_state['bombs']:
        if (list(position) in explosion_radius_single_bomb(bomb[:2])):
            if bomb[2] == 0:
                danger = 5
            else:
                danger = np.max([(0.633-0.133*bomb[2]), danger])
    if self.game_state['explosions'][tuple(position)] == 2:
        danger = 5
    return danger + 4 * np.clip(danger, 0, 0.1)

def number_of_crates_in_explosion_radius(self):
    own_pos = np.array(self.game_state['self'][:2])
    explosion_radius = explosion_radius_single_bomb(own_pos)
    tiles = np.array([self.game_state['arena'][tuple(item)] for item in explosion_radius])
    number = len(np.arange(len(tiles))[tiles == 1])
    number = 0.7 * number if number <= 2 else number
    return number

def next_move_danger(self):
    # returns 0 if no danger is in the corresponding direction or a wall 
    danger = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for i in range(4):
        danger[i] = position_in_danger(own_pos + pos_diffs[i], self)
        pos = own_pos + pos_diffs[i]
        if (np.abs(self.game_state['arena'][tuple([pos[0], pos[1]])]) == 1):
            danger[i] = 0.8
    return danger

def directions_blocked(pos, self):
    pos = np.array(pos)
    # returns 0 if direction is free, 1 if it is blocked 
    blocked = np.zeros(4)
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for i in range(4):
        blocked[i] = np.abs(self.game_state['arena'][tuple(pos + pos_diffs[i])])
        if list(pos + pos_diffs[i]) in [list(coord[:-1]) for coord in self.game_state['bombs']]:
            blocked[i] = 1
        if list(pos + pos_diffs[i]) in [list(coord[:2]) for coord in self.game_state['others']]:
            blocked[i] = 1
    return blocked
    
def next_move_blocked (self):
    blocked = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) 
    for i in range(4):
        current_pos = np.array(own_pos + pos_diffs[i])
        if (current_pos < 1).any() or (current_pos > 15).any():
            blocked[i] = 1
        else:
            blocked[i] = -3 + np.clip(np.sum(directions_blocked(current_pos, self)), 3, 4)
    return blocked

def tile_blocked(pos, self):
    blocked = np.abs(self.game_state['arena'][tuple(pos)])
    if list(pos) in [list(coord[:-1]) for coord in self.game_state['bombs']]:
        blocked = 1
    if list(pos) in [list(coord[:2]) for coord in self.game_state['others']]:
        blocked = 1
    return blocked

def no_through_road(self, check_no_bomb = False):
    blocked = np.zeros(4)
    own_pos = np.array(self.game_state['self'][:2])
    if (list(own_pos) not in [list(coord[:-1]) for coord in self.game_state['bombs']] and not check_no_bomb):
        return blocked
    pos_diffs = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) # left, right, up, down 
    orthogonal_list = np.array([[2, 3], [0, 1]])
    for i in range(4): #all directions
        cur_orthogonal = orthogonal_list[int(np.floor(0.5*i))]
        for j in range(4):
            current_pos = own_pos + (j+1)*pos_diffs[i]
            if (tile_blocked(current_pos, self) == 1):
                blocked[i] = 1
                break
            elif (directions_blocked(current_pos, self)[cur_orthogonal] == 0).any():
                break
    return blocked

def no_bomb(self):
    all_directions_blocked = no_through_road(self, check_no_bomb = True).all()
    return 1 if all_directions_blocked else 0

def killable_opponents(self):
    number = 0
    if own_bomb_ticking(self) == 0:
        explosion_radius = explosion_radius_single_bomb(self.game_state['self'][:2])
        for opponent in self.game_state['others']:
            if list(opponent[:2]) in explosion_radius:
                number += 1
    return number


def q_function(theta_q, features):
    f = theta_q[:,-1]
    for i in range(len(features)):
        f = f + theta_q[:, i] * features[i] 
    return f


def build_features (self):
    current_pos = self.game_state['self'][:2]
    features = difference(self) # [diff, *direction, *mean_coin] also 5 Werte, Indizes 0, 1, 2, 3, 4
    features = np.append(features, check_even_odd(current_pos)) # 2 Werte, Indizes 5, 6
    features = np.append(features, last_move(self)) # 4 Werte, Indizes 7, 8, 9, 10
    features = np.append(features, position_in_danger(current_pos, self)) # 1 Wert, Index 11
    features = np.append(features, own_bomb_ticking(self)) # 1 Wert, Index 12
    features = np.append(features, number_of_crates_in_explosion_radius(self)) # 1 Wert, Index 13
    features = np.append(features, directions_blocked(current_pos, self)) # 4 Werte, Indizes 14, 15, 16, 17
    features = np.append(features, next_move_danger(self)) # 4 Werte, Indizes 18, 19, 20, 21
    features = np.append(features, crate_diff(self)) # 5 Werte, Indizes 22, 23, 24, 25, 26
    features = np.append(features, no_through_road(self)) # 4 Werte, Indizes 27, 28, 29, 30
    features = np.append(features, no_bomb(self)) # 1 Wert, Index 31
    features = np.append(features, killable_opponents(self)) # 1 Wert, Index 32
    self.last_coords = current_pos
    return features



def setup(self):
    self.theta = np.array([[-9.41628111e-01,9.22945874e+00,1.36345789e+00,-1.90475662e+00
,-1.65306401e+00,-8.63364688e-01,5.50669699e-01,-5.15574501e+00
,3.75939344e-01,1.77268861e+00,2.17951132e+00,-9.58596567e-02
,-3.45980922e+00,-2.37804123e+00,-2.76523345e+01,-2.61112025e+00
,-5.03728689e-01,-5.03579680e-01,-6.47056045e+00,1.31629723e+00
,1.15013407e+00,6.61613235e-01,6.57211430e-01,-8.60887862e-02
,3.33243019e-01,-3.44057142e-01,-1.04700878e+00,-2.05704236e+01
,7.90610821e-01,6.84426502e-02,5.60601157e-02,2.69731029e-01
,-9.72607233e-02,1.08960072e+00]
,[-7.47103808e-01,-8.55628045e+00,-1.69117138e+00,8.86671819e-01
,1.93795729e+00,-2.87329228e-01,8.35363466e-01,-9.43445239e-01
,-5.32246964e+00,8.03716245e-01,1.11732823e+00,2.24446650e+00
,-4.68404657e+00,-2.31038121e+00,-2.50294614e+00,-2.55027333e+01
,-6.62964222e-02,-4.16801849e-01,1.68366652e+00,-5.92047207e+00
,7.19934691e-01,6.17838762e-01,-8.85110625e-01,-1.00449663e-01
,-2.68631597e-01,-3.71620079e-01,-1.12591020e+00,8.73300230e-01
,-2.02797588e+01,3.19398089e-01,-1.66234116e-02,-9.74416300e-02
,2.36057203e-03,1.25371616e+00]
,[-1.31982732e+00,-4.38922085e-01,5.90198501e+00,2.75927102e-01
,2.86722119e+00,1.20805449e+00,-3.41211678e+00,1.01792637e+00
,1.23257435e+00,-5.48612540e+00,1.37678382e-01,5.13044503e-01
,-4.44410288e+00,-2.22900938e+00,-5.73460531e-01,-2.12375102e-01
,-2.88276888e+01,-2.33329976e+00,1.42148428e+00,6.01720291e-01
,-6.03637039e+00,1.40930834e+00,-1.14680130e-01,5.54452850e-01
,-1.07282664e-01,2.32432697e-01,-8.14434865e-01,-2.85017325e-02
,-1.43599372e-01,-2.09525198e+01,6.67737039e-01,2.14565004e-02
,8.32773698e-03,3.27352473e+00]
,[-3.75475795e-01,8.46706565e-02,-4.55688327e+00,-1.90695444e-02
,-1.35402449e+00,6.78542268e-01,-3.40926195e+00,1.16900183e+00
,1.43949292e+00,-2.76197317e-01,-5.02602389e+00,2.04928331e+00
,-4.32040761e+00,-2.19486945e+00,-3.04186662e-01,1.05258846e-01
,-2.79952667e+00,-2.67214583e+01,3.60769109e-01,3.37203718e-01
,1.58781945e+00,-7.30643924e+00,-2.97305552e-02,-8.77360088e-01
,-9.40642135e-02,-4.67297856e-01,-7.78429982e-01,2.45066172e-01
,-1.66621191e-02,8.91651574e-01,-2.13712434e+01,-1.55939741e-01
,-2.37258441e-02,4.15434473e+00]
,[-1.18755008e-01,-2.58375095e+00,4.63629243e+00,2.93828884e+00
,-4.60678227e+00,-1.52279224e+00,-3.13831134e-01,4.94954498e-01
,1.03158515e+00,5.86231788e-01,4.96112607e-01,-7.00000000e+01
,8.98388430e-01,7.52180969e-02,-4.69825117e+00,-3.49091024e+00
,-3.56101977e+00,-3.15913202e+00,2.75664245e+00,2.75914266e+00
,2.50776343e+00,2.24736801e+00,-1.81883957e-02,2.22228627e-02
,1.08706419e-01,-1.47139910e-01,1.62876800e+00,2.13886726e-01
,-1.37923482e+00,1.75010842e-02,-1.02418655e+00,-7.89453257e+00
,7.54407884e-02,-1.82982888e+00]
,[-3.18570569e-02,-3.66520085e-01,-1.16445417e+00,3.13877484e-01
,1.27605230e+00,-1.02652453e+00,-8.57630863e-01,2.52824719e-01
,-2.05295994e-02,1.27858073e-02,1.56056096e-01,-2.00000000e+01
,-5.49030345e+01,3.22744958e+00,-1.09596647e+00,-1.11788134e+00
,-9.46493274e-01,-8.58317789e-01,2.98603776e-01,2.70411722e-01
,1.40575066e-01,5.71486225e-02,9.22885997e-02,5.40059433e-02
,-1.42553876e-02,-1.28669165e-01,2.21082707e+00,-2.47096453e+00
,-2.17640318e+00,-4.78100476e-01,-9.28862489e-01,-3.35639105e+01
,5.25775985e+00,-8.19891458e-01]])
    self.last_coords = [0, 0]
    #self.theta = np.load('{}thetas/theta_q.npy'.format(path))


def act(self):
    self.next_action = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'][int(np.argmax(q_function(self.theta, build_features(self))))]
    return None        

def reward_update(self):
    return None
    global moves, round_number
    last_event = self.q_data[-1]
    f = last_event[2:]
    
    #sleep(10)

    event_dict = {0 : "MOVED_LEFT",
            1 : "MOVED_RIGHT",
            2 : "MOVED_UP",
            3 : "MOVED_DOWN",
            4 : "WAITED",
            5 : "INTERRUPTED",
            6 : "INVALID_ACTION",
            7 : "BOMB_DROPPED",
            8 : "BOMB_EXPLODED",
            9 : "CRATE_DESTROYED",
            10 : "COIN_FOUND",
            11 : "COIN_COLLECTED",
            12 : "KILLED_OPPONENT",
            13 : "KILLED_SELF",
            14 : "GOT_KILLED",
            15 : "OPPONENT_ELIMINATED",
            16 : "SURVIVED_ROUND"}

    rewards = {"MOVED_LEFT" : -2 + f[1] + 0.2*f[3] + 0.2*f[5] - 3*f[7] - 35*f[18] + 2*f[22] + 0.4*f[24] - 22*f[27], 
                    "MOVED_RIGHT" : -2 - f[1] - 0.2*f[3] + 0.2*f[5] - 3*f[8] - 35*f[19] - 2*f[22] - 0.4*f[24] - 22*f[28],
                    "MOVED_UP" : -2 + f[2] + 0.2*f[4] + 0.2*f[6] - 3*f[9] - 35*f[20] + 2*f[23] + 0.4*f[25] - 22*f[29],
                    "MOVED_DOWN" : -2 - f[2] - 0.2*f[4] + 0.2*f[6] - 3*f[10] - 35*f[21] - 2*f[23] - 0.4*f[25] - 22*f[30],
                    "WAITED" : -6 - 300*f[11], 
                    "INTERRUPTED" : 0,
                    "INVALID_ACTION" : -5 - 5*f[5] - 5*f[6] - 50*f[12] - 10*np.clip(np.sum(f[14:18]), 0, 1),
                    
                    "BOMB_DROPPED" :  -5 + 5*f[13] + 2*f[26] - 30*f[31] + 10*f[32] - 4*f[11],
                    "BOMB_EXPLODED" : 0, 
                    
                    "CRATE_DESTROYED" : 2,
                    "COIN_FOUND" : 0,
                    "COIN_COLLECTED" : 20,

                    "KILLED_OPPONENT" : 0,
                    "KILLED_SELF" : 0,

                    "GOT_KILLED" : 0,
                    "OPPONENT_ELIMINATED" : 0,
                    "SURVIVED_ROUND" : 0}

    reward = 0.667761 * np.sum([rewards[event_dict[item]] for item in self.events])

    global crate_counter
    for event in self.events:
        if event == 9:
            crate_counter += 1

    moves = np.append(moves, [last_event[0], reward, *f])
    moves = np.reshape(moves, (int(len(moves.flat)/(2+len(f))), 2+len(f)))
    return None


def end_of_episode(self):
    return None
    try:
        global moves, round_number, path, crate_counter
        alpha = 0.04        
        gamma = 0.4
        n = 7    
        for t in range(len(moves)):
            if (t >= len(moves) - n):
                n = len(moves) - t - 1
            q_next = np.max(q_function(self.theta, moves[t + n, 2:])) 
            y_t = np.sum([gamma**(t_ - t - 1) * moves[t_, 1] for t_ in range(t + 1, t + n + 1)]) + gamma**n * q_next

            q_update = self.q_data[-len(moves) + t, 1] + alpha * (y_t - self.q_data[-len(moves) + t, 1])   
            
            self.q_data[-len(moves) + t, 1] = q_update        

        
        with open('{}moves.txt'.format(path), 'a') as f:
            new_coins = self.game_state['self'][4]
            last_score = self.game_state['self'][4]
            f.write(str(round_number) + ' ' + str(len(moves)) + ' ' + str(new_coins) + ' ' + str(crate_counter) + '\n')
            crate_counter = 0

        if (round_number % 100 == 0):
            try: 
                theta = []
                for i in range(6):
                    regression_data = copy.deepcopy(self.q_data)
                    if len(self.all_data) > 6000:
                        tmp = self.all_data[:-6000]
                        if len(tmp) > 6000:
                            tmp = tmp[np.array(random.sample(list(np.arange(len(tmp))), 6000))]
                        regression_data = np.append(tmp, regression_data, axis = 0)
                    
                    mask = regression_data[:,0]==i
                    regression_data = regression_data[mask][:, 1:]
                    
                    popt, pcov = curve_fit(func_curve, (regression_data[:, 1:].T), regression_data[:, 0], p0=self.theta[i])
                    theta.append(popt)
                self.theta = np.clip(np.array(theta), -90, 20)

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
         
        if len(self.q_data) > 6000:
            self.q_data = self.q_data[-6000:]       
        
        if len(self.all_data) > 30000:
            self.all_data = self.all_data[np.array(random.sample(list(np.arange(len(self.all_data))), 30000))]    

        if round_number % 20 == 0:
            np.save('{}thetas/theta_q.npy'.format(path), self.theta)
            np.save('{}q_data/q_data.npy'.format(path), self.q_data)
            np.save('{}all_data/all_data.npy'.format(path), self.all_data)
            np.save('{}thetas/theta_nach_{}_spielen.npy'.format(path, round_number), self.theta)
            if round_number % 60 == 0:
                np.save('{}all_data/all_data_nach_{}_spielen.npy'.format(path, round_number), self.all_data)

        round_number += 1
        moves = np.array([[]])    
        # print('END')
    except Exception as e:
        print('Exception as e:')
        print('line: ', sys.exc_info()[2].tb_lineno)
        print(type(e), e) 
        print('Traceback:')
        traceback.print_tb(sys.exc_info()[2], file = sys.stdout)
        print('end of traceback.')
        
    return None


