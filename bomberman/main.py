from time import time, sleep
import contextlib
from time import time
import time as time_

with contextlib.redirect_stdout(None):
    import pygame
from pygame.locals import *
import numpy as np
import multiprocessing as mp
import threading

from environment import BombeRLeWorld, ReplayWorld
from settings import s


# Function to run the game logic in a separate thread
def game_logic(world, user_inputs):
    last_update = time()
    while True:
        # Game logic
        if (s.turn_based and len(user_inputs) == 0):
            sleep(0.1)
        elif (s.gui and (time()-last_update < s.update_interval)):
            sleep(s.update_interval - (time() - last_update))
        else:
            last_update = time()
            if world.running:
                try:
                    world.do_step(user_inputs.pop(0) if len(user_inputs) else 'WAIT')
                except Exception as e:
                    world.end_round()
                    raise


def main():
    pygame.init()

    # Emulate Windows process spawning behaviour under Unix (for testing)
    # mp.set_start_method('spawn')

    agent_list = [
            ('cbt_agent', True),
            ('simple_agent', False),
            ('simple_agent', False),
            ('simple_agent', False)
        ]
    # stores the path to the last agent, which is used for the round number and extraction of theta at every 20th round 
    path = './agent_code/{}/'.format(agent_list[-1][0])

    # Initialize environment and agents 
    world = BombeRLeWorld(agent_list)

    # world = ReplayWorld('replay_3')# ('Replay 2019-01-30 16:57:42')
    user_inputs = []

    # Start game logic thread
    t = threading.Thread(target=game_logic, args=(world, user_inputs))
    t.daemon = True
    t.start()

    start_time = time()
    print('Start am {2}.{1}.{0}, um {3:02}:{4:02}:{5:02} Uhr.'.format(*time_.localtime(start_time)))
    # Run one or more games
    for i in range(s.n_rounds):
        if (i % 20 == 0):
            print(i)
        if not world.running:
            world.ready_for_restart_flag.wait()
            world.ready_for_restart_flag.clear()
            world.new_round()

        # First render
        if s.gui:
            world.render()
            pygame.display.flip()

        round_finished = False
        last_update = time()
        last_frame = time()
        user_inputs.clear()

        # Main game loop
        while not round_finished:
            # Grab events
            key_pressed = None
            for event in pygame.event.get():
                if event.type == QUIT:
                    world.end_round()
                    world.end()
                    return
                elif event.type == KEYDOWN:
                    key_pressed = event.key
                    if key_pressed in (K_q, K_ESCAPE):
                        world.end_round()
                    if not world.running:
                        round_finished = True
                    # Convert keyboard input into actions
                    if s.input_map.get(key_pressed):
                        if s.turn_based:
                            user_inputs.clear()
                        user_inputs.append(s.input_map.get(key_pressed))

            if not world.running and not s.gui:
                round_finished = True

            # Rendering
            if s.gui and (time()-last_frame >= 1/s.fps):
                world.render()
                pygame.display.flip()
                last_frame = time()
            else:
                sleep_time = 1/s.fps - (time() - last_frame)
                if sleep_time > 0:
                    sleep(sleep_time)
                if not s.gui:
                    last_frame = time()

    world.end()
    end_time = time()
    print('Ende am {2}.{1}.{0}, um {3:02}:{4:02}:{5:02} Uhr.'.format(*time_.localtime(end_time)))
    print('Duration =', end_time - start_time, 's =', (end_time - start_time)/60, 'min')


if __name__ == '__main__':
    main()
