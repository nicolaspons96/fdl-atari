import gym
import argparse
import numpy as np
import atari_py
from ddqn_game_model import DDQNTrainer, DDQNSolver
from gym_wrappers import MainGymWrapper

import time

FRAMES_IN_OBSERVATION = 4#time
FRAME_SIZE = 84#84x84 input images
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)

class Atari:
    def __init__(self):
        game_mode, render, total_step_limit, total_run_limit, clip = self.get_args()
        env_name = "BreakoutDeterministic-v4"  # Handles frame skipping (4) at every iteration in deterministic env
        env = MainGymWrapper.wrap(gym.make(env_name))
        self.main_loop(self.game_model(game_mode, env.action_space.n), env, render, total_step_limit, total_run_limit, clip)

    def main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        run = 0
        total_step = 0
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print("Reached total step limit of: " + str(total_step_limit))
                    game_model.save_model()
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()
                    time.sleep(0.01) # handle the game's speed

                action = game_model.move(current_state)
                next_state, reward, done, info = env.step(action)
                if clip:
                    np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, done)
                current_state = next_state

                game_model.step_update(total_step)

                if done:
                    game_model.save_run(score, step, run)
                    break
        game_model.save_model()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mode", help="Choose from available modes: training, testing, best. Default is 'training'.", default="training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'", default=True, type=bool)
        args = parser.parse_args()
        game_mode = args.mode
        render = args.render
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = args.clip
        print("Game: Breakout")
        print("Selected mode: " + str(game_mode))
        print("Should render: " + str(render))
        print("Should clip: " + str(clip))
        print("Total step limit: " + str(total_step_limit))
        print("Total run limit: " + str(total_run_limit))
        return game_mode, render, total_step_limit, total_run_limit, clip

    def game_model(self, game_mode, action_space):
        """
        action_space: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        """
        if game_mode == "training":
            return DDQNTrainer(INPUT_SHAPE, action_space)
        elif game_mode == "best":
            return DDQNSolver(INPUT_SHAPE, action_space, is_best=True)
        elif game_mode == "testing":
            return DDQNSolver(INPUT_SHAPE, action_space, is_best=False)
        else:
            print("Unrecognized mode. Use --help")
            exit(1)

if __name__ == "__main__":
    Atari()
