import os
import numpy as np

import time
import argparse
import pandas as pd

import torch

from env import PathPlanningEnv
from utils import seed_everything

# A*
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# DQN
from dqn import DQN
# RAINBOW
from rainbow import Network as RainbowNet

def astar(grid_state):
    # grid 맵을 정의합니다. 1은 장애물을, 0은 이동 가능한 공간을 의미합니다.
    
    grid = Grid(matrix=grid_state.tolist())

    # 시작점과 끝점을 정의합니다.
    start = grid.node(0, 0)
    end = grid.node(9, 9)

    # A* 알고리즘을 초기화하고 경로를 찾습니다.
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)

    if not path:
        # print("no path")
        return None
    else:
        # print('operations:', runs, 'path length:', len(path))
        # print(grid.grid_str(path=path, start=start, end=end))
        return len(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--num_eval", type=int, default=100, help="")
    parser.add_argument("--method", type=str, default="dqn", choices=["astar", "dqn", "rainbow"], help="")

    # for ENV
    parser.add_argument("--env_level", type=int, default=0, choices=[0, 1, 2], help="")
    parser.add_argument("--agent_jump", type=int, default=1, choices=[0, 1, 2, 3], help="how agent jump")
    
    args = parser.parse_args()
    
    
    seed_everything(args.seed)

    # make environment with env config
    env = PathPlanningEnv(args.env_level, args.agent_jump)
    
    rewards = []
    times = []
    
    if args.method == "astar":
        for _ in range(args.num_eval):
            state, reward, done = env.reset()  # Reset the environment before starting
            
            """
            pathfinding.core.grid.Grid
            deal with <= 0 as obstacle, > 0 as the weight of a field that can be walked on
            """
            state[0, 0] = 0
            state[9, 9] = 0
            
            inverted_array = np.logical_not(state).astype(int)
            start_time = time.time()
            path_len = astar(inverted_array)
            
            if path_len is not None:
                reward = env.step_reward * (path_len-2) + 10   # exclude start and goal, add goal reward
                running_time = time.time() - start_time
                # print(running_time, reward)
                
                rewards.append(reward)
                times.append(running_time)
    
    elif args.method == "dqn":
        dqn = DQN(args, is_training=False)
        dqn.policy_net.load_state_dict(torch.load("dqn_ep400.pt"))
        dqn.policy_net.to('cpu')
        dqn.policy_net.eval()
        
        with torch.no_grad():
            for _ in range(args.num_eval):
                state, reward, done = env.reset()  # Reset the environment before starting
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                cum_reward = 0
                start_time = time.time()
                count = 0
                while not done:
                    action = dqn.select_action(state)
                    next_state, reward, done = env.step(action.item())
                    
                    cum_reward += reward
                    
                    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                    count += 1
                    if count > 1000:
                        break
                    
                if count <= 1000:
                    running_time = time.time() - start_time
                    rewards.append(cum_reward)
                    times.append(running_time)
    elif args.method == 'rainbow':
        v_min = 0.0
        v_max = 200.0
        atom_size = 51
        
        rainbow = RainbowNet(100, 4, atom_size, torch.linspace(v_min, v_max, atom_size))
        rainbow.load_state_dict(torch.load("rainbow_lv2_rp50000_fr10000.pt"))
        rainbow.eval()
        
        with torch.no_grad():
            for _ in range(args.num_eval):
                state, reward, done = env.reset()  # Reset the environment before starting
                state = torch.tensor(state, dtype=torch.float32).reshape(1, 1, -1)
                
                cum_reward = 0
                start_time = time.time()
                count = 0
                while not done:
                    action = rainbow(state).argmax()
                    action = action.detach().cpu().numpy()
                    next_state, reward, done = env.step(action.item())
                    
                    cum_reward += reward
                    
                    next_state = torch.tensor(next_state, dtype=torch.float32).reshape(1, 1, -1)
                    count += 1
                    if count > 1000:
                        break
                if count <= 1000:
                    running_time = time.time() - start_time
                    rewards.append(cum_reward)
                    times.append(running_time)
                
    else:
        raise NotImplemented
    
    df = pd.DataFrame({"reward": rewards, "time":times})
    df.to_csv(f'{args.method}_lv{args.env_level}.csv')
    
    print(len(rewards), np.mean(rewards), np.mean(times))


