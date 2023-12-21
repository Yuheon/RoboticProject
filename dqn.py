import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
import argparse
from tqdm import tqdm
from itertools import count

import numpy as np
from collections import namedtuple, deque
from model import SimpleDQNNetwork

from env import PathPlanningEnv
from utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class DQN():
    def __init__(self, args, is_training=True):
        
        self.is_training = is_training
        self.policy_net = SimpleDQNNetwork(1, 4).to(device)
        if is_training:
            self.target_net = SimpleDQNNetwork(1, 4).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.batch_size = args.batch_size
        
            self.memory = ReplayMemory(args.replay_size)
            self.gamma = args.gamma
            self.tau = args.tau
            self.eps_start = args.eps_start
            self.eps_end = args.eps_end
            self.eps_decay = args.eps_decay
        
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)
        
            self.steps_done = 0

    def select_action(self, state):
        if self.is_training:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
                    # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
                    # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[np.random.choice(4)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
        # 전환합니다.
        batch = Transition(*zip(*transitions))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # 변화도 클리핑 바꿔치기
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def train(self, args):
        
        env = PathPlanningEnv(args.env_level, args.agent_jump)
        
        for e_i in tqdm(range(args.n_episodes)):
            cum_reward = 0
            
            # 환경과 상태 초기화
            state, reward, done = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            for t in count():
                
                action = self.select_action(state)
                next_state, reward, done = env.step(action.item())
                
                cum_reward += reward
                
                reward = torch.tensor([reward], device=device)
        
        
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

                # 메모리에 변이 저장
                self.memory.push(state, action, next_state, reward)

                # 다음 상태로 이동
                state = next_state

                # (정책 네트워크에서) 최적화 한단계 수행
                self.optimize_model()

                # 목표 네트워크의 가중치를 소프트 업데이트
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                # if t > 1000:
                #     done = True
                if done:
                    # self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    break
            print(f'Episode {e_i}: t: {t}, cumulative reward:{cum_reward}, cum_rweward/t: {cum_reward / t}')
            if e_i % 50 == 0:
                torch.save(self.policy_net.state_dict(), f"dqn_lv{args.env_level}_rp{args.replay_size}_ep{e_i}.pt")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_episodes", type=int, default=1000, help="")
    parser.add_argument("--lr", type=float, default=0.001, help="")
    parser.add_argument("--batch_size", type=int, default=512, help="")
    
    # for RL
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--replay_size", type=int, default=200000, help="")
    parser.add_argument("--tau", type=float, default=0.005, help="")
    # exploration
    parser.add_argument("--eps_start", type=float, default=0.9, help="")
    parser.add_argument("--eps_end", type=float, default=0.05, help="")
    parser.add_argument("--eps_decay", type=int, default=1000, help="")


    # for ENV
    parser.add_argument("--env_level", type=int, default=0, choices=[0, 1, 2], help="")
    parser.add_argument("--agent_jump", type=int, default=1, choices=[0, 1, 2, 3], help="how agent jump")
    
    args = parser.parse_args()
    
    # seed_everything(args.seed)
    
    dqn = DQN(args)
    dqn.train(args)
    
    