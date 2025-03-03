import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from core.GymEnvironment import PacmanEnv
from model import *
from ai import *

class PacmanPPO:
    def __init__(self, lr, lmbda, epochs, eps, gamma, device, input_channel_num = 4, num_actions = 5, extra_info_dim = 4):
        self.net = PacmanNet(num_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.explore_rate = 0.2

    def process_state(self, state_dict):
        board = state_dict["board"]
        if isinstance(board, list):
            board = np.array(board)

        if "board_size" in state_dict:
            size = state_dict["board_size"]
        else:
            size = board.shape[0]

        # pad board to 42 * 42
        padding_num = 42 - size
        board = np.pad(board, pad_width=((padding_num, 0), (padding_num, 0)),
                   mode="constant", constant_values=0)

        # # pacman position matrix
        # pacman_pos = np.zeros((42, 42))
        # if "pacman_coord" in state_dict:
        #     pacman_pos[state_dict["pacman_coord"][0] + padding_num][
        #         state_dict["pacman_coord"][1] + padding_num
        #     ] = 1

        if "pacman_coord" in state_dict:
            board[state_dict["pacman_coord"][0] + padding_num][
                state_dict["pacman_coord"][1] + padding_num
            ] = 10

        # # ghost position matrix
        # ghost_pos = np.zeros((42, 42))
        # if "ghosts_coord" in state_dict:
        #     for ghost in state_dict["ghosts_coord"]:
        #         ghost_pos[ghost[0] + padding_num][ghost[1] + padding_num] = 1

        if "ghosts_coord" in state_dict:
            for ghost in state_dict["ghosts_coord"]:
                board[ghost[0] + padding_num][ghost[1] + padding_num] = 11

        # portal_pos = np.zeros((42, 42))
        # if "portal_coord" in state_dict:
        #     portal = state_dict["portal_coord"]
        #     if portal[0] != -1 and portal[1] != -1:
        #         portal_pos[portal[0] + padding_num][portal[1] + padding_num] = 1
        
        if "portal_coord" in state_dict:
            portal = state_dict["portal_coord"]
            if portal[0] != -1 and portal[1] != -1:
                board[portal[0] + padding_num][portal[1] + padding_num] = 12

        # level = state_dict["level"]
        # if "round" in state_dict:
        #     round = state_dict["round"]
        # else:
        #     round = 0

        # portal_available = 0
        # if "portal_available" in state_dict:
        #     portal_available = int(state_dict["portal_available"])

        # beannumber = 0
        # if "bean_number" in state_dict:
        #     beannumber = state_dict["bean_number"]

        # skill_status = np.zeros(5)
        # if "pacman_skill_status" in state_dict:
        #     skill_status = torch.from_numpy(state_dict["pacman_skill_status"])

        # return torch.tensor(
        #     np.stack([board, pacman_pos, ghost_pos, portal_pos]),
        #     dtype=torch.float32,
        #     device=self.device
        #     ).unsqueeze(0), torch.cat([
        #     torch.tensor(
        #     [level, round, size, portal_available, beannumber],
        #     ), skill_status], 
        #     ).unsqueeze(0).to(self.device).to(torch.float32)
        return torch.from_numpy(board).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.float32)

    def take_action(self, x):
        if random.random() < self.explore_rate:
            action = random.randint(0, 4)
        else:
            probs, _ = self.net(x)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
        return action

    def gae(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantages_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.FloatTensor(advantages_list)

    def update(self, transition_dist):
        states = torch.cat(transition_dist['states'], dim=0).to(self.device)
        # extras = torch.cat(transition_dist['extras'], dim=0).to(self.device)
        actions = torch.tensor(transition_dist['pacman_actions']).contiguous().reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dist['pacman_rewards']).contiguous().reshape((-1, 1)).to(self.device)
        next_states = torch.cat(transition_dist['next_states'], dim=0).to(self.device)
        # next_extras = torch.cat(transition_dist['next_extras'], dim=0).to(self.device)
        dones = torch.FloatTensor(transition_dist['dones']).contiguous().reshape((-1, 1)).to(self.device)
        probs, values = self.net(states)
        _, next_values = self.net(next_states)
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        td_deltas = td_targets - values
        advantage = self.gae(td_deltas.cpu()).to(self.device)
        old_log_probs = torch.log(probs.gather(1, actions) * (1 - self.explore_rate) + self.explore_rate / 5).detach()

        for _ in range(self.epochs):
            prob, value = self.net(states)
            log_probs = torch.log(prob.gather(1, actions) * (1 - self.explore_rate) + self.explore_rate / 5)
            value_coeff = 0.5
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            loss = torch.mean(-torch.min(surr1, surr2)) + torch.mean(F.mse_loss(value, td_targets.detach())) * value_coeff
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

class GhostPPO:
    def __init__(self, lr, lmbda, epochs, eps, gamma, device, input_channel_num = 1, num_actions = 125, extra_info_dim = 0):
        self.net = GhostNet(input_channel_num, num_actions, extra_info_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.explore_rate = 0.2

    def demical_to_base5(self, num):
        base5 = [0, 0, 0]
        for i in range(3):
            base5[2 - i] = num % 5
            num = num // 5
        return base5
    
    def base5_to_demical(self, base5):
        num = 0
        for i in range(3):
            num = num * 5 + base5[i]
        return num
    
    def process_state(self, state_dict):
        board = state_dict["board"]
        if isinstance(board, list):
            board = np.array(board)

        if "board_size" in state_dict:
            size = state_dict["board_size"]
        else:
            size = board.shape[0]

        # pad board to 42 * 42
        padding_num = 42 - size
        board = np.pad(board, pad_width=((padding_num, 0), (padding_num, 0)),
                   mode="constant", constant_values=0)

        # # pacman position matrix
        # pacman_pos = np.zeros((42, 42))
        # if "pacman_coord" in state_dict:
        #     pacman_pos[state_dict["pacman_coord"][0] + padding_num][
        #         state_dict["pacman_coord"][1] + padding_num
        #     ] = 1

        if "pacman_coord" in state_dict:
            board[state_dict["pacman_coord"][0] + padding_num][
                state_dict["pacman_coord"][1] + padding_num
            ] = 10

        # # ghost position matrix
        # ghost_pos = np.zeros((42, 42))
        # if "ghosts_coord" in state_dict:
        #     for ghost in state_dict["ghosts_coord"]:
        #         ghost_pos[ghost[0] + padding_num][ghost[1] + padding_num] = 1

        if "ghosts_coord" in state_dict:
            i = 0
            for ghost in state_dict["ghosts_coord"]:
                board[ghost[0] + padding_num][ghost[1] + padding_num] = 11 + i
                i += 1

        # portal_pos = np.zeros((42, 42))
        # if "portal_coord" in state_dict:
        #     portal = state_dict["portal_coord"]
        #     if portal[0] != -1 and portal[1] != -1:
        #         portal_pos[portal[0] + padding_num][portal[1] + padding_num] = 1
        
        if "portal_coord" in state_dict:
            portal = state_dict["portal_coord"]
            if portal[0] != -1 and portal[1] != -1:
                board[portal[0] + padding_num][portal[1] + padding_num] = 14

        # level = state_dict["level"]
        # if "round" in state_dict:
        #     round = state_dict["round"]
        # else:
        #     round = 0

        # portal_available = 0
        # if "portal_available" in state_dict:
        #     portal_available = int(state_dict["portal_available"])

        # beannumber = 0
        # if "bean_number" in state_dict:
        #     beannumber = state_dict["bean_number"]

        # skill_status = np.zeros(5)
        # if "pacman_skill_status" in state_dict:
        #     skill_status = torch.from_numpy(state_dict["pacman_skill_status"])

        # return torch.tensor(
        #     np.stack([board, pacman_pos, ghost_pos, portal_pos]),
        #     dtype=torch.float32,
        #     device=self.device
        #     ).unsqueeze(0), torch.cat([
        #     torch.tensor(
        #     [level, round, size, portal_available, beannumber],
        #     ), skill_status], 
        #     ).unsqueeze(0).to(self.device).to(torch.float32)
        return torch.from_numpy(board).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.float32)
    
    def take_action(self, x):
        if random.random() < self.explore_rate:
            action = random.randint(0, 125)
        else:
            probs, _ = self.net(x)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample().item()
        return action

    def gae(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantages_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.FloatTensor(advantages_list)

    def update(self, transition_dist):
        states = torch.cat(transition_dist['extras'], dim=0).to(self.device)
        # extras = torch.cat(transition_dist['extras'], dim=0).to(self.device)
        actions = torch.tensor(transition_dist['ghost_actions']).contiguous().reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dist['ghost_rewards']).contiguous().reshape((-1, 1)).to(self.device)
        next_states = torch.cat(transition_dist['next_extras'], dim=0).to(self.device)
        # next_extras = torch.cat(transition_dist['next_extras'], dim=0).to(self.device)
        dones = torch.FloatTensor(transition_dist['dones']).contiguous().reshape((-1, 1)).to(self.device)
        probs, values = self.net(states)
        _, next_values = self.net(next_states)
        td_targets = rewards + self.gamma * next_values * (1 - dones)
        td_deltas = td_targets - values
        advantage = self.gae(td_deltas.cpu()).to(self.device)
        old_log_probs = torch.log(probs.gather(1, actions) * (1 - self.explore_rate) + self.explore_rate / 5).detach()

        for _ in range(self.epochs):
            prob, value = self.net(states)
            log_probs = torch.log(prob.gather(1, actions) * (1 - self.explore_rate) + self.explore_rate / 5)
            value_coeff = 0.5
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            loss = torch.mean(-torch.min(surr1, surr2)) + torch.mean(F.mse_loss(value, td_targets.detach())) * value_coeff
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train():
    lr = 1e-4
    num_episodes = 1000
    gamma = 0.3
    lmbda_pacman = 0.8
    lmbda_ghost = 0.99
    epochs = 8
    eps = 0.2
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    env = PacmanEnv("local")
    pacman = PacmanPPO(lr, lmbda_pacman, epochs, eps, gamma, device, input_channel_num=4, num_actions=5, extra_info_dim=10)
    ghost = GhostPPO(lr, lmbda_ghost, epochs, eps, gamma, device, input_channel_num=1, num_actions=125, extra_info_dim=0)
    pacman.net.load_state_dict(torch.load('pacman_model.pth', map_location=device))
    # ghost.net.load_state_dict(torch.load('ghost_model.pth', map_location=device))
    pacman.net.train()
    ghost.net.train()

    pacman_return_list = []
    ghost_return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                pacman_episode_return = 0
                ghost_episode_return = 0
                transition_dict = {'states': [], 'extras': [], 'pacman_actions': [], 'ghost_actions': [], 'next_states': [], 'next_extras': [], 'pacman_rewards': [], 'ghost_rewards': [], 'dones': []}
                state_dict = env.reset(mode="local")
                valid = False
                center = state_dict['board_size'] // 2
                # state, extra = pacman.process_state(state_dict)
                state = pacman.process_state(state_dict)
                extra = ghost.process_state(state_dict)
                where = np.zeros((state_dict['board_size'], state_dict['board_size']))
                where[state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1]] = 1
                done = False
                if random.random() < 0.5:
                    valid = True
                while not done :
                    # pause = input()
                    # pacman_action = pacman.take_action(state, extra)
                    pacman_action = pacman.take_action(state)
                    if not valid :
                        action = random.randint(0, 1)
                        if state_dict['pacman_coord'][0] == center and state_dict['pacman_coord'][1] == center:
                            valid = True
                            print('here-------------------')
                        elif state_dict['pacman_coord'][0] < center and state_dict['board'][state_dict['pacman_coord'][0] + 1][state_dict['pacman_coord'][1]] != 0:
                            if action == 0:
                                pacman_action = 1
                            else:
                                if state_dict['pacman_coord'][1] < center and state_dict['board'][state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1] + 1] != 0:
                                    pacman_action = 4
                                elif state_dict['pacman_coord'][1] > center and state_dict['board'][state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1] - 1] != 0:
                                    pacman_action = 2
                                else:
                                    pacman_action = 1
                        elif state_dict['pacman_coord'][0] > center and state_dict['board'][state_dict['pacman_coord'][0] - 1][state_dict['pacman_coord'][1]] != 0:
                            if action == 0:
                                pacman_action = 3
                            else:
                                if state_dict['pacman_coord'][1] < center and state_dict['board'][state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1] + 1] != 0:
                                    pacman_action = 4
                                elif state_dict['pacman_coord'][1] > center and state_dict['board'][state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1] - 1] != 0:
                                    pacman_action = 2
                                else:
                                    pacman_action = 3
                        elif state_dict['pacman_coord'][1] < center and state_dict['board'][state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1] + 1] != 0:
                            pacman_action = 4
                        elif state_dict['pacman_coord'][1] > center and state_dict['board'][state_dict['pacman_coord'][0]][state_dict['pacman_coord'][1] - 1] != 0:
                            pacman_action = 2
                        else:
                            valid = True
                            print('here-------------------')
                    ghost_action = ghost.take_action(extra)
                    ghost_action = ghost.demical_to_base5(ghost_action)
                    # if i_episode % 5 == 0:
                    #     ghost_action = ai_func(env.game_state())
                    # else:
                    #     ghost_action = [0, 0, 0]
                    # ghost_action = [random.randint(0, 4), random.randint(0, 4), random.randint(0, 4)]
                    next_state_dict, pacman_reward, ghost_reward, done, _ = env.step(pacmanAction=pacman_action, ghostAction=ghost_action)
                    # next_state, next_extra = pacman.process_state(next_state_dict)
                    next_state = pacman.process_state(next_state_dict)
                    next_extra = ghost.process_state(next_state_dict)
                    extra_reward = 1 if where[next_state_dict['pacman_coord'][0]][next_state_dict['pacman_coord'][1]] == 0 else -1
                    where[next_state_dict['pacman_coord'][0]][next_state_dict['pacman_coord'][1]] = 1
                    if next_state_dict['pacman_coord'][0] == state_dict['pacman_coord'][0] and next_state_dict['pacman_coord'][1] == state_dict['pacman_coord'][1]:
                        extra_reward -= 1
                    # if pacman_reward > 40:
                    #     pacman_reward -= 50
                    transition_dict['states'].append(state)
                    transition_dict['extras'].append(extra)
                    transition_dict['pacman_actions'].append(pacman_action)
                    print(pacman_action, torch.argmax(pacman.net(state)[0]).item(), (pacman_reward + extra_reward) * 10)
                    transition_dict['ghost_actions'].append(ghost_action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['next_extras'].append(next_extra)
                    transition_dict['pacman_rewards'].append((pacman_reward + extra_reward) * 10)
                    transition_dict['ghost_rewards'].append((sum(ghost_reward) - pacman_reward - extra_reward) * 10)
                    transition_dict['dones'].append(done)
                    state = next_state
                    state_dict = next_state_dict
                    extra = next_extra
                    pacman_episode_return += pacman_reward
                    ghost_episode_return += sum(ghost_reward)

                pacman_return_list.append(pacman_episode_return)
                ghost_return_list.append(ghost_episode_return)
                pacman.update(transition_dict)
                ghost.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(pacman_return_list[-10:])})
                pbar.update(1)
            torch.save(pacman.net.state_dict(), 'pacman_model.pth')
            torch.save(ghost.net.state_dict(), 'ghost_model.pth')


    episodes_list = list(range(len(pacman_return_list)))
    plt.plot(episodes_list, pacman_return_list, label='Pacman')
    plt.plot(episodes_list, ghost_return_list, label='Ghost')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.savefig('result_training_results.png')
    plt.show()

if __name__ == '__main__':
    train()