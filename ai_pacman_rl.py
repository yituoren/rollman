from core.gamedata import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class PacmanNet(nn.Module):
    def __init__(self, num_actions = 5):
        super().__init__()
        # [B, 1, 42, 42]
        self.embeddings = nn.Embedding(13, 16)
        # [B, 16, 42, 42]
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # [B, 32, 42, 42]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # [B, 64, 42, 42]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # [B, 128, 21, 21]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # [B, 256, 11, 11]

        self.fc1 = nn.Linear(256 * 11 * 11, 1024)
        self.fc2 = nn.Linear(1024, 32)
        self.policy = nn.Linear(32, num_actions)
        self.value = nn.Linear(32, 1)
    
    def forward(self, x):
        B, _, H, W = x.shape
        flattened_channel = x.view(B, -1).long()
        embedded_channel = self.embeddings(flattened_channel)
        embedded_channel = embedded_channel.permute(0, 2, 1).view(B, 16, H, W)
        x = embedded_channel

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy(x), dim=1)
        value = self.value(x)
        return policy, value

class PacmanAI:
    def __init__(self):
        self.net = PacmanNet()
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, "pacman_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.net.to(self.device)
        self.net.eval()

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
        probs, _ = self.net(x)
        action = torch.argmax(probs, dim=1)
        return action.item()

    def choose_move(self, game_state: GameState):
        state_dict = game_state.gamestate_to_statedict()
        x = self.process_state(state_dict)
        action = self.take_action(x)
        return [action]

ai_func = PacmanAI().choose_move
__all__ = ["ai_func"]
