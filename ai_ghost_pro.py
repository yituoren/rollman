from core.gamedata import *
import random
import numpy as np


class GhostAI:
    def __init__(self):
        pass

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def a_star_search(self, start: np.ndarray, goal: np.ndarray, game_state: GameState):
        open_set = set()
        open_set.add(tuple(start))
        came_from = {}

        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.manhattan_distance(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float("inf")))
            if current == tuple(goal):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            open_set.remove(current)
            for direction, _ in self.get_valid_moves(list(current), game_state):
                neighbor = tuple(direction)
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(
                        neighbor, goal
                    )
                    if neighbor not in open_set:
                        open_set.add(neighbor)

        return []
    
    def middle_point(self, pos1, pos2, pos3):
        return (pos1[0] + pos2[0] + pos3[0]) / 3, (pos1[1] + pos2[1] + pos3[1]) / 3
    
    def all_manhattan_distance(self, pos1, pos2, pos3, target):
        return (
            abs(pos1[0] - target[0]) + abs(pos1[1] - target[1])
            + abs(pos2[0] - target[0]) + abs(pos2[1] - target[1])
            + abs(pos3[0] - target[0]) + abs(pos3[1] - target[1])
        )
        
    
    def rectangular_area(self, pos1, pos2, pos3):
        x_min = min(pos1[0], pos2[0], pos3[0])
        x_max = max(pos1[0], pos2[0], pos3[0])
        y_min = min(pos1[1], pos2[1], pos3[1])
        y_max = max(pos1[1], pos2[1], pos3[1])
        return (x_max - x_min) * (y_max - y_min)

    def get_valid_moves(self, pos, game_state):
        valid_moves = []
        directions = [
            ([0, 0], 0),  # STAY
            ([1, 0], 1),  # UP
            ([-1, 0], 3),  # DOWN
            ([0, -1], 2),  # LEFT
            ([0, 1], 4),  # RIGHT
        ]

        for direction, move_value in directions:
            new_pos = [pos[0] + direction[0], pos[1] + direction[1]]
            if (
                0 <= new_pos[0] < game_state.board_size
                and 0 <= new_pos[1] < game_state.board_size
                and game_state.board[new_pos[0]][new_pos[1]] != 0
            ):
                valid_moves.append((new_pos, move_value))
        return valid_moves

    def choose_moves(self, game_state: GameState):
        pacman_pos = game_state.pacman_pos
        ghost_pos = game_state.ghosts_pos
        
        # 获取 Pacman 的可能移动位置（减少考虑的位置数量）
        pacman_valid_moves = self.get_valid_moves(pacman_pos, game_state)
        pacman_possible_positions = [pos for pos, _ in pacman_valid_moves]
        
        if not pacman_possible_positions:
            pacman_possible_positions = [pacman_pos]
        else:
            # 加入当前位置，考虑不动的情况
            pacman_possible_positions.append(pacman_pos)
            # 只保留最有可能的几个位置（当前位置和最多3个可能移动位置）
            if len(pacman_possible_positions) > 4:
                # 使用曼哈顿距离预筛选，保留当前位置和离幽灵最远的3个位置
                distances = []
                for pos in pacman_possible_positions:
                    if pos[0] == pacman_pos[0] and pos[1] == pacman_pos[1]:  # 保留当前位置
                        continue
                    # 计算到所有幽灵的最小曼哈顿距离
                    min_dist = min(self.manhattan_distance(pos, g) for g in ghost_pos)
                    distances.append((pos, min_dist))
                # 按距离排序，保留最远的几个
                distances.sort(key=lambda x: -x[1])  # 降序排列
                selected_positions = [p for p, _ in distances[:3]]
                pacman_possible_positions = selected_positions + [pacman_pos]
        
        # 获取每个幽灵的有效移动，使用曼哈顿距离作为初步距离评估
        valid_moves = []
        path_cache = {}  # 缓存A*寻路结果
        
        for ghost_id in range(3):
            current_pos = ghost_pos[ghost_id]
            valid_move = self.get_valid_moves(current_pos, game_state)
            moves = []
            
            for pos, move in valid_move:
                # 先用曼哈顿距离作为初步评估
                manhattan_dist = self.manhattan_distance(pos, pacman_pos)
                
                # 只对曼哈顿距离近的位置计算A*路径
                if manhattan_dist < game_state.board_size:
                    # 检查缓存中是否已有相同起点和终点的路径
                    cache_key = (tuple(pos), tuple(pacman_pos))
                    if cache_key in path_cache:
                        distance = path_cache[cache_key]
                    else:
                        path = self.a_star_search(pos, pacman_pos, game_state)
                        distance = len(path) if path else game_state.board_size * 2
                        path_cache[cache_key] = distance
                else:
                    distance = manhattan_dist
                    
                moves.append((pos, move, distance))
            
            valid_moves.append(moves)
        
        # 记录每个移动组合被选为最佳的次数
        move_counts = {}
        
        # 对每个可能的 Pacman 位置分别评估
        for p_pos in pacman_possible_positions:
            # 使用与原逻辑相同的初始设置
            best_mid_pos = self.middle_point(ghost_pos[0], ghost_pos[1], ghost_pos[2])
            best_mid_distance = (p_pos[0] - best_mid_pos[0])**2 + (p_pos[1] - best_mid_pos[1])**2
            
            # 检查 Pacman 是否在矩形外
            out = False
            if (p_pos[0] < min(ghost_pos[0][0], ghost_pos[1][0], ghost_pos[2][0])) or \
               (p_pos[0] > max(ghost_pos[0][0], ghost_pos[1][0], ghost_pos[2][0])):
                out = True
            if (p_pos[1] < min(ghost_pos[0][1], ghost_pos[1][1], ghost_pos[2][1])) or \
               (p_pos[1] > max(ghost_pos[0][1], ghost_pos[1][1], ghost_pos[2][1])):
                out = True
                    
            current_area = self.rectangular_area(ghost_pos[0], ghost_pos[1], ghost_pos[2])
            best_area = current_area
            best_manhattan_distance = self.all_manhattan_distance(ghost_pos[0], ghost_pos[1], ghost_pos[2], p_pos)
            current_manhattan_distance = [
                self.manhattan_distance(ghost_pos[0], p_pos),
                self.manhattan_distance(ghost_pos[1], p_pos),
                self.manhattan_distance(ghost_pos[2], p_pos)
            ]
            
            # 计算对当前 Pacman 位置的本地距离
            local_valid_moves = []
            for ghost_id in range(3):
                current_pos = ghost_pos[ghost_id]
                valid_move = self.get_valid_moves(current_pos, game_state)
                moves = []
                
                for pos, move in valid_move:
                    # 优先使用缓存或曼哈顿距离
                    cache_key = (tuple(pos), tuple(p_pos))
                    if cache_key in path_cache:
                        distance = path_cache[cache_key]
                    else:
                        # 只有当距离近时才使用A*
                        manhattan_dist = self.manhattan_distance(pos, p_pos)
                        if manhattan_dist < game_state.board_size:
                            path = self.a_star_search(pos, p_pos, game_state)
                            distance = len(path) if path else game_state.board_size * 2
                            path_cache[cache_key] = distance
                        else:
                            distance = manhattan_dist
                    
                    moves.append((pos, move, distance))
                    
                # 按距离排序，只保留前3个最好的移动
                moves.sort(key=lambda x: x[2])
                local_valid_moves.append(moves[:3])
            
            best_moves = [local_valid_moves[0][0][1], local_valid_moves[1][0][1], local_valid_moves[2][0][1]]
            best_distance = local_valid_moves[0][0][2] + local_valid_moves[1][0][2] + local_valid_moves[2][0][2]
            
            # 评估组合（减少评估的组合数量）
            for pos1, move1, distance1 in local_valid_moves[0]:
                for pos2, move2, distance2 in local_valid_moves[1]:
                    for pos3, move3, distance3 in local_valid_moves[2]:
                        mid_pos = self.middle_point(pos1, pos2, pos3)
                        distance = distance1 + distance2 + distance3
                        mid_distance = (p_pos[0] - mid_pos[0])**2 + (p_pos[1] - mid_pos[1])**2
                        area = self.rectangular_area(pos1, pos2, pos3)
                        
                        # 使用曼哈顿距离代替全部的A*距离计算
                        manhattan_distance = self.all_manhattan_distance(pos1, pos2, pos3, p_pos)
                        
                        if not out:  # Pacman 在矩形内
                            if distance < best_distance:
                                best_moves = [move1, move2, move3]
                                best_distance = distance
                                best_mid_distance = mid_distance
                            elif distance == best_distance and mid_distance < best_mid_distance:
                                best_moves = [move1, move2, move3]
                                best_mid_distance = mid_distance
                        else:  # Pacman 在矩形外
                            if current_area >= 64:
                                if distance < best_distance:
                                    best_moves = [move1, move2, move3]
                                    best_distance = distance
                            elif area > current_area and \
                                 self.manhattan_distance(pos1, p_pos) < current_manhattan_distance[0] and \
                                 self.manhattan_distance(pos2, p_pos) < current_manhattan_distance[1] and \
                                 self.manhattan_distance(pos3, p_pos) < current_manhattan_distance[2]:
                                if area > best_area:
                                    best_area = area
                                    best_moves = [move1, move2, move3]
                                    best_mid_distance = mid_distance
                                    best_manhattan_distance = manhattan_distance
                                    best_distance = distance
                                elif mid_distance < best_mid_distance:
                                    best_mid_distance = mid_distance
                                    best_moves = [move1, move2, move3]
                                    best_manhattan_distance = manhattan_distance
                                    best_distance = distance
            
            # 处理 Pacman 在矩形外且没有找到好的移动的情况
            if out and best_moves[0] == 0 and best_moves[1] == 0 and best_moves[2] == 0:
                best_distance = float("inf")
                for pos1, move1, distance1 in local_valid_moves[0]:
                    for pos2, move2, distance2 in local_valid_moves[1]:
                        for pos3, move3, distance3 in local_valid_moves[2]:
                            distance = distance1 + distance2 + distance3
                            if distance < best_distance:
                                best_moves = [move1, move2, move3]
                                best_distance = distance
            
            # 统计这个移动组合被选为最佳的次数
            move_key = tuple(best_moves)
            move_counts[move_key] = move_counts.get(move_key, 0) + 1
        
        # 选择被最多 Pacman 位置认为是最佳的移动组合
        best_count = 0
        final_moves = [0, 0, 0]
        for moves, count in move_counts.items():
            if count > best_count:
                best_count = count
                final_moves = list(moves)
        
        return final_moves


ai_func = GhostAI().choose_moves
__all__ = ["ai_func"]
