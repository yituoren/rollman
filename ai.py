from core.gamedata import GameState


def example_ai(game_state: GameState) -> list[int]:
    import numpy as np

    ghost_pos = game_state.ghosts_pos
    pacman_pos = game_state.pacman_pos

    vector1 = np.array(ghost_pos[0]) - np.array(pacman_pos)
    standardize1 = vector1 / np.linalg.norm(vector1)

    vector2 = np.array(ghost_pos[1]) - np.array(pacman_pos)
    standardize2 = vector2 / np.linalg.norm(vector2)

    vector3 = np.array(ghost_pos[2]) - np.array(pacman_pos)
    standardize3 = vector3 / np.linalg.norm(vector3)

    ret = []
    if abs(standardize1[0]) > abs(standardize1[1]):
        if standardize1[0] > 0:
            ret.append(2)
        else:
            ret.append(4)
    else:
        if standardize1[1] > 0:
            ret.append(3)
        else:
            ret.append(1)
            
    if abs(standardize2[0]) > abs(standardize2[1]):
        if standardize2[0] > 0:
            ret.append(2)
        else:
            ret.append(4)
    else:
        if standardize2[1] > 0:
            ret.append(3)
        else:
            ret.append(1)
            
    if abs(standardize3[0]) > abs(standardize3[1]):
        if standardize3[0] > 0:
            ret.append(2)
        else:
            ret.append(4)
    else:
        if standardize3[1] > 0:
            ret.append(3)
        else:
            ret.append(1)

    return ret



ai_func = example_ai
__all__ = ['ai_func']