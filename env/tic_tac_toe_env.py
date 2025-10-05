import gymnasium as gym
import numpy as np

from gymnasium.utils.env_checker import check_env


# Constant numbers for representing de agent that interacts with the environment
# and the agent that is part of the environment and plays against it
EXTERNAL_AGENT = 1
INTERNAL_AGENT = -1


def board_solved(board: np.ndarray, user: int):
    assert user == EXTERNAL_AGENT or user == INTERNAL_AGENT, "ERROR: Invalid User"

    solved = False

    # Test if the board has a column, row or diagonal filled by the user number
    if column_solved(board, user) or row_solved(board, user) or diagonal_solved(board, user):
        solved = True

    return solved


def column_solved(board: np.ndarray, user: int):
    return np.any(np.all(board==user, axis=0))


def row_solved(board: np.ndarray, user: int):
    return np.any(np.all(board==user, axis=1))


def diagonal_solved(board: np.ndarray, user: int):
    return np.all(board.diagonal()==user) or np.all(np.fliplr(board).diagonal())


class TicTacToeEnv(gym.Env):
    TIME_PUNSIHMENT = -0.5
    WIN_REWARD = 5
    LOSE_REWARD = -5

    def __init__(self, size: int = 3):
        # Set the board size (3x3 by default)
        self.board_size = size

        # Define the observations space
        self.observation_space = gym.spaces.Box(low=INTERNAL_AGENT, high=EXTERNAL_AGENT, shape=(size, size), dtype=np.int8)

        # Define the actions space
        self.action_space = gym.spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)

        # Define the attribute that represents the board
        self._board = None


    def reset(self, seed: int = None, options: dict = None):
        if seed is not None:
            super().reset(seed=seed)
            self.action_space.seed(seed=seed)

        self._board = np.zeros(shape=(self.board_size, self.board_size), dtype=np.int8)

        # Get observations and extra information values
        observation = self._board
        info = {}

        return observation, info
    

    def step(self, action: gym.spaces.Box):
        # Set the action as a tuple value for indexing the correct value of the array board
        action_tuple = tuple(action)

        # Update tic tac toe board depending on the action taken by the agent
        if self._board[action_tuple] == 0:
            self._board[action_tuple] = EXTERNAL_AGENT

        # Get observations, reward, termination information and extra information
        observation = self._board
        terminated, winner = self._is_terminated()
        truncated = False
        reward = self._calculate_reward(terminated, winner)
        info = {}

        if not terminated and not truncated:
            internal_action_tuple = tuple(self.action_space.sample())
            while self._board[internal_action_tuple] != 0:
                internal_action_tuple = tuple(self.action_space.sample())
            self._board[internal_action_tuple] = INTERNAL_AGENT

            observation = self._board
            terminated, winner = self._is_terminated()
            truncated = False
            reward = self._calculate_reward(terminated, winner)
            info = {}

        return observation, reward, terminated, truncated, info
    

    def print_board(self):
        print()
        for row in range(2 * self.board_size-1):
            row_string = ""
            for col in range(2 * self.board_size-1):
                if row % 2 == 0:
                    if col % 2 == 0:
                        row_string += str(self._board[int(row/2), int(col/2)])
                    else:
                        row_string += "|"
                else:
                    row_string += "-"

            print(row_string)
        print()
    

    def _is_terminated(self):
        terminated = False
        winner = None

        if board_solved(board=self._board, user=EXTERNAL_AGENT):
            terminated = True
            winner = EXTERNAL_AGENT
        elif board_solved(board=self._board, user=INTERNAL_AGENT):
            terminated = True
            winner = INTERNAL_AGENT

        return terminated, winner
    

    def _calculate_reward(self, terminated, winner):
        reward = None

        if not terminated:
            reward = self.TIME_PUNSIHMENT / self.board_size
        else:
            if winner == 1:
                reward = self.WIN_REWARD
            elif winner == -1:
                reward = self.LOSE_REWARD

        return reward
    

if __name__ == "__main__":
    # Create an instance of the environment
    env = TicTacToeEnv()

    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")