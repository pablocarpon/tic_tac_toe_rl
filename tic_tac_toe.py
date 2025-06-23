import numpy as np
import random
from tqdm import tqdm
import pickle

WEIGHTS_PATH = r"C:\Users\7pabl\Desktop\code\python_code\reinforcement_learning\tic_tac_toe_wights.pkl"


class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # Representación del tablero en una lista
        self.current_winner = None  # Guarda el ganador
    
    def reset(self):
        self.board = [' '] * 9
        self.current_winner = None
        return self.get_state()
    
    def get_state(self):
        return ''.join(self.board)  # Estado del tablero como string
    
    def get_winner(self):
        if self.current_winner == "O":
            return "Humano"
        elif self.current_winner == "X":
            return "Agente"
        else:
            return None
    
    def available_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']
    
    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False
    
    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all(s == letter for s in row):
            return True
        
        col_ind = square % 3
        col = [self.board[col_ind+i*3] for i in range(3)]
        if all(s == letter for s in col):
            return True
        
        if square % 2 == 0:
            diag1 = [self.board[i] for i in [0, 4, 8]]
            if all(s == letter for s in diag1):
                return True
            diag2 = [self.board[i] for i in [2, 4, 6]]
            if all(s == letter for s in diag2):
                return True
        return False
    
    def is_full(self):
        return ' ' not in self.board
    
    def game_over(self):
        return self.current_winner is not None or self.is_full()

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # Diccionario de Q-values
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Exploración
    
    def get_q_values(self, state):
        return self.q_table.setdefault(state, np.zeros(9))
    
    def choose_action(self, state, available_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)
        q_values = self.get_q_values(state)
        return max(available_moves, key=lambda x: q_values[x])
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        if done:
            q_values[action] = reward
        else:
            next_q_values = self.get_q_values(next_state)
            q_values[action] += self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No se encontró el archivo, comenzando con una tabla Q vacía.")


class MinMaxRival:
    def __init__(self, letter='O', max_depth=5):
        self.letter = letter
        self.opponent = 'X' if letter == 'O' else 'O'
        self.memo = {}  # Diccionario para memoization
        self.max_depth = max_depth

    def minimax(self, board, depth, is_maximizing, alpha, beta):
        board_state = ''.join(board)  # Convertimos el tablero a string para cache

        if board_state in self.memo:
            return self.memo[board_state]  # Devolvemos el valor guardado

        available_moves = [i for i in range(9) if board[i] == ' ']

        # Condiciones de victoria o empate
        if TicTacToe().winner(0, 'X'):
            return -10 + depth  # Penaliza más en niveles profundos
        elif TicTacToe().winner(0, 'O'):
            return 10 - depth  # Premia más en niveles profundos
        elif not available_moves:
            return 0  # Empate

        # Si alcanzamos la profundidad máxima, devolvemos una heurística
        if depth >= self.max_depth:
            return self.evaluate_board(board)

        if is_maximizing:
            best_score = -float('inf')
            for move in available_moves:
                board[move] = self.letter
                score = self.minimax(board, depth + 1, False, alpha, beta)
                board[move] = ' '
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Poda Alpha-Beta
            self.memo[board_state] = best_score  # Guardamos el resultado en cache
            return best_score
        else:
            best_score = float('inf')
            for move in available_moves:
                board[move] = self.opponent
                score = self.minimax(board, depth + 1, True, alpha, beta)
                board[move] = ' '
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Poda Alpha-Beta
            self.memo[board_state] = best_score  # Guardamos el resultado en cache
            return best_score

    def best_move(self, board):
        best_score = -float('inf')
        move = None
        prioritized_moves = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # Orden inteligente

        for i in prioritized_moves:
            if board[i] == ' ':
                board[i] = self.letter
                score = self.minimax(board, 0, False, -float('inf'), float('inf'))
                board[i] = ' '
                if score > best_score:
                    best_score = score
                    move = i
        return move

    def evaluate_board(self, board):
        """ Función heurística para evaluar el tablero en profundidad máxima """
        score = 0
        win_patterns = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
            [0, 4, 8], [2, 4, 6]             # Diagonales
        ]

        for pattern in win_patterns:
            values = [board[i] for i in pattern]
            if values.count(self.letter) == 2 and values.count(' ') == 1:
                score += 5  # Premia jugadas ganadoras
            if values.count(self.opponent) == 2 and values.count(' ') == 1:
                score -= 5  # Penaliza jugadas peligrosas

        return score


def train_agent(env, agent, num_episodes=1000):
    print("Entrenando agente con recompensas densas")

    rival = MinMaxRival()
    for _ in tqdm(range(num_episodes)):
        state = env.reset()
        while True:
            # Movimiento del rival MinMax
            rival_action = rival.best_move(env.board)
            env.make_move(rival_action, 'O')
            next_state = env.get_state()

            if env.game_over():
                reward = calculate_reward(env, rival_action, 'O')
                agent.update(state, rival_action, reward, next_state, True)
                break

            # Movimiento del agente Q-learning
            action = agent.choose_action(state, env.available_moves())
            env.make_move(action, 'X')
            next_state = env.get_state()

            # Calcular recompensa basada en la acción tomada
            reward = calculate_reward(env, action, 'X')
            done = env.game_over()

            agent.update(state, action, reward, next_state, done)

            if done:
                break  # Termina la partida

            state = next_state

    print("Entrenamiento completado con recompensas densas")
    return agent


def calculate_reward(env, action, player):
    """
    Calcula la recompensa para una acción específica.
    - `env`: instancia del juego TicTacToe.
    - `action`: movimiento realizado.
    - `player`: 'X' (agente) o 'O' (rival).
    """
    reward = 0
    board = env.board.copy()  # Copiamos el tablero actual

    # Si el juego terminó después de esta jugada, asignamos recompensa final
    if env.current_winner == player:
        return 20  # Ganó la partida
    elif env.current_winner is not None:
        return -20  # Perdió la partida

    # **Bonificaciones y penalizaciones estratégicas**
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
        [0, 4, 8], [2, 4, 6]             # Diagonales
    ]

    # Si puso ficha en el centro → bonificación
    if action == 4:
        reward += 1

    # Si puso ficha en una posición lateral → pequeña penalización
    if action in [1, 3, 5, 7]:
        reward -= 1

    for pattern in win_patterns:
        values = [board[i] for i in pattern]

        # Bonificación si coloca una ficha en línea con otra suya
        for i in range(len(pattern) - 1):
            if values[i] == player and values[i+1] == player:
                reward += 0.3  # Jugada ofensiva buena
                break  # Salir después de encontrar la primera pareja consecutiva

        # Bonificación si bloquea una línea del oponente
        opponent = 'O' if player == 'X' else 'X'
        if values.count(opponent) == 2 and values.count(' ') == 1:
            # Verificamos si la acción es en la posición crítica
            empty_space = pattern[values.index(' ')]  # La posición vacía en la línea
            if empty_space == action:
                reward += 0.2  # Bloqueo útil

        # Penalización si deja una línea del oponente sin bloquear
        if values.count(opponent) == 2 and action not in pattern:
            reward -= 0.2  # No bloqueó al rival

    return reward


def get_player_move():
    while True:
        row_move = int(input("\nSeleccione una fila (1,3): "))
        if row_move >= 1 and row_move <= 3:
            column_move = int(input("Seleccione una columna (1,3): "))
            if column_move >= 1 and column_move <= 3:
                return (3*(row_move-1)) + (column_move-1)
            else:
                print("La columna seleccionada está fuera de rango\n")
        else:
            print("La fila seleccionad está fuera de rango\n")


def show_board(env, turn):
    if turn == 0:
        print("Tablero")
    elif turn == 1:
        print("Tablero (movimiento humano):")
    elif turn == 2:
        print("Tablero (moviemiento agente):")

    print(env.board[:3])
    print(env.board[3:6])
    print(env.board[6:])


if __name__ == "__main__":
    # Entrenamiento del agente
    env = TicTacToe()
    # training_agent = QLearningAgent(alpha=0.01)
    # training_agent.load_q_table(WEIGHTS_PATH)

    # num_episodes = 10000000
    # trained_agent = train_agent(env, training_agent, num_episodes)

    # trained_agent.save_q_table(WEIGHTS_PATH)

    # Jugar contra el agente
    print("\n\nJuega contra el agente de RL!\n")
    # agent = MinMaxRival()
    agent = QLearningAgent(epsilon=0.0)
    agent.load_q_table(WEIGHTS_PATH)
    env.reset()
    show_board(env, 0)

    while not env.game_over():   
        # Q-Learning move
        ai_move = agent.choose_action(env.get_state(), env.available_moves())
        # # Minmax move
        # ai_move = agent.best_move(env.board)
        env.make_move(ai_move, 'X')
        show_board(env, 2)
        
        if env.game_over():
            break
        
        player_move = get_player_move()
        env.make_move(player_move, 'O')
        show_board(env, 1)

    if env.is_full() and not env.get_winner():
        print("Juego terminado! Empate")
    else:
        print(f"Juego terminado! Ganador {env.get_winner()}")