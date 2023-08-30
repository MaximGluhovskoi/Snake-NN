import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 2000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # control for randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft() when reaches MAX_MEMORY
        self.model = Linear_QNet(14, 500, 3) # (input, hidden, output)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        point_l2 = Point(head.x - 40, head.y)
        point_r2 = Point(head.x + 40, head.y)
        point_u2 = Point(head.x, head.y - 40)
        point_d2 = Point(head.x, head.y + 40)


        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger straight + 1

            (dir_r and game.is_collision(point_r2)) or
            (dir_l and game.is_collision(point_l2)) or 
            (dir_u and game.is_collision(point_u2)) or
            (dir_d and game.is_collision(point_d2)),



            # danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or 
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            #danger right + 1

            (dir_r and game.is_collision(point_d2)) or
            (dir_l and game.is_collision(point_u2)) or 
            (dir_u and game.is_collision(point_r2)) or
            (dir_d and game.is_collision(point_l2)),



            # danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or 
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # dange left + 1
            (dir_r and game.is_collision(point_u2)) or
            (dir_l and game.is_collision(point_d2)) or 
            (dir_u and game.is_collision(point_l2)) or
            (dir_d and game.is_collision(point_r2)),



            # Current move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y
            ]
        #print(np.array(state, dtype=int))
        return np.array(state, dtype=int) # changes booleans to 0s and 1s

        

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popped left if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else: 
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration/ exploitation (at beginning want more random moves at end u want to use less random and more of model)
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # as games progress epsilon decreases lower how often random move happens
            move = random.randint(0, 2)
            final_move[move] = 1
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # finds which number in the 3 numbered list is the highest and returns the index
            final_move[move] = 1 

        return final_move


def train():
    plot_scores = []
    plot_mean_score = []
    plot_mean10_score = []
    plot_mean10 = []
    plot_records = []
    mean10 = 0
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory/experience replay
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            mean10 = 0
            plot_mean10_score.insert(0, score)
            if len(plot_mean10_score) > 10:
                plot_mean10_score[10] = 0
            for index in range(len(plot_mean10_score)):
                mean10 += plot_mean10_score[index]
            if len(plot_mean10_score) <= 10:
                mean10 = mean10/len(plot_mean10_score)
            else:
                mean10 = mean10/10
            plot_mean10.append(mean10)
            plot_records.append(record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/ agent.n_games
            plot_mean_score.append(mean_score)
            #print("length: ", len(plot_mean10_score), "size: ", mean10, "list: ", plot_mean10_score)
            plot(plot_scores, plot_mean_score, plot_records, plot_mean10)

            



if __name__ == '__main__':
    train()

