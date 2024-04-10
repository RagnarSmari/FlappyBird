from argparse import Action
from os import access
from ple.games.flappybird import FlappyBird
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from ple import PLE
import random
import math
import numpy as np
import matplotlib.pyplot as plt


### DOES NOT RUN!! ###


class NeuralNetworkAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, discountFactor=1.0):
        # Initalizing variables for our agent
        self.alpha = alpha
        self.epsilon = epsilon
        self.discountfactor = discountFactor
        self.actions = [0,1]
        self.scores = []
        self.highestScore = 0
        self.av_scores = {'AverageScore': [], 'Iterations': []} # Key = iteration, Value = av of the score
   
        self.model = MLPRegressor(hidden_layer_sizes=(100, 10), activation='logistic', learning_rate=0.01)
        dummy_data_X = np.zeros((15,15,15,19))
        dummy_data_Y = np.zeros((15,15,15,19)) 
        self.model.fit(dummy_data_X, dummy_data_Y)
        self.replay_buffer = []
        self.groundLevel = 0.79*512
        self.birdHeight = 24
        self.pipeDist = 283

        # Where key is state,action
        self.q_values = {}

        #Q Table \^o^/, where first column is x_dist, second is pipe_top_y, third is pipe_bottom_y,
        self.q_table = np.zeros((15, 15, 15, 19, 2))


        #States
        self.player_y = 0
        self.next_pipe_top_y = 0
        self.next_pipe_dist_to_player = 0
        self.player_vel = 0

    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def sig(self, x):
        # Sigmoid activation function
        return 1/(1+ np.exp(-x))
    
    def normalize(self, state):
        y = state['player_y']
        topY = state['next_pipe_top_y']
        pipeDist = state['next_pipe_dist_to_player']
        vel = state['player_vel']

        array = np.array([y, topY, pipeDist, vel])
        return preprocessing.normalize([array])

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """

        # Update the weights fo the network
        s2 = self.state_filter(s2)
        current_q = self.q_table[s1["player_y"],s1["next_pipe_top_y"],s1["next_pipe_dist_to_player"],s1["player_vel"], a]
        future_q = self.q_table[s2["player_y"],s2["next_pipe_top_y"],s2["next_pipe_dist_to_player"],s2["player_vel"], a]
        max_future_q = future_q.max()
        max_current_q = current_q.max()
        # new_q_value = current_q + self.alpha * (r + self.discountfactor * max_future_q -current_q)

        

        # Normalize the states
        s1Normal = self.normalize(s1)
        s2Normal = self.normalize(s2)


        self.replay_buffer.append(s1Normal)

        if len(self.replay_buffer) >= 1000:
            # Get 100 samples from the replay buffer
            X = []
            for _ in range(100):
                value = self.replay_buffer[random.randrange(0, len(self.replay_buffer))]
                X.append(value)
            current_q = (1 - self.alpha)* current_q + self.alpha * (r + self.discountfactor * max_current_q)
            future_q = (1 - self.alpha)* current_q + self.alpha * (r + self.discountfactor * max_future_q)
            y = (current_q, future_q)

            new_q_value = self.model.partial_fit(X, y)
        else:
            new_q_value = (1 - self.alpha)* current_q + self.alpha * (r + self.discountfactor * max_future_q)


        self.q_table[s1["player_y"],s1["next_pipe_top_y"],s1["next_pipe_dist_to_player"],s1["player_vel"], a] = new_q_value
        self.q_table[s1["player_y"],s1["next_pipe_top_y"],s1["next_pipe_dist_to_player"],s1["player_vel"], a] = new_q_value
        # Set the output vector y = (qflap, qnoflap)
            # where Qa = r + alpha * max'aQ(s2, a, o)
            # and qb = Q(s,b,o)




        return
    
    def state_filter(self, state):
        # next_pipe_dist_to_player: 288pixels/15intervals = 19 per interval
        # player_y = 512pixels/15 intervals = 34 per interval
        # next_pipe_top_y = 512pixels/15 intervals = 34 per interval

        if state == 'terminal':
            return state
        new_state = {}

        self.player_y = math.floor(state['player_y']/34)
        self.next_pipe_top_y = math.floor(state['next_pipe_top_y']/34)
        if math.floor(state['next_pipe_dist_to_player']/19) >= 15:
            self.next_pipe_dist_to_player = 14
        else:
             self.next_pipe_dist_to_player = math.floor(state['next_pipe_dist_to_player']/19)



        self.player_vel = state['player_vel']+8
        new_state['player_vel'] = int(self.player_vel)
        new_state["player_y"] = self.player_y
        new_state["next_pipe_dist_to_player"] = self.next_pipe_dist_to_player
        new_state['next_pipe_top_y'] = self.next_pipe_top_y


        return new_state
    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
         ##Er hærra q gildi að hoppa eða ekki hoppa med thvi ad setja state og 0 og state og 1


        # print("state in trianing: ",self.q_table[state["player_y"]])

        if random.random() < self.epsilon:
            #Pick random action
            if random.random() < 0.5:
                return 0
            else:
                return 1

        return self.policy(state)


        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        try:
            q_value_flap =  self.q_table[state["player_y"],state["next_pipe_top_y"],state["next_pipe_dist_to_player"],state["player_vel"], 0]
            q_value_no_flap = self.q_table[state["player_y"],state["next_pipe_top_y"],state["next_pipe_dist_to_player"],state["player_vel"], 1]
        except IndexError:
            print(state)
        if q_value_flap > q_value_no_flap:
            return 0
        else:
            return 1

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    scores = []

    # reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(agent.state_filter(env.game.getGameState()))

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition
        agent.highestScore = score
        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % agent.highestScore)
            env.reset_game()
            nb_episodes -= 1
            agent.scores.append(agent.highestScore)
            score = 0

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()
    score = 0

    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        state = agent.state_filter(state)
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        ## Gera eh
        agent.observe(state, action, reward, newState, env.game_over())
        score += reward

        # reset the environment if the game is over
        if env.game_over():
            # print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


#TODO Láta forritið testa á X episodes fresti og geyma Score-ið fyrir hvert X, þegar forritið er búið að keyra að grapha það upp

# agent = FlappyAgent()
# for i in range(1,10):
#     train(5000 * i, agent)
#     run_game(100, agent)
#     av_score = sum(agent.scores) / len(agent.scores)
#     agent.av_scores["AverageScore"].append(av_score)
#     agent.av_scores["Iterations"].append(i*5000)

# import pandas as pd
# import seaborn as sns

# dataframe = pd.DataFrame.from_dict(agent.av_scores)
# sns.lineplot(data=dataframe, x='Iterations', y='AverageScore')
# plt.show()
# agent = NeuralNetworkAgent()
# train(100, agent)
# run_game(10,agent)