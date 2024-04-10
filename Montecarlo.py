from argparse import Action
from audioop import avg
from os import access
from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, discountFactor=0.1):
        # Initalizing variables for our agent
        self.epsilon = epsilon
        self.alpha = alpha
        self.discountfactor = discountFactor
        self.actions = [0,1]
        self.learningRate = 0.1
        self.highestScore = 0
        self.scores = []
        self.av_scores = {'AverageScore': [], 'Iterations': []}
        self.soft = (1 - self.epsilon) + self.epsilon / 2

        # Key: (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel, action), Value: Q value 
        self.Qtable = {} # Holds the q value for each state that has happ

        # Key: (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel), Value: Reward
        self.policy_table = {} # Holds all of the states with action that we have seen

        # (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel, action), Value: Reward
        self.episode_states = {} # Holds all of the states with action that we have seen each episode 
        
        # STATES in dictionaries:
            # (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)

        # Populate the tables, otherwise we get index error
        for player_y in range(0, 15): # player_y
            for next_pipe_top_y in range(0, 15): # next_pipe_top_y
                for next_pipe_dist_to_player in range(0, 15): # next_pipe_dist_to_player
                    for player_vel in range(0, 19): # player_vel
                        for action in range(0, 2): # action
                            Qstate = (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel, action)                           
                            self.Qtable[Qstate] = 0.0
                        # state = (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)
                            self.policy_table[Qstate] = 0.0
        

    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        
        # (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)
        # Changed the observe because we observer after each episode
        
        # Check if it is the end
        s1 = self.state_filter(s1)
        vel = s1["player_vel"]
        yPos = s1["player_y"]
        pipeDist = s1["next_pipe_dist_to_player"]
        pipeTopY = s1["next_pipe_top_y"]
        state = (yPos, pipeTopY, pipeDist, vel,a)
        self.episode_states[state] = r # In each frame update the reward for that particular state
        
        if end: # Observer what we learned during the episode
            # First for loop
            # Second for loop
            for state, reward in self.episode_states.items():
                if reward != 0: # Since we populated before we have all of the states set as 0
                    # Update the Qtable 
                    self.Qtable[state] = (1 - self.discountfactor) * self.Qtable[state] + self.learningRate
                    
                    # Check if the flap state is higher or not from the Qtable
                    flapState = (state[0], state[1], state[2], state[3], 0)
                    noFlapState = (state[0], state[1], state[2], state[3], 1)
                    
                    if self.Qtable[noFlapState] >= self.Qtable[flapState]:
                        # Update the policy table depending on formula or Q value
                        self.policy_table[state] =  self.soft
                    else:
                        self.policy_table[state] = self.epsilon / 2
            self.episode_states = {} # Reset the states dictionary 
        return

    def state_filter(self, state):
        # next_pipe_dist_to_player: 288pixels/15intervals = 19 per interval
        # player_y = 512pixels/15 intervals = 34 per interval
        # next_pipe_top_y = 512pixels/15 intervals = 34 per interval

        if state == 'terminal':
            return state
        new_state = {}

        player_y = math.floor(state['player_y']/34)
        if player_y < 0:
            player_y = 0

        next_pipe_top_y = math.floor(state['next_pipe_top_y']/34)
        if math.floor(state['next_pipe_dist_to_player']/19) >= 15:
            next_pipe_dist_to_player = 14
        else:
            next_pipe_dist_to_player = math.floor(state['next_pipe_dist_to_player']/19)
        player_vel = state['player_vel']+8
        if player_vel >= 18:
            player_vel = 18
        new_state['player_vel'] = int(player_vel)
        new_state["player_y"] = player_y
        new_state["next_pipe_dist_to_player"] = next_pipe_dist_to_player
        new_state['next_pipe_top_y'] = next_pipe_top_y


        return new_state

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # Get the state in which the bird is in
        # Get the action if the bird should flap or no flap
        # Check which is higher and do that, but with the non greedy policy e.g. soft
        # (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)
        # TODO CHANGE
        state = self.state_filter(state)
        vel = state["player_vel"]
        yPos = state["player_y"]
        pipeDist = state["next_pipe_dist_to_player"]
        pipeTopY = state["next_pipe_top_y"]
        flapState = (yPos, pipeTopY, pipeDist, vel, 0)
        noFlapState = (yPos, pipeTopY, pipeDist, vel, 1)
        
        flap = self.policy_table[flapState]
        noFlap = self.policy_table[noFlapState]

        if flap < noFlap:
            return 1
        elif noFlap < flap:
            return 0
        else:
            if random.random() < self.epsilon:
                return 0
            else: return 1
        





    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        # Get the state which the bird is in and do which one is higher, else do random > 0.5
        state = self.state_filter(state)
        vel = state["player_vel"]
        yPos = state["player_y"]
        pipeDist = state["next_pipe_dist_to_player"]
        pipeTopY = state["next_pipe_top_y"]
        # (player_y, next_pipe_top_y, next_pipe_dist_to_player, player_vel)
        # print("In policy: ", state)
        flapState = (yPos, pipeTopY, pipeDist, vel, 0)
        noFapState = (yPos, pipeTopY, pipeDist, vel, 1)
        flap = self.policy_table[flapState]
        noFlap = self.policy_table[noFapState]

        if flap < noFlap:
            return 1
        elif noFlap > flap:
            return 0
        else:
            if random.random() < self.epsilon:
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
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
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
            # print("score for this episode: %d" % agent.highestScore)
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
            print(nb_episodes)
            # print("score for this episode: %d" % score)
            # Before resetting, add states to a_r_table for ai to learn
            env.reset_game()
            nb_episodes -= 1
            score = 0
            print(nb_episodes)


# agent = FlappyAgent()
# train(20000, agent)
# run_game(100, agent)
