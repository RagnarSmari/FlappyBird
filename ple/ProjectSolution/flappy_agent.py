from argparse import Action
from os import access
from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import math
import numpy as np


class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        self.alpha = 0.1
        self.epsilon = 0.1
        self.discountfactor = 0.1
        self.actions = [0,1]
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
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        #ST+1 er newstate
        # if end:
        #     s2 = "terminal"
        # Update Q
        s2 = self.state_filter(s2)
        current_q = self.q_table[s1["player_y"],s1["next_pipe_top_y"],s1["next_pipe_dist_to_player"],s1["player_vel"], a]
        future_q = self.q_table[s2["player_y"],s2["next_pipe_top_y"],s2["next_pipe_dist_to_player"],s2["player_vel"], a]
        max_future_q = future_q.max()
        new_q_value = (1 - self.alpha)* current_q + self.alpha * (r + self.discountfactor * max_future_q)
        # new_q_value = current_q + self.alpha * (r + self.discountfactor * max_future_q -current_q)

        self.q_table[s1["player_y"],s1["next_pipe_top_y"],s1["next_pipe_dist_to_player"],s1["player_vel"], a] = new_q_value

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

        # # print('state in statefilter ', state)
        # if state == 'terminal':
        #     return state
        # new_state = {}
        
        # velocity = state['player_vel']+8
        # # print('velocity ',int(velocity))
        # new_state['player_vel'] = int(velocity)
        
        # y_values = ['player_y', 'next_pipe_top_y']
        
        # for field in y_values:
        #     if state[field] <= 0.0:
        #         new_state[field] = 0
        #     elif state[field] > self.groundLevel - self.birdHeight:
        #         new_state[field] = 14
        #     else:
        #         new_state[field] = int(state[field] * 14 // (self.groundLevel - self.birdHeight))
            
        # field = 'next_pipe_dist_to_player'
        # if state[field] > 288:
        #     state[field] = 288
            


        # if state[field] <= 0.0:
        #     new_state[field] = 0
        # elif state[field] > self.pipeDist:
        #     new_state[field] = 14
        # else:
        #     new_state[field] = int(state[field] * 14 // self.pipeDist)
        
        # return new_state

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
            return 0
        # print('ToFlap: ', q_value_flap)
        # print('ToNotFlap', q_value_no_flap)
        if q_value_flap > q_value_no_flap:
            return 0
        else:
            return 1


