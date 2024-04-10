from ple.games.flappybird import FlappyBird
from ple import PLE

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    scores = []

    # reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    # To speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        action = agent.policy(agent.state_filter(env.game.getGameState()))

        # step the environment
        reward = env.act(env.getActionSet()[action])

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
        # let the agent observe the current state transition
        newState = env.game.getGameState()
        ## Gera eh
        agent.observe(state, action, reward, newState, env.game_over())
        score += reward

        # reset the environment if the game is over
        if env.game_over():
            # print("score for this episode: %d" % score)
            print(nb_episodes)
            env.reset_game()
            nb_episodes -= 1
            score = 0

# from task3 import FlappyAgent

# agent = FlappyAgent() # Default
# for i in range(1,3):
#     train(100 * i, agent)
#     run_game(50, agent)
#     av_score = sum(agent.scores) / len(agent.scores)
#     agent.av_scores["AverageScore"].append(av_score)
#     agent.av_scores["Iterations"].append(i*100)

# print(len(agent.scores))