from train_run import  train, run_game
from NN import NeuralNetworkAgent
from Qlearning import QlearningAgent
from Montecarlo import MonteCarloAgent
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def train_and_plot(iterations,agent, name_of_plot="Name", title="Title"):

    for i in range(1,iterations):
        train(5000, agent)
        run_game(300, agent)
        av_score = sum(agent.scores) / len(agent.scores)
        print(av_score)
        agent.av_scores["AverageScore"].append(av_score)
        agent.av_scores["Iterations"].append(5000 * i+1)

    dataframe = pd.DataFrame.from_dict(agent.av_scores)
    print(dataframe)
    plot = sns.lineplot(data=dataframe, x='Iterations', y='AverageScore')
    plt.title(title)
    fig = plot.get_figure()
    fig.savefig(name_of_plot)
    # plt.close()


# Qagent = QlearningAgent()
# train_and_plot(6, Qagent, "Qlearning", "Qlearning")

# If you want multiple line, each line for one agent, then run those at the same time
# If you want a plot with only one agent/line the only run that one and comment out the others
Qagent = QlearningAgent(discountFactor=0.8)
train_and_plot(6, Qagent, "QlearningDiscountFactor08", "Qlearning")

Qagent = QlearningAgent(discountFactor=0.8, epsilon=0.01)
train_and_plot(6, Qagent, "QlearningDis0.8&Eps001", "Qlearning")

Qagent = QlearningAgent(discountFactor=0.8, epsilon=0.01, alpha=0.01)
train_and_plot(6, Qagent, "QlearningDis08&Eps001&Alpha001", "Qlearning")

Magent = MonteCarloAgent()
train_and_plot(6, Magent, "MonteCarlo", "MonteCarlo")

Magent = MonteCarloAgent(epsilon=0.01)
train_and_plot(6, Magent, "MonteCarlo&Eps001", "Monte Carlo")




























