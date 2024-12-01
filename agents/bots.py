import random
import numpy as np
import itertools
from abc import ABC, abstractmethod


def find_winner(user_choice, bot_choice):
    wins_against = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
    if user_choice == bot_choice:
        return 'tie'
    elif wins_against[user_choice] == bot_choice:
        return 'user'
    else:
        return 'bot'


def get_user_result(user_choice, bot_choice):
    winner_to_result = {'tie': 'tie', 'user': 'win', 'bot': 'loose'}
    winner = find_winner(user_choice, bot_choice)
    return winner_to_result[winner]


def get_reward(user_choice, bot_choice):
    winner = find_winner(user_choice, bot_choice)
    reward = {'user': 1, 'bot':0, 'tie': 0}
    return reward[winner]


class Bot(ABC):
    """
    Abstract base class for rock paper scissor playing bots
    """

    def __init__(self):
        self.choices = ["rock", "paper", "scissors"]
        self.result_history = []
        self.bot_history = []
        self.user_history = []
        self.name = None

    @abstractmethod
    def play(self):
        pass

    @abstractmethod
    def update(self, user_choice, bot_choice):
        pass


class RandomBot(Bot):
    """
    Simple, but unbeatable in the long run: Play random option.
    """
    def __init__(self):
        super().__init__()
        self.name = "Random"

    def play(self):
        return random.choice(self.choices)

    def update(self, user_choice, bot_choice):
        return


class SameBot(Bot):
    """
    Pick first option randomly, then keep playing the same.
    """
    def __init__(self):
        super().__init__()
        self.name = 'Same'

    def play(self):
        if not self.bot_history:
            return random.choice(self.choices)
        else:
            return self.bot_history[-1]

    def update(self, user_choice, bot_choice):
        self.bot_history.append(bot_choice)
        self.user_history.append(user_choice)


class LoopBot(Bot):
    """
    Always pick the next option in the order Rock, Paper, Scissors.
    """
    def __init__(self):
        super().__init__()
        self.name = 'Loop'

    def play(self):
        return self.choices[len(self.user_history) % len(self.choices)]

    def update(self, user_choice, bot_choice):
        self.user_history.append(user_choice)


class WSLS(Bot):
    """
    Win stay, loose shift: If the last game was won, repeat. Else shift to another option.
    """
    def __init__(self):
        super().__init__()
        self.name = "WSLS"

    def play(self):
        if not self.result_history or self.result_history[-1] == 'tie':
            return random.choice(self.choices)
        bot_won_last_game = self.result_history[-1] == 'bot'
        last_choice = self.bot_history[-1]
        if bot_won_last_game:
            return last_choice

        return random.choice([choice for choice in self.choices if choice != last_choice])

    def update(self, user_choice, bot_choice):
        winner = find_winner(user_choice, bot_choice)
        self.result_history.append(winner)
        self.bot_history.append(bot_choice)


class ForwardBackward(Bot):
    """
    This strategy can be found in the article:
    Paper-Rock-Scissors: an exploration of the dynamics of players’ strategies
    """

    def __init__(self):
        super().__init__()
        self.name = "ForwardBackward"

    def play(self):
        if not self.result_history or self.result_history[-1] == 'tie':
            return random.choice(self.choices)
        bot_won_last_game = self.result_history[-1] == 'bot'
        last_choice_index = self.choices.index(self.bot_history[-1])
        if bot_won_last_game:
            forward = (last_choice_index + 1) % 3
            return self.choices[forward]
        else:
            backward = (last_choice_index - 1) % 3
            return self.choices[backward]

    def update(self, user_choice, bot_choice):
        winner = find_winner(user_choice, bot_choice)
        self.result_history.append(winner)
        self.bot_history.append(bot_choice)


class ThompsonSampling(Bot):
    """
    Use Thompson Sampling to learn which of the three options has the best winning probability.
    """

    def __init__(self):
        super().__init__()
        self.name = "ThompsonSampling"
        self.alpha = {c:1 for c in self.choices}
        self.beta = {c:1 for c in self.choices}

    def play(self):
        theta = np.random.beta(list(self.alpha.values()), list(self.beta.values()))
        index = theta.argmax()
        return self.choices[index]

    def update(self, user_choice, bot_choice):
        reward = get_reward(user_choice, bot_choice)
        self.alpha[bot_choice] += reward # successes
        self.beta[bot_choice] += 1 - reward # fails


class ThompsonSamplingHistory(Bot):
    """
    Use Thompson Sampling to learn which option has the best winning probability,
    but track winning probabilities for options depending on the previous round.
    """

    def __init__(self):
        super().__init__()
        self.name = "ThompsonSamplingHistory"
        self.states = list(itertools.product(self.choices, self.choices))  # user / bot
        self.alpha = {(state, choice): 1 for state, choice in itertools.product(self.states, self.choices)}
        self.beta = {(state, choice): 1 for state, choice in itertools.product(self.states, self.choices)}

    def play(self):
        if not self.user_history:
            return random.choice(self.choices)
        last_state = (self.user_history[-1], self.bot_history[-1])
        state_action_candidates = [(last_state, choice) for choice in self.choices]
        alphas = [self.alpha[sac] for sac in state_action_candidates]
        betas = [self.beta[sac] for sac in state_action_candidates]
        theta = np.random.beta(alphas, betas)
        index = theta.argmax()
        return state_action_candidates[index][1]

    def update(self, user_choice, bot_choice):
        reward = get_reward(user_choice, bot_choice)
        if self.user_history:
            last_state = (self.user_history[-1], self.bot_history[-1])
            key = (last_state, bot_choice)
            self.alpha[key] += reward  # successes
            self.beta[key] += 1 - reward  # fails

        self.bot_history.append(bot_choice)
        self.user_history.append(user_choice)


class QLearningAgent(Bot):
    """
    Q learner.
    Differences to Thompson History:
    - exploration is done with either with probability epsilon (epsilon_greedy) or
        with a softmax depending in q values (soft_max)
    - we don't try to only predict the reward after the current action, but assume that this action also
      affects rewards after that.
    - there is no bayesian interpretation for learning rate updates.
    """
    def __init__(self, learning_rate=0.15, epsilon=0.2, temperature=0.2, discount=0.5, exploration_strategy='soft_max'):
        super().__init__()
        self.name = "QL"
        self.states = list(itertools.product(self.choices, self.choices))  # last action by user / bot
        self.q_table = np.zeros((len(self.states), len(self.choices)))
        self.lr = learning_rate
        self.gamma = discount
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.temperature = temperature

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def play(self):
        if not self.user_history :
            index = random.randint(0, self.q_table.shape[1] - 1)
        else:
            if self.exploration_strategy == 'epsilon_greedy':
                if random.uniform(0, 1) < self.epsilon: # Explore
                    index = random.randint(0, self.q_table.shape[1] - 1)
                else:
                    state = (self.user_history[-1], self.bot_history[-1])
                    state_index = self.states.index(state)
                    index = np.argmax(self.q_table[state_index])  # Exploit
            else: #'softmax'
                state = (self.user_history[-1], self.bot_history[-1])
                state_index = self.states.index(state)
                q_values = self.q_table[state_index]
                probabilities = self.softmax(q_values / self.temperature)
                # Choose action based on probabilities
                index = np.random.choice(len(self.choices), p=probabilities)
        return self.choices[index]

    def update(self, user_choice, bot_choice):
        if self.user_history:
            reward = get_reward(user_choice, bot_choice)
            last_state = (self.user_history[-1], self.bot_history[-1])
            last_index = self.states.index(last_state)
            new_state = (user_choice, bot_choice)
            new_index = self.states.index(new_state)
            action_index = self.choices.index(bot_choice)
            self.update_q(state=last_index, action=action_index, reward=reward, next_state=new_index)

        self.bot_history.append(bot_choice)
        self.user_history.append(user_choice)

    def update_q(self, state, action, reward, next_state):
        """
        Estimate from this step for the q / value function of state:
          reward + discounted value of next state (if the best next state is chosen based on current value estimate)
        We replace our current estimate with a mix of the current and the new estimate, where the weight of the new
        estimate is given by the learning rate
        """
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

    def train(self, episodes):
        for episode in episodes:
            self.bot_history, self.user_history = [], []
            for user_choice, bot_choice in zip(episode['user_choices'], episode['bot_choices']):
                self.update(user_choice, bot_choice)

    def save(self, fname):
        np.savetxt(fname, self.q_table)

    def load(self, fname):
        self.q_table = np.loadtxt(fname)