import sys
sys.path.append('')
from agents import bots
import plotly.express as px
import pandas as pd


class Evaluator:
    """
    Simulate games and visualize outcome.
    """

    def __init__(self, stats=None, num_games=500):
        self.num_games = num_games
        if stats:
            self.outcomes = stats.user_results
            self.user_choices = stats.user_choices
            self.bot_choices = stats.bot_choices
        else:
            self.outcomes = []
            self.user_choices = []
            self.bot_choices = []

        self.bots = []

    def simulate(self, user, bot):
        """
        Simulate a game. Both opponents are bots.
        """
        self.bot = bot
        self.user = user
        for step in range(self.num_games):
            user_choice = self.user.play()
            bot_choice = self.bot.play()
            self.bot.update(user_choice=user_choice, bot_choice=bot_choice)
            self.user.update(user_choice=bot_choice, bot_choice=user_choice)
            user_result = bots.get_user_result(user_choice, bot_choice)
            self.outcomes.append(user_result)
            self.user_choices.append(user_choice)
            self.bot_choices.append(bot_choice)

        user_wins = len([i for i in self.outcomes if i == 'win'])
        self.win_ratio = user_wins / len(self.outcomes)


    def plot_win_ratio(self):
        df = pd.DataFrame({'Game': range(1, len(self.outcomes) + 1), 'Outcome': self.outcomes})

        df['Win'] = df['Outcome'].eq('win').cumsum() / df.Game
        df['Tie'] = df['Outcome'].eq('tie').cumsum() / df.Game

        fig = px.line(df, x='Game', y=['Win', 'Tie'], title=f"Win and Tie Ratio Over Games Played",
                      labels={'value': 'Ratio', 'variable': 'Outcome Type'})
        fig.update_yaxes(range=[0, 1])
        fig.add_hline(y=0.33, line_dash="dash", line_color="orange")
        fig.update_layout(
            xaxis_title='Game Number',
            yaxis_title='Ratio',
            width=800,
            height=400,
            margin=dict(l=50, r=20, t=30, b=50),
        )


        return fig

    def plot_run(self, labels=['User', 'Bot']):
        user_choices = self.user_choices
        bot_choices = self.bot_choices

        data = {
            'Index': list(range(len(user_choices))) + list(range(len(bot_choices))),
            'Choice': user_choices + bot_choices,
            'Type': ['User'] * len(user_choices) + ['Bot'] * len(bot_choices),
            'y': [1] * len(user_choices) + [-1] * len(bot_choices),
        }

        df = pd.DataFrame(data)

        color_map = {'rock': 'red', 'paper': 'blue', 'scissors': 'green'}

        fig = px.scatter(df, x='Index', y='y', color='Choice',
                         title=f'{labels[0]} vs {labels[1]} Choices',
                         labels={'Choice': 'Choice'},
                         color_discrete_sequence=list(color_map.values()))
        fig.update_layout(yaxis=dict(ticktext=labels,
                                     tickvals=[1, -1]))
        return fig
