import numpy as np
from agents import bots
import json


class GameStats:

    def __init__(self, session_id = '', selected_bot = ''):
        self.user_choices = []
        self.bot_choices = []
        self.user_results = []
        self.result_stats = {"win": 0, "loose": 0, "tie": 0}
        self.session_id = session_id
        self.bot = selected_bot

    def update(self, user_choice, bot_choice):
        self.user_choices.append(user_choice)
        self.bot_choices.append(bot_choice)
        result = bots.get_user_result(user_choice, bot_choice)
        self.user_results.append(result)
        self.result_stats[result] += 1

    def format_stats(self):
        output = {"user_choices": self.user_choices,
                  "bot_choices": self.bot_choices,
                  "session_id": self.session_id,
                  "result_stats":self.result_stats,
                  "bot": self.bot}
        return json.dumps(output)

    def wilson_score_lower_bound(self, positive, total):
        """
        Wilson score confidence interval: balance positive ratio and confidence in
        observed ratio, see
        https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        """
        if total == 0:
            return 0
        z = 1.96  # 95% confidence interval
        phat = positive / total
        denominator = 1 + z ** 2 / total
        numerator = phat + z ** 2 / (2 * total) - z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * total)) / total)
        return numerator / denominator

    def calc_score(self):
        num_rounds = len(self.user_choices)
        results = {"number rounds": num_rounds,
                   "win ratio": self.result_stats["win"] / num_rounds if num_rounds > 0 else 0,
                   "wilson score": self.wilson_score_lower_bound(self.result_stats["win"], num_rounds),
                   "session id": self.session_id}
        return results