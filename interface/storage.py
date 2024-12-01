import os
import pandas as pd
import logging


def add_file_tracking(logger, fname):
    file_handler = logging.FileHandler(fname, mode='a')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


class Tracking:
    """
    Input/Output module:
    - write statistics to logs
    - manage persistence of high scores
    """

    def __init__(self, logger):
        self.logger = logger
        self.high_scores = []
        self.hs_path = './data/high_scores_{}.csv'

    def track_outcomes(self, stats):
        self.logger.info(stats.format_stats())

    def load_high_scores(self, new_score, selected_bot):
        fname = self.hs_path.format(selected_bot)
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            self.high_scores = df.to_dict('records')
        else:
            self.high_scores = []
        new_score['bot'] = selected_bot
        self.high_scores = [new_score] + self.high_scores

    def highscores_for_display(self):

        def highlight_first_row(val):
            return 'background-color: rgba(100, 149, 237, 0.2);'

        df = pd.DataFrame(self.high_scores)
        sorted_df = df.sort_values(by='wilson score', ascending=False)
        sorted_df['rank'] = range(1, len(sorted_df)+1)
        first_row = sorted_df.loc[[0]]
        top_10 = sorted_df[sorted_df.index != 0].head(10)
        highscores = pd.concat([first_row, top_10])
        styled_df = highscores.style.applymap(highlight_first_row, subset=(0, slice(None)))
        return styled_df

    def save_high_scores(self, selected_bot):
            df = pd.DataFrame(self.high_scores).sort_values(by='wilson score', ascending=False)
            fname = self.hs_path.format(selected_bot)
            df.head(100).to_csv(fname, index=False)