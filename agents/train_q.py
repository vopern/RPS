"""
Pretrain a q-learning agent from user logs
"""

import sys
import json
import re
from bots import QLearningAgent

def parse_line(line):
    pattern = r'\{.*\}'

    match = re.search(pattern, line)
    if match:
        data_str = match.group(0)
        data = json.loads(data_str)
        return data
    else:
        print("No valid dictionary found in the line")

def parse_logs(fname):
    episodes = []
    for line in open(fname, 'r').readlines():
        d = parse_line(line)
        episodes.append(d)
    print(f'Loaded {len(episodes)} episodes')
    return episodes


def filter_episodes(episodes):
    episodes = [e for e in episodes if len(e['bot_choices']) > 5]
    print(f' {len(episodes)} episodes after filtering')
    return episodes


if __name__ == '__main__':
    episodes = parse_logs('../data/logs.txt')
    episodes = filter_episodes(episodes)
    bot = QLearningAgent()
    bot.train(episodes)
    bot.save('../data/trained_agent.txt')

