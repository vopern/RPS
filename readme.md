#  Learning How to Play Rock-Paper-Scissors

A project on strategies for playing rock-paper-scissors. The hypothesis is that 
regularities in human game play can be learnt by a simple agent. 
After all, algorithms can also adapt to people's interests on video platforms.

General idea:
- Implement some strategies and multi-armed bandit / reinforcement learning agents which can adapt to them.
- Implement a streamlit UI to collect data from played games.
- Train an agent.
- See if something interesting is learned.

## Project structure

### Interface
Streamlit ui and helper modules.
Run locally from root directory with `streamlit run ./interface/streamlit_ui.py`

### Jupyter 
A notebook on strategies and agents.

### Agents
Module to implement agents / strategies and a script to train a simple agent.

### Data
Store in files:
- Trained agent.
- Logs from streamlit application.
- High scores from played rounds.