import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st

from agents import bots
from streamlit.logger import get_logger
import uuid
from stats import GameStats
from storage import Tracking, add_file_tracking
from interface.evaluation import Evaluator


st.set_page_config(
    page_title="RPS", page_icon="✂️", initial_sidebar_state="auto"
)

@st.cache_resource
def get_my_logger():
    logger = get_logger(__name__)
    logger = add_file_tracking(logger, './data/logs.txt')
    return logger

logger = get_my_logger()

emojis = {'rock': '🪨', 'paper': '📄', 'scissors':'✂️'}

st.markdown("""
<style>
.stButton > button {
    height: 100px;
}
</style>
""", unsafe_allow_html=True)


def reset_all():
    if 'bot' in st.session_state:
        del st.session_state['bot']

    if 'stats' in st.session_state:
        del st.session_state['stats']


def main():

    st.title("Rock Paper Scissors")
    st.subheader('Can you beat the bot?')
    st.sidebar.image('./interface/logo.png')
    st.sidebar.markdown('')

    bot_options = ['', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7']
    selected_bot = st.selectbox(label='Choose your opponent', options=bot_options, index=0, on_change=reset_all,
                                help="A game with not-too-complicated agents. "
                                     "Some strategies you might guess. Others are self-learning, good luck! "
                                     "Here's the [Repository](https://github.com/vopern/RPS).")

    if "session_id" not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())

    if "bot" not in st.session_state and selected_bot != '':
        if selected_bot == 'Level 1':
            bot = bots.SameBot()
        elif selected_bot == 'Level 2':
            bot = bots.LoopBot()
        elif selected_bot == 'Level 3':
            bot = bots.ForwardBackward()
        elif selected_bot == 'Level 4':
            bot = bots.WSLS()
        elif selected_bot == 'Level 5':
            bot = bots.ThompsonSampling()
        elif selected_bot == 'Level 6':
            bot = bots.ThompsonSamplingHistory()
        elif selected_bot == 'Level 7':
            bot = bots.QLearningAgent(learning_rate=0.4, epsilon=0.1)
            bot.load('./data/trained_agent.txt')
        else:
            bot = bots.RandomBot()
        st.session_state.bot = bot

    result_message = {'win': ':blue[You win!]', 'loose': ':red[Bot wins!]', 'tie': "It's a tie!"}

    if "stats" not in st.session_state:
        st.session_state.stats = GameStats(st.session_state['session_id'], selected_bot)
    if "tracker" not in st.session_state:
        st.session_state.tracker = Tracking(logger)

    st.text("")

    user_choice = None
    col1, col2, col3 = st.columns([1., 1., 1.])

    with col1:
        rock = st.button(f'{emojis["rock"]}', disabled=selected_bot == '', use_container_width=True)
    with col2:
        paper = st.button(f"{emojis['paper']}", disabled=selected_bot == '', use_container_width=True)
    with col3:
        scissors = st.button(f"{emojis['scissors']}", disabled=selected_bot == '', use_container_width=True)

    if rock:
        user_choice = 'rock'
    elif paper:
        user_choice = 'paper'
    elif scissors:
        user_choice = 'scissors'

    if user_choice:
        with st.expander(label='Outcome', expanded=True):
            computer_choice = st.session_state.bot.play()
            result = bots.get_user_result(user_choice, computer_choice)

            col1, col2, col3 = st.columns([1., 1., 1.])
            with col1:
                st.markdown(f"You played:")
                st.markdown(f"### {user_choice} {emojis[user_choice]} ")

            with col2:
                st.write(result_message[result])
            with col3:
                st.markdown(f"Bot played:")
                st.markdown(f"### {computer_choice} {emojis[computer_choice]}")

            st.session_state.bot.update(user_choice, computer_choice)
            st.session_state.stats.update(user_choice, computer_choice)


    with st.expander(label='', expanded=True):
        col1, col2, col3 = st.columns([1., 1., 1.])
        wins, losses = st.session_state.stats.result_stats['win'], st.session_state.stats.result_stats['loose']
        c1, c2 = ('blue', 'red') if wins >= losses else ('red', 'blue')
        with col1:
            st.write(f"You")
            st.write(f"### :{c1}[{wins}]")
        with col2:
            st.write("Ties")
            st.write(f"### {st.session_state.stats.result_stats['tie']}")
        with col3:
            st.write("Bot")
            st.write(f"### :{c2}[{losses}]")

    if st.button("Evaluate & Reset"):
        st.session_state.tracker.load_high_scores(st.session_state.stats.calc_score(), selected_bot)
        eval = Evaluator(stats=st.session_state.stats)
        t1, t2, t3 = st.tabs(["Win Ratio", "Choices", "High Scores" ])
        with t1:
            st.plotly_chart(eval.plot_win_ratio(), use_container_width=False, theme="streamlit")
        with t2:
            st.plotly_chart(eval.plot_run(), use_container_width=False, theme="streamlit")
        with t3:
            st.subheader('High Scores')
            st.dataframe(st.session_state.tracker.highscores_for_display(), hide_index=True)
            st.write("Order takes into account "
                     "number of wins as well as number of games "
                     "(Wilson score confidence interval lower bound).")

        st.session_state.tracker.track_outcomes(st.session_state.stats)
        st.session_state.tracker.save_high_scores(selected_bot)
        reset_all()


if __name__ == "__main__":

    main()
