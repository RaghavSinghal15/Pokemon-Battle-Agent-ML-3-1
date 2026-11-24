"""The explanation dashboard (F13, #81): step through one saved game, turn by turn.

    python -m pokeagent.dashboard                     # pick a game recorded by the demo
    python -m pokeagent.dashboard path/to/demo.jsonl  # or open one directly

Each turn shows what the agent saw and why it chose its move: the matchup and our predicted
win probability, the opponent forecast and which features drove it, the beliefs about the
opponent's hidden set, the full payoff table, and the choice with its explanation. Games are
recorded by scripts/demo_explanations.py (one JSON line per turn, agent/explain.py).
"""

import json
import pickle
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pokeagent.config import OPPONENT_MODEL_PATH, REPORTS_DIR
from pokeagent.dashboard.contributions import forecast_contributions, top_contributions

EXAMPLES_DIR = REPORTS_DIR / "examples"


@st.cache_data
def load_turns(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@st.cache_resource
def load_model():
    if not OPPONENT_MODEL_PATH.is_file():
        return None
    with OPPONENT_MODEL_PATH.open("rb") as stream:
        return pickle.load(stream)


def pick_trace() -> Path | None:
    arguments = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    if arguments:
        return Path(arguments[0])
    traces = sorted(EXAMPLES_DIR.glob("*.jsonl"), reverse=True)
    if not traces:
        return None
    return st.sidebar.selectbox("Recorded game", traces, format_func=lambda p: p.name)


def replay_for(trace: Path, battle: str) -> Path | None:
    """The HTML replay poke-env saved next to the trace for this battle, if any."""
    matches = sorted(trace.parent.glob(f"*{battle}*.html"))
    return matches[0] if matches else None


def payoff_frame(turn: dict) -> pd.DataFrame:
    table = turn["table"]
    frame = pd.DataFrame(table["values"], index=table["rows"], columns=table["columns"])
    probs = table["column_probs"]
    if probs is not None:
        frame.insert(0, "expected", frame.to_numpy() @ probs)
    frame.insert(0, "worst case", frame[table["columns"]].min(axis=1))
    return frame


def main() -> None:
    st.set_page_config(page_title="Agent explanations", layout="wide")
    trace = pick_trace()
    if trace is None:
        st.error("No recorded game found. Record one with: python scripts/demo_explanations.py")
        return
    turns = load_turns(str(trace))
    battles = sorted({t["battle"] for t in turns})
    battle = st.sidebar.selectbox("Battle", battles) if len(battles) > 1 else battles[0]
    turns = [t for t in turns if t["battle"] == battle]
    numbers = [t["turn"] for t in turns]
    number = st.sidebar.select_slider("Turn", options=numbers)
    turn = next(t for t in turns if t["turn"] == number)

    me, opp = turn["me"], turn["opp"]
    st.title(
        f"Turn {number}: {me['species']} ({me['hp']:.0%}) vs {opp['species']} ({opp['hp']:.0%})"
    )
    top = st.columns(3)
    if turn["win_probability"] is not None:
        top[0].metric("Predicted P(win)", f"{turn['win_probability']:.0%}")
    top[1].metric("Strategy", turn["strategy"].replace("_", " "))
    top[2].metric("Chose", turn["chosen"])

    history = pd.DataFrame(
        {"turn": [t["turn"] for t in turns], "P(win)": [t["win_probability"] for t in turns]}
    ).set_index("turn")
    if history["P(win)"].notna().any():
        st.caption("Predicted win probability over the game")
        st.line_chart(history, height=160)

    left, right = st.columns(2)
    with left:
        st.subheader("Opponent forecast")
        st.bar_chart(pd.Series(turn["forecast"], name="probability"), height=200)
        model = load_model()
        if model is not None and turn.get("features"):
            contributions = forecast_contributions(model, turn["features"])
            if contributions:
                predicted = max(turn["forecast"], key=turn["forecast"].get)
                target = st.radio(
                    "Why? Feature contributions for",
                    [c for c in contributions if c != "hidden"],
                    index=[c for c in contributions if c != "hidden"].index(predicted),
                    horizontal=True,
                )
                top_features = pd.DataFrame(
                    top_contributions(contributions[target]), columns=["feature", "contribution"]
                )
                chart = (
                    alt.Chart(top_features)
                    .mark_bar()
                    .encode(
                        x=alt.X("contribution:Q", title="contribution (raw score)"),
                        y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                        color=alt.condition(
                            "datum.contribution > 0", alt.value("#2e7d32"), alt.value("#c62828")
                        ),
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    "SHAP values from LightGBM: positive pushed this class up, negative pushed"
                    " it down (raw scores, before the softmax)."
                )
    with right:
        st.subheader(f"Beliefs about {opp['species']}")
        beliefs = turn["beliefs"] or {}
        for title, key in (("Role", "roles"), ("Moves it may have", "moves"), ("Item", "items")):
            if beliefs.get(key):
                st.caption(title)
                st.dataframe(
                    pd.Series(beliefs[key], name="probability").to_frame(),
                    width="stretch",
                )

    st.subheader("Payoff table")
    frame = payoff_frame(turn)
    styled = frame.style.format("{:+.3f}").background_gradient(cmap="RdYlGn", axis=None)
    styled = styled.apply(
        lambda row: ["font-weight: bold" if row.name == turn["chosen"] else "" for _ in row],
        axis=1,
    )
    st.dataframe(styled, width="stretch")
    if turn["table"]["column_probs"] is not None:
        st.caption(
            "Columns are the opponent's possible actions; 'expected' weights them by the"
            " forecast (best response picks its highest row)."
        )
    if turn["mixture"] and turn["strategy"] == "lp":
        st.caption("Minimax mixture")
        st.bar_chart(pd.Series(turn["mixture"], name="probability"), height=180)

    st.subheader("Explanation")
    st.write(turn["explanation"])

    replay = replay_for(trace, battle)
    if replay is not None:
        with st.expander("Replay of the whole game"):
            components.html(replay.read_text(encoding="utf-8"), height=650, scrolling=True)


main()
