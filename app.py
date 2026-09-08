"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
STREAMLIT FOUNDATION + LIVE DERIV TICKS
VERSION 1.1
============================================================
"""

import asyncio
import threading
import time
from queue import Empty, Queue

import pandas as pd
import streamlit as st

from deriv.websocket import (
    DerivWebSocketClient,
    DEFAULT_MARKET,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PRINCE PAUL FX - DigitMatch AI",
    page_icon="👑",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "tick_queue": None,

    "tick_thread": None,

    "stream_running": False,

    "connection_status": "DISCONNECTED",

    "latest_quote": "--",

    "latest_digit": "--",

    "latest_epoch": "--",

    "tick_buffer": [],

    "predicted_digit": None,

    "prediction_probability": 0.0,

    "scanner_status": "WAIT",

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.title(
    "👑 PRINCE PAUL FX"
)

st.subheader(
    "DIGITMATCH AI ANALYZER"
)

st.caption(
    "Real-time Deriv tick stream • "
    "50-tick analysis window • "
    "LightGBM + XGBoost architecture"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ ENGINE SETTINGS"
    )

    market = st.selectbox(
        "🎯 Target Market",
        [
            "1HZ10V",
            "R_10",
            "R_25",
            "R_50",
            "R_75",
            "R_100",
        ],
    )

    window_size = st.number_input(
        "📊 Tick Window",
        min_value=20,
        max_value=500,
        value=50,
        step=10,
    )

    st.divider()

    st.header(
        "📡 LIVE CONNECTION"
    )

    st.info(
        "This first live version uses "
        "Deriv's public market-data stream. "
        "No trading credentials are required."
    )


# ============================================================
# TICK WORKER
# ============================================================

def run_tick_worker(
    selected_market: str,
    queue: Queue,
):
    """
    Runs the asynchronous Deriv stream in a
    background thread so Streamlit remains responsive.
    """

    async def worker():

        client = DerivWebSocketClient(
            market=selected_market,
            buffer_size=50,
        )

        try:

            async for tick in client.stream(
                authenticated=False
            ):

                queue.put(
                    {
                        "type": "connected",
                        "value": True,
                    }
                )

                queue.put(
                    {
                        "type": "tick",
                        "tick": tick,
                    }
                )

        except Exception as error:

            queue.put(
                {
                    "type": "error",
                    "value": str(error),
                }
            )

        finally:

            await client.stop()

            queue.put(
                {
                    "type": "stopped",
                    "value": True,
                }
            )

    asyncio.run(
        worker()
    )


# ============================================================
# START STREAM
# ============================================================

def start_stream(
    selected_market: str,
):

    if (
        st.session_state.tick_thread
        and st.session_state.tick_thread.is_alive()
    ):

        return

    tick_queue = Queue()

    st.session_state.tick_queue = (
        tick_queue
    )

    st.session_state.stream_running = True

    thread = threading.Thread(
        target=run_tick_worker,
        args=(
            selected_market,
            tick_queue,
        ),
        daemon=True,
    )

    st.session_state.tick_thread = thread

    thread.start()


# ============================================================
# STOP STREAM
# ============================================================

def stop_stream():

    st.session_state.stream_running = False

    st.session_state.connection_status = (
        "STOPPED"
    )


# ============================================================
# PROCESS QUEUED TICKS
# ============================================================

if st.session_state.tick_queue:

    while True:

        try:

            message = (
                st.session_state.tick_queue.get_nowait()
            )

        except Empty:

            break

        message_type = message.get(
            "type"
        )

        if message_type == "connected":

            st.session_state.connection_status = (
                "CONNECTED"
            )

        elif message_type == "error":

            st.session_state.connection_status = (
                "ERROR"
            )

        elif message_type == "stopped":

            st.session_state.connection_status = (
                "STOPPED"
            )

        elif message_type == "tick":

            tick = message["tick"]

            st.session_state.latest_quote = (
                tick.quote
            )

            st.session_state.latest_digit = (
                tick.last_digit
            )

            st.session_state.latest_epoch = (
                tick.epoch
            )

            st.session_state.tick_buffer.append(
                tick.last_digit
            )

            if len(
                st.session_state.tick_buffer
            ) > window_size:

                st.session_state.tick_buffer = (
                    st.session_state.tick_buffer[
                        -window_size:
                    ]
                )


# ============================================================
# CONTROL BUTTONS
# ============================================================

control_col1, control_col2 = st.columns(2)

with control_col1:

    if st.button(
        "🚀 START LIVE SCANNER",
        use_container_width=True,
    ):

        start_stream(
            market
        )

        st.rerun()


with control_col2:

    if st.button(
        "🛑 STOP SCANNER",
        use_container_width=True,
    ):

        stop_stream()

        st.rerun()


st.divider()


# ============================================================
# STATUS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Connection",
        st.session_state.connection_status,
    )

with c2:

    st.metric(
        "Market",
        market,
    )

with c3:

    st.metric(
        "Current Quote",
        st.session_state.latest_quote,
    )

with c4:

    st.metric(
        "Last Digit",
        st.session_state.latest_digit,
    )


# ============================================================
# LIVE DIGITS
# ============================================================

st.divider()

st.header(
    "🔢 Live Digit Stream"
)

if st.session_state.tick_buffer:

    digit_string = " ".join(
        str(digit)
        for digit in st.session_state.tick_buffer
    )

    st.code(
        digit_string
    )

else:

    st.info(
        "Press START LIVE SCANNER to receive "
        "real Deriv ticks."
    )


# ============================================================
# FREQUENCY
# ============================================================

st.header(
    "📊 Digit Distribution"
)

if st.session_state.tick_buffer:

    series = pd.Series(
        st.session_state.tick_buffer
    )

    counts = series.value_counts()

    total = len(series)

    rows = []

    for digit in range(10):

        count = int(
            counts.get(
                digit,
                0,
            )
        )

        percentage = (
            count / total * 100
            if total
            else 0
        )

        rows.append(
            {
                "Digit": digit,
                "Count": count,
                "Frequency": (
                    f"{percentage:.2f}%"
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Waiting for tick data..."
    )


# ============================================================
# AI PLACEHOLDER
# ============================================================

st.divider()

st.header(
    "🧠 AI Ensemble"
)

ai1, ai2, ai3 = st.columns(3)

with ai1:

    st.metric(
        "LightGBM",
        "NOT TRAINED",
    )

with ai2:

    st.metric(
        "XGBoost",
        "NOT TRAINED",
    )

with ai3:

    st.metric(
        "Ensemble",
        "WAIT",
    )


st.warning(
    "The live tick engine is active, but the AI models "
    "are intentionally not producing predictions yet. "
    "The next stage will create the real feature matrix "
    "and train LightGBM/XGBoost against historical ticks."
)


# ============================================================
# AUTO REFRESH WHILE RUNNING
# ============================================================

if st.session_state.stream_running:

    time.sleep(1)

    st.rerun()
