"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
DERIV HISTORICAL TICKS
VERSION 1.0
============================================================

Purpose:
    Download historical Deriv ticks through the public
    WebSocket and convert them into a clean digit dataset.

This module:
    - uses public market data
    - requires no API token
    - requests historical ticks
    - extracts the last digit
    - removes malformed records
    - preserves chronological order

It does NOT:
    - place trades
    - use account credentials
    - train models
    - execute recovery strategies

============================================================
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import websockets


# ==========================================================
# CONFIGURATION
# ==========================================================

PUBLIC_WEBSOCKET_URL = (
    "wss://api.derivws.com/"
    "trading/v1/options/ws/public"
)

DEFAULT_MARKET = "1HZ100V"

DEFAULT_COUNT = 5000

REQUEST_TIMEOUT = 30


# ==========================================================
# HISTORICAL TICK
# ==========================================================

@dataclass
class HistoricalTick:

    symbol: str

    quote: float

    epoch: int

    last_digit: int

    raw: dict


# ==========================================================
# DIGIT EXTRACTION
# ==========================================================

def extract_last_digit(
    quote,
) -> int:
    """
    Extract the final decimal digit without allowing
    normal float formatting to accidentally change the
    displayed decimal representation.
    """

    quote_string = str(
        quote
    ).strip()

    if not quote_string:

        raise ValueError(
            "Empty quote."
        )

    if "e" in quote_string.lower():

        quote_string = format(
            float(quote),
            "f",
        )

    digit_characters = [
        character
        for character in quote_string
        if character.isdigit()
    ]

    if not digit_characters:

        raise ValueError(
            f"Unable to extract digit from "
            f"quote: {quote}"
        )

    return int(
        digit_characters[-1]
    )


# ==========================================================
# NORMALIZE RESPONSE
# ==========================================================

def normalize_tick(
    message: dict,
) -> Optional[HistoricalTick]:
    """
    Convert a Deriv tick response into HistoricalTick.
    """

    tick_data = message.get(
        "history"
    )

    if not isinstance(
        tick_data,
        dict,
    ):

        return None

    symbol = tick_data.get(
        "symbol"
    )

    prices = tick_data.get(
        "prices"
    )

    times = tick_data.get(
        "times"
    )

    if (
        symbol is None
        or prices is None
        or times is None
    ):

        return None

    return None


# ==========================================================
# HISTORICAL DATA CLIENT
# ==========================================================

class DerivHistoricalTicks:
    """
    Historical tick-data downloader.

    Uses the public Deriv WebSocket.
    """

    def __init__(
        self,
        market: str = DEFAULT_MARKET,
    ):

        self.market = market

        self.websocket = None


    # ======================================================
    # FETCH HISTORY
    # ======================================================

    async def fetch(
        self,
        count: int = DEFAULT_COUNT,
    ) -> list[HistoricalTick]:
        """
        Request historical ticks for the selected market.

        Returns records in chronological order.
        """

        if count < 20:

            raise ValueError(
                "count must be at least 20."
            )

        if count > 50000:

            raise ValueError(
                "count cannot exceed 50000."
            )

        request = {
            "ticks_history": self.market,
            "count": int(count),
            "end": "latest",
            "style": "ticks",
        }

        async with websockets.connect(
            PUBLIC_WEBSOCKET_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as websocket:

            self.websocket = websocket

            await websocket.send(
                json.dumps(request)
            )

            while True:

                try:

                    raw_message = (
                        await asyncio.wait_for(
                            websocket.recv(),
                            timeout=REQUEST_TIMEOUT,
                        )
                    )

                except asyncio.TimeoutError:

                    raise TimeoutError(
                        "Timed out waiting for "
                        "historical tick data."
                    )

                if isinstance(
                    raw_message,
                    bytes,
                ):

                    raw_message = (
                        raw_message.decode(
                            "utf-8"
                        )
                    )

                try:

                    message = json.loads(
                        raw_message
                    )

                except json.JSONDecodeError:

                    continue

                # ------------------------------------------
                # API error
                # ------------------------------------------

                if "error" in message:

                    error = message[
                        "error"
                    ]

                    if isinstance(
                        error,
                        dict,
                    ):

                        error_message = (
                            error.get(
                                "message",
                                "Unknown Deriv API error.",
                            )
                        )

                    else:

                        error_message = str(
                            error
                        )

                    raise RuntimeError(
                        "Deriv historical tick "
                        f"request failed: "
                        f"{error_message}"
                    )

                # ------------------------------------------
                # History response
                # ------------------------------------------

                if (
                    message.get(
                        "msg_type"
                    )
                    != "history"
                ):

                    continue

                history = message.get(
                    "history"
                )

                if not isinstance(
                    history,
                    dict,
                ):

                    raise RuntimeError(
                        "Deriv returned an invalid "
                        "history response."
                    )

                prices = history.get(
                    "prices"
                )

                times = history.get(
                    "times"
                )

                symbol = history.get(
                    "symbol",
                    self.market,
                )

                if not isinstance(
                    prices,
                    list,
                ):

                    raise RuntimeError(
                        "Historical response does "
                        "not contain prices."
                    )

                if not isinstance(
                    times,
                    list,
                ):

                    raise RuntimeError(
                        "Historical response does "
                        "not contain times."
                    )

                if len(prices) != len(times):

                    raise RuntimeError(
                        "Historical prices and "
                        "timestamps have different "
                        "lengths."
                    )

                records = []

                for quote, epoch in zip(
                    prices,
                    times,
                ):

                    try:

                        quote_float = float(
                            quote
                        )

                        last_digit = (
                            extract_last_digit(
                                quote
                            )
                        )

                        records.append(
                            HistoricalTick(
                                symbol=str(
                                    symbol
                                ),
                                quote=quote_float,
                                epoch=int(
                                    epoch
                                ),
                                last_digit=(
                                    last_digit
                                ),
                                raw=message,
                            )
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                # ------------------------------------------
                # Chronological order
                # ------------------------------------------

                records.sort(
                    key=lambda item:
                    item.epoch
                )

                return records


    # ======================================================
    # SYNCHRONOUS WRAPPER
    # ======================================================

    def fetch_sync(
        self,
        count: int = DEFAULT_COUNT,
    ) -> list[HistoricalTick]:
        """
        Synchronous wrapper for Streamlit and scripts.
        """

        return asyncio.run(
            self.fetch(
                count=count
            )
        )


# ==========================================================
# CONVERT TO DIGITS
# ==========================================================

def ticks_to_digits(
    ticks: list[HistoricalTick],
) -> list[int]:
    """
    Convert historical ticks into a digit sequence.
    """

    return [
        tick.last_digit
        for tick in ticks
    ]


# ==========================================================
# CONVERT TO RECORDS
# ==========================================================

def ticks_to_records(
    ticks: list[HistoricalTick],
) -> list[dict]:
    """
    Convert tick objects into serializable dictionaries.
    """

    return [
        {
            "symbol": tick.symbol,
            "quote": tick.quote,
            "epoch": tick.epoch,
            "last_digit": tick.last_digit,
        }
        for tick in ticks
    ]


# ==========================================================
# SAVE CSV
# ==========================================================

def save_ticks_csv(
    ticks: list[HistoricalTick],
    path: str | Path,
) -> Path:
    """
    Save historical ticks as CSV.
    """

    import pandas as pd

    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        ticks_to_records(
            ticks
        )
    )

    dataframe.to_csv(
        destination,
        index=False,
    )

    return destination


# ==========================================================
# LOAD DIGITS FROM CSV
# ==========================================================

def load_digits_csv(
    path: str | Path,
) -> list[int]:
    """
    Load a previously saved digit dataset.
    """

    import pandas as pd

    source = Path(
        path
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Tick dataset not found: "
            f"{source}"
        )

    dataframe = pd.read_csv(
        source
    )

    if "last_digit" not in dataframe.columns:

        raise ValueError(
            "CSV does not contain "
            "'last_digit' column."
        )

    digits = []

    for value in dataframe[
        "last_digit"
    ].tolist():

        try:

            digit = int(value)

        except (
            TypeError,
            ValueError,
        ):

            continue

        if 0 <= digit <= 9:

            digits.append(
                digit
            )

    return digits


# ==========================================================
# DOWNLOAD AND SAVE
# ==========================================================

def download_and_save(
    market: str = DEFAULT_MARKET,
    count: int = DEFAULT_COUNT,
    output_path: str | Path = (
        "data/historical_ticks.csv"
    ),
) -> Path:
    """
    Download historical ticks and save them to CSV.
    """

    client = DerivHistoricalTicks(
        market=market
    )

    ticks = client.fetch_sync(
        count=count
    )

    if not ticks:

        raise RuntimeError(
            "No historical ticks were received."
        )

    path = save_ticks_csv(
        ticks,
        output_path,
    )

    return path


# ==========================================================
# DIAGNOSTIC
# ==========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "PRINCE PAUL FX "
            "Deriv historical tick downloader"
        )
    )

    parser.add_argument(
        "--market",
        default=DEFAULT_MARKET,
        help="Deriv market symbol.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of historical ticks.",
    )

    parser.add_argument(
        "--output",
        default="data/historical_ticks.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    print(
        "================================================"
    )

    print(
        "PRINCE PAUL FX"
    )

    print(
        "HISTORICAL TICK DOWNLOADER"
    )

    print(
        "================================================"
    )

    print(
        f"Market: {args.market}"
    )

    print(
        f"Requested ticks: {args.count}"
    )

    try:

        output = download_and_save(
            market=args.market,
            count=args.count,
            output_path=args.output,
        )

        print(
            f"\nSaved dataset to: {output}"
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )
