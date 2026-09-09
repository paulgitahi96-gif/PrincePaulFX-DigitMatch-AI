"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
DERIV HISTORICAL TICKS
VERSION 2.0
============================================================

Purpose:
    Download a large historical tick dataset from Deriv
    using multiple ticks_history requests.

Version 2.0 improvements:

    - Handles the 1,000-tick request limit safely.
    - Downloads multiple chronological batches.
    - Retrieves market pip_size.
    - Uses pip_size for last-digit extraction.
    - Preserves trailing-zero precision.
    - Deduplicates historical ticks.
    - Sorts ticks chronologically.
    - Saves pip_size in the CSV.
    - Supports datasets larger than 1,000 ticks.

This module:
    - uses public market data
    - requires no API token
    - does not place trades
    - does not train models
    - does not execute recovery strategies

============================================================
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import websockets


# ============================================================
# CONNECTION
# ============================================================

PUBLIC_WEBSOCKET_URL = (
    "wss://api.derivws.com/"
    "trading/v1/options/ws/public"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MARKET = "1HZ100V"

DEFAULT_COUNT = 5000

MAX_BATCH_SIZE = 1000

REQUEST_TIMEOUT = 30

MAX_EMPTY_BATCHES = 3


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class HistoricalTick:

    symbol: str

    quote: float

    epoch: int

    last_digit: int

    pip_size: int

    raw: dict


# ============================================================
# PIP SIZE
# ============================================================

def normalize_pip_size(
    pip_size,
) -> Optional[int]:
    """
    Convert Deriv pip_size information into the number
    of decimal places.

    Deriv may expose pip_size as:

        1
        0.1
        0.01
        0.001
        ...

    or in some responses as an integer precision:

        1
        2
        3
        ...

    The active_symbols response normally provides pip_size
    as a decimal price increment.

    Examples:

        1       -> 0
        0.1     -> 1
        0.01    -> 2
        0.001   -> 3
    """

    if pip_size is None:
        return None

    try:

        value = float(
            pip_size
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if value <= 0:
        return None

    # Decimal pip increment.
    if value < 1:

        decimal_places = int(
            round(
                -math.log10(
                    value
                )
            )
        )

        if (
            decimal_places >= 0
            and abs(
                value
                - (
                    10 ** (
                        -decimal_places
                    )
                )
            )
            < 1e-9
        ):
            return decimal_places

    # Some responses/tools can expose precision
    # directly as an integer.
    if value.is_integer():

        integer_value = int(
            value
        )

        if 0 <= integer_value <= 10:
            return integer_value

    return None


# ============================================================
# LAST DIGIT EXTRACTION
# ============================================================

def extract_last_digit(
    quote,
    pip_size: Optional[int] = None,
) -> int:
    """
    Extract the actual final displayed decimal digit.

    pip_size is preferred because Python floats do not preserve
    trailing zeros.

    Example:

        quote = 123.4
        pip_size = 2

    Displayed price:

        123.40

    Correct last digit:

        0
    """

    if quote is None:
        raise ValueError(
            "Quote is empty."
        )

    # --------------------------------------------------------
    # Preferred method: Deriv precision
    # --------------------------------------------------------

    if pip_size is not None:

        try:

            decimal_quote = Decimal(
                str(quote)
            )

            scale = Decimal(
                10
            ) ** int(
                pip_size
            )

            scaled = (
                decimal_quote
                * scale
            )

            integer_scaled = int(
                scaled.to_integral_value()
            )

            return abs(
                integer_scaled
            ) % 10

        except (
            InvalidOperation,
            ValueError,
            TypeError,
        ):
            pass

    # --------------------------------------------------------
    # Fallback method
    # --------------------------------------------------------

    quote_string = str(
        quote
    ).strip()

    if not quote_string:
        raise ValueError(
            "Empty quote."
        )

    if "e" in quote_string.lower():

        try:

            quote_string = format(
                float(quote),
                "f",
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                f"Unable to process quote: "
                f"{quote}"
            )

    # Remove sign.
    quote_string = (
        quote_string
        .lstrip("-+")
    )

    # Decimal portion.
    if "." in quote_string:

        decimal_part = (
            quote_string
            .split(".", 1)[1]
        )

        digits = [
            character
            for character in decimal_part
            if character.isdigit()
        ]

        if digits:

            return int(
                digits[-1]
            )

    # Integer fallback.
    digits = [
        character
        for character in quote_string
        if character.isdigit()
    ]

    if not digits:
        raise ValueError(
            f"Unable to extract "
            f"last digit from quote: "
            f"{quote}"
        )

    return int(
        digits[-1]
    )


# ============================================================
# ACTIVE SYMBOL / PIP SIZE LOOKUP
# ============================================================

async def fetch_market_pip_size(
    websocket,
    market: str,
) -> Optional[int]:
    """
    Ask Deriv for active-symbol metadata and determine
    the decimal precision for the selected market.
    """

    request = {
        "active_symbols": "brief",
        "product_type": "basic",
    }

    await websocket.send(
        json.dumps(request)
    )

    deadline = (
        asyncio.get_running_loop().time()
        + REQUEST_TIMEOUT
    )

    while True:

        remaining = (
            deadline
            - asyncio.get_running_loop().time()
        )

        if remaining <= 0:

            raise TimeoutError(
                "Timed out waiting for "
                "active_symbols response."
            )

        try:

            raw_message = (
                await asyncio.wait_for(
                    websocket.recv(),
                    timeout=remaining,
                )
            )

        except asyncio.TimeoutError:

            raise TimeoutError(
                "Timed out waiting for "
                "active_symbols response."
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

        if "error" in message:

            error = message[
                "error"
            ]

            if isinstance(
                error,
                dict,
            ):

                error_message = error.get(
                    "message",
                    "Unknown Deriv API error.",
                )

            else:

                error_message = str(
                    error
                )

            raise RuntimeError(
                "Deriv active_symbols "
                f"request failed: "
                f"{error_message}"
            )

        if (
            message.get("msg_type")
            != "active_symbols"
        ):
            continue

        symbols = message.get(
            "active_symbols"
        )

        if not isinstance(
            symbols,
            list,
        ):

            raise RuntimeError(
                "Deriv returned an invalid "
                "active_symbols response."
            )

        for item in symbols:

            if not isinstance(
                item,
                dict,
            ):
                continue

            symbol = item.get(
                "underlying_symbol"
            )

            # Backward-compatible field.
            if symbol is None:

                symbol = item.get(
                    "symbol"
                )

            if str(symbol) != str(
                market
            ):
                continue

            raw_pip_size = item.get(
                "pip_size"
            )

            precision = normalize_pip_size(
                raw_pip_size
            )

            return precision

        raise ValueError(
            f"Market '{market}' was not "
            "found in active_symbols."
        )


# ============================================================
# HISTORY REQUEST
# ============================================================

async def request_history_batch(
    websocket,
    market: str,
    count: int,
    end,
) -> tuple[list[HistoricalTick], Optional[int]]:
    """
    Request one historical batch.

    Returns:

        ticks
        pip_size
    """

    request = {
        "ticks_history": market,
        "count": int(count),
        "end": end,
        "style": "ticks",
        "subscribe": 0,
    }

    await websocket.send(
        json.dumps(request)
    )

    deadline = (
        asyncio.get_running_loop().time()
        + REQUEST_TIMEOUT
    )

    while True:

        remaining = (
            deadline
            - asyncio.get_running_loop().time()
        )

        if remaining <= 0:

            raise TimeoutError(
                "Timed out waiting for "
                "historical tick data."
            )

        try:

            raw_message = (
                await asyncio.wait_for(
                    websocket.recv(),
                    timeout=remaining,
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

        if "error" in message:

            error = message[
                "error"
            ]

            if isinstance(
                error,
                dict,
            ):

                error_message = error.get(
                    "message",
                    "Unknown Deriv API error.",
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

        if (
            message.get("msg_type")
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
                "Invalid history response."
            )

        prices = history.get(
            "prices"
        )

        times = history.get(
            "times"
        )

        symbol = history.get(
            "symbol",
            market,
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

        # ----------------------------------------------------
        # pip_size can be returned by ticks_history.
        # ----------------------------------------------------

        response_pip_size = (
            normalize_pip_size(
                message.get(
                    "pip_size"
                )
            )
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
                        quote,
                        response_pip_size,
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
                        pip_size=(
                            response_pip_size
                            or 0
                        ),
                        raw=message,
                    )
                )

            except (
                TypeError,
                ValueError,
                InvalidOperation,
            ):

                continue

        records.sort(
            key=lambda item: item.epoch
        )

        return (
            records,
            response_pip_size,
        )


# ============================================================
# HISTORICAL CLIENT
# ============================================================

class DerivHistoricalTicks:
    """
    Multi-batch Deriv historical tick downloader.
    """

    def __init__(
        self,
        market: str = DEFAULT_MARKET,
    ):

        self.market = market

        self.websocket = None

        self.pip_size: Optional[int] = None


    async def fetch(
        self,
        count: int = DEFAULT_COUNT,
    ) -> list[HistoricalTick]:

        if count < 20:

            raise ValueError(
                "count must be at least 20."
            )

        if count > 50000:

            raise ValueError(
                "count cannot exceed 50000."
            )

        print()
        print(
            "Connecting to Deriv public "
            "historical data..."
        )

        all_ticks: dict[
            tuple[int, float],
            HistoricalTick
        ] = {}

        async with websockets.connect(
            PUBLIC_WEBSOCKET_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as websocket:

            self.websocket = websocket

            # ------------------------------------------------
            # Determine market precision.
            # ------------------------------------------------

            try:

                self.pip_size = (
                    await fetch_market_pip_size(
                        websocket,
                        self.market,
                    )
                )

            except Exception as error:

                print(
                    "Warning: unable to obtain "
                    f"pip_size from active_symbols: "
                    f"{error}"
                )

                self.pip_size = None

            if self.pip_size is not None:

                print(
                    f"Market pip precision: "
                    f"{self.pip_size} decimal places"
                )

            else:

                print(
                    "Market pip precision: "
                    "not available; using fallback"
                )

            # ------------------------------------------------
            # Download backwards in batches.
            # ------------------------------------------------

            remaining = int(
                count
            )

            end = "latest"

            batch_number = 0

            empty_batches = 0

            while remaining > 0:

                batch_number += 1

                batch_size = min(
                    MAX_BATCH_SIZE,
                    remaining,
                )

                print(
                    f"Downloading batch "
                    f"{batch_number}: "
                    f"{batch_size} ticks..."
                )

                batch, response_pip_size = (
                    await request_history_batch(
                        websocket=websocket,
                        market=self.market,
                        count=batch_size,
                        end=end,
                    )
                )

                if response_pip_size is not None:

                    self.pip_size = (
                        response_pip_size
                    )

                if not batch:

                    empty_batches += 1

                    print(
                        "Received an empty batch."
                    )

                    if (
                        empty_batches
                        >= MAX_EMPTY_BATCHES
                    ):

                        print(
                            "Stopping after "
                            f"{MAX_EMPTY_BATCHES} "
                            "empty batches."
                        )

                        break

                    continue

                empty_batches = 0

                # ------------------------------------------------
                # Deduplicate.
                #
                # Epoch + quote is used as the tick identity.
                # ------------------------------------------------

                before = len(
                    all_ticks
                )

                for tick in batch:

                    identity = (
                        tick.epoch,
                        tick.quote,
                    )

                    all_ticks[
                        identity
                    ] = tick

                added = (
                    len(all_ticks)
                    - before
                )

                print(
                    f"Batch returned: "
                    f"{len(batch)}"
                )

                print(
                    f"New unique ticks: "
                    f"{added}"
                )

                print(
                    f"Total unique ticks: "
                    f"{len(all_ticks)}"
                )

                # ------------------------------------------------
                # Stop when enough ticks exist.
                # ------------------------------------------------

                if len(all_ticks) >= count:
                    break

                # ------------------------------------------------
                # Move historical endpoint backwards.
                # ------------------------------------------------

                oldest_epoch = min(
                    tick.epoch
                    for tick in batch
                )

                # Ask for data before the oldest tick
                # in the current batch.
                end = max(
                    1,
                    oldest_epoch - 1,
                )

                remaining = (
                    count
                    - len(all_ticks)
                )

            # ----------------------------------------------------
            # Convert dictionary to chronological list.
            # ----------------------------------------------------

            records = sorted(
                all_ticks.values(),
                key=lambda item: (
                    item.epoch,
                    item.quote,
                ),
            )

            # ----------------------------------------------------
            # Trim to requested count.
            #
            # We downloaded backwards from latest, so keep
            # the newest requested number after sorting.
            # ----------------------------------------------------

            if len(records) > count:

                records = records[
                    -count:
                ]

            print()
            print(
                "Historical download complete."
            )

            print(
                f"Requested: "
                f"{count}"
            )

            print(
                f"Received unique: "
                f"{len(records)}"
            )

            if records:

                print(
                    f"Earliest epoch: "
                    f"{records[0].epoch}"
                )

                print(
                    f"Latest epoch: "
                    f"{records[-1].epoch}"
                )

            return records


    def fetch_sync(
        self,
        count: int = DEFAULT_COUNT,
    ) -> list[HistoricalTick]:

        return asyncio.run(
            self.fetch(
                count=count
            )
        )


# ============================================================
# CONVERSION HELPERS
# ============================================================

def ticks_to_digits(
    ticks: list[HistoricalTick],
) -> list[int]:

    return [
        tick.last_digit
        for tick in ticks
    ]


def ticks_to_records(
    ticks: list[HistoricalTick],
) -> list[dict]:

    return [
        {
            "symbol": tick.symbol,
            "quote": tick.quote,
            "epoch": tick.epoch,
            "last_digit": tick.last_digit,
            "pip_size": tick.pip_size,
        }
        for tick in ticks
    ]


# ============================================================
# SAVE CSV
# ============================================================

def save_ticks_csv(
    ticks: list[HistoricalTick],
    path: str | Path,
) -> Path:

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


# ============================================================
# LOAD DIGITS
# ============================================================

def load_digits_csv(
    path: str | Path,
) -> list[int]:

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

            digit = int(
                value
            )

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


# ============================================================
# DOWNLOAD + SAVE
# ============================================================

def download_and_save(
    market: str = DEFAULT_MARKET,
    count: int = DEFAULT_COUNT,
    output_path: str | Path = (
        "data/historical_ticks.csv"
    ),
) -> Path:

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


# ============================================================
# DIAGNOSTIC SUMMARY
# ============================================================

def print_digit_summary(
    ticks: list[HistoricalTick],
) -> None:

    if not ticks:

        print(
            "No ticks available "
            "for digit summary."
        )

        return

    from collections import Counter

    counts = Counter(
        tick.last_digit
        for tick in ticks
    )

    print()
    print(
        "=" * 64
    )

    print(
        "DIGIT DISTRIBUTION CHECK"
    )

    print(
        "=" * 64
    )

    total = len(
        ticks
    )

    for digit in range(
        10
    ):

        count = counts.get(
            digit,
            0,
        )

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"Digit {digit}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )

    print()

    print(
        f"Total ticks: {total}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "PRINCE PAUL FX "
            "Deriv multi-batch "
            "historical tick downloader"
        )
    )

    parser.add_argument(
        "--market",
        default=DEFAULT_MARKET,
        help=(
            "Deriv market symbol. "
            f"Default: {DEFAULT_MARKET}"
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "Number of historical ticks "
            f"to download. "
            f"Default: {DEFAULT_COUNT}"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/historical_ticks.csv"
        ),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    print(
        "=" * 64
    )

    print(
        "PRINCE PAUL FX"
    )

    print(
        "HISTORICAL TICK DOWNLOADER"
    )

    print(
        "VERSION 2.0"
    )

    print(
        "=" * 64
    )

    print(
        f"Market: "
        f"{args.market}"
    )

    print(
        f"Requested ticks: "
        f"{args.count}"
    )

    try:

        client = DerivHistoricalTicks(
            market=args.market
        )

        ticks = client.fetch_sync(
            count=args.count
        )

        if not ticks:

            raise RuntimeError(
                "No historical ticks were received."
            )

        output = save_ticks_csv(
            ticks,
            args.output,
        )

        print_digit_summary(
            ticks
        )

        print()
        print(
            "=" * 64
        )

        print(
            "DATASET SAVED"
        )

        print(
            "=" * 64
        )

        print(
            f"File: {output}"
        )

        print(
            f"Ticks saved: {len(ticks)}"
        )

        print(
            f"Market: {args.market}"
        )

    except KeyboardInterrupt:

        print()
        print(
            "Download interrupted."
        )

    except Exception as error:

        print()
        print(
            "=" * 64
        )

        print(
            "ERROR"
        )

        print(
            "=" * 64
        )

        print(
            error
        )
