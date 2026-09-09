"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
DERIV HISTORICAL TICKS
VERSION 2.1
============================================================

Purpose:
    Download large historical Deriv tick datasets safely.

Version 2.1 fixes:

    - Removes obsolete active_symbols product_type.
    - Removes subscribe from one-time ticks_history requests.
    - Downloads in 1,000-tick batches.
    - Uses Deriv pip_size when available.
    - Preserves correct decimal precision.
    - Deduplicates ticks.
    - Maintains chronological order.
    - Saves complete historical datasets.
    - Provides digit-distribution diagnostics.

No authentication required.

This module does NOT:
    - place trades
    - access account funds
    - execute contracts
    - train models
    - implement recovery

============================================================
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

import websockets


# ============================================================
# DERIV CONNECTION
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
# DATA MODEL
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
# PIP SIZE NORMALIZATION
# ============================================================

def normalize_pip_size(
    pip_size,
) -> Optional[int]:
    """
    Convert Deriv pip_size into decimal precision.

    Examples:

        1       -> 0
        0.1     -> 1
        0.01    -> 2
        0.001   -> 3

    Current Deriv responses may also expose pip_size
    directly as decimal precision, for example:

        2
        3
        5

    In that case the integer value is used directly.
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

    if value < 0:
        return None

    # --------------------------------------------------------
    # Decimal increment format
    # --------------------------------------------------------

    if (
        value > 0
        and value < 1
    ):

        try:

            decimal_places = int(
                round(
                    -math.log10(
                        value
                    )
                )
            )

        except (
            ValueError,
            OverflowError,
        ):

            return None

        expected = (
            10 ** (
                -decimal_places
            )
        )

        if abs(
            value - expected
        ) < 1e-9:

            return decimal_places

    # --------------------------------------------------------
    # Direct precision format
    # --------------------------------------------------------

    if value.is_integer():

        precision = int(
            value
        )

        if 0 <= precision <= 10:

            return precision

    return None


# ============================================================
# LAST DIGIT EXTRACTION
# ============================================================

def extract_last_digit(
    quote,
    pip_size: Optional[int] = None,
) -> int:
    """
    Extract the actual final displayed digit.

    pip_size is used whenever available because converting
    prices through binary floating point can remove trailing
    zeros.

    Example:

        quote = 123.40
        precision = 2

    Result:

        0
    """

    if quote is None:

        raise ValueError(
            "Quote is empty."
        )

    # --------------------------------------------------------
    # Precision-aware extraction
    # --------------------------------------------------------

    if pip_size is not None:

        try:

            decimal_quote = Decimal(
                str(quote)
            )

            scale = (
                Decimal(10)
                ** int(pip_size)
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
    # Fallback extraction
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

    quote_string = (
        quote_string
        .lstrip("+-")
    )

    # Decimal part first.
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
# ACTIVE SYMBOL PRECISION
# ============================================================

async def fetch_market_pip_size(
    websocket,
    market: str,
) -> Optional[int]:
    """
    Retrieve the selected market's pip_size.

    IMPORTANT:
        The current Deriv API no longer accepts the old
        product_type parameter in active_symbols.

    Therefore the request is intentionally minimal:

        {
            "active_symbols": "brief"
        }
    """

    request = {
        "active_symbols": "brief",
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
                "active_symbols."
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
                "active_symbols."
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
                "Invalid active_symbols response."
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

            return normalize_pip_size(
                raw_pip_size
            )

        raise ValueError(
            f"Market '{market}' was not "
            "found in active_symbols."
        )


# ============================================================
# HISTORICAL BATCH REQUEST
# ============================================================

async def request_history_batch(
    websocket,
    market: str,
    count: int,
    end,
) -> tuple[
    list[HistoricalTick],
    Optional[int],
]:
    """
    Request one historical batch.

    IMPORTANT:
        subscribe is intentionally omitted.

    This is a one-time historical-data request.
    """

    request = {
        "ticks_history": market,
        "count": int(count),
        "end": end,
        "style": "ticks",
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
                "Historical response "
                "does not contain prices."
            )

        if not isinstance(
            times,
            list,
        ):

            raise RuntimeError(
                "Historical response "
                "does not contain times."
            )

        if len(prices) != len(times):

            raise RuntimeError(
                "Historical prices and "
                "timestamps have different "
                "lengths."
            )

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

                precision = (
                    response_pip_size
                    if response_pip_size
                    is not None
                    else None
                )

                last_digit = (
                    extract_last_digit(
                        quote,
                        precision,
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
                            precision
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
            HistoricalTick,
        ] = {}

        async with websockets.connect(
            PUBLIC_WEBSOCKET_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as websocket:

            self.websocket = websocket

            # ------------------------------------------------
            # Get market precision.
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
                    "Market precision: "
                    f"{self.pip_size} "
                    "decimal places"
                )

            else:

                print(
                    "Market precision: "
                    "not available; using "
                    "response/fallback precision"
                )

            # ------------------------------------------------
            # Download batches.
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

                print()
                print(
                    f"Downloading batch "
                    f"{batch_number}: "
                    f"{batch_size} ticks..."
                )

                (
                    batch,
                    response_pip_size,
                ) = await request_history_batch(
                    websocket=websocket,
                    market=self.market,
                    count=batch_size,
                    end=end,
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

                if len(all_ticks) >= count:

                    break

                # ------------------------------------------------
                # Move backwards.
                # ------------------------------------------------

                oldest_epoch = min(
                    tick.epoch
                    for tick in batch
                )

                end = max(
                    1,
                    oldest_epoch - 1,
                )

                remaining = (
                    count
                    - len(all_ticks)
                )

            records = sorted(
                all_ticks.values(),
                key=lambda item: (
                    item.epoch,
                    item.quote,
                ),
            )

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
# CONVERSION
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
# DIGIT DISTRIBUTION
# ============================================================

def print_digit_summary(
    ticks: list[HistoricalTick],
) -> None:

    if not ticks:

        print(
            "No ticks available."
        )

        return

    counts = Counter(
        tick.last_digit
        for tick in ticks
    )

    total = len(
        ticks
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
# DOWNLOAD AND SAVE
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
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "PRINCE PAUL FX "
            "Deriv historical "
            "tick downloader"
        )
    )

    parser.add_argument(
        "--market",
        default=DEFAULT_MARKET,
        help=(
            "Deriv market symbol."
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "Number of historical "
            "ticks to download."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/historical_ticks.csv"
        ),
        help=(
            "Output CSV path."
        ),
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
        "VERSION 2.1"
    )

    print(
        "=" * 64
    )

    print(
        f"Market: {args.market}"
    )

    print(
        f"Requested ticks: {args.count}"
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
                "No historical ticks "
                "were received."
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
