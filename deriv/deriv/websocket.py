"""
============================================================
PRINCE PAUL FX
DERIV LIVE TICK WEBSOCKET
VERSION 1.0
============================================================

Purpose:
    Connect to Deriv's live market-data WebSocket,
    subscribe to a selected market, receive ticks,
    extract the last digit and maintain a rolling
    50-tick buffer.

This module is DATA ONLY.

It does NOT:
    - place trades
    - contain the ML model
    - contain Streamlit UI
    - store API credentials
    - execute recovery strategies

Those components will be added separately.

============================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
import websockets


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("PrincePaulFX.Deriv")


# ============================================================
# DERIV ENDPOINTS
# ============================================================

REST_BASE_URL = "https://api.derivws.com"

PUBLIC_WEBSOCKET_URL = (
    "wss://api.derivws.com/"
    "trading/v1/options/ws/public"
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_MARKET = "1HZ100V"

DEFAULT_BUFFER_SIZE = 50

RECONNECT_DELAY = 5

MAX_RECONNECT_ATTEMPTS = 20

REQUEST_TIMEOUT = 15


# ============================================================
# TICK DATA OBJECT
# ============================================================

@dataclass
class Tick:
    """
    Normalized Deriv tick.
    """

    symbol: str

    quote: float

    epoch: int

    last_digit: int

    raw: dict


# ============================================================
# DERIV CLIENT
# ============================================================

class DerivWebSocketClient:
    """
    Deriv live tick client.

    Public mode:
        Used for live market data.
        No API token required.

    Authenticated mode:
        Uses the Options API OTP endpoint to obtain
        the authenticated WebSocket URL.

    The authenticated mode will be useful later for
    account/trading functionality.
    """

    def __init__(
        self,
        market: str = DEFAULT_MARKET,
        app_id: Optional[str] = None,
        bearer_token: Optional[str] = None,
        account_id: Optional[str] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):

        self.market = market

        self.app_id = app_id

        self.bearer_token = bearer_token

        self.account_id = account_id

        self.buffer_size = buffer_size

        self.tick_buffer: deque[Tick] = deque(
            maxlen=buffer_size
        )

        self.websocket = None

        self.connected = False

        self.authenticated = False

        self.stop_requested = False

        self.reconnect_attempts = 0

    # ========================================================
    # DIGIT BUFFER
    # ========================================================

    @property
    def digits(self) -> list[int]:

        return [
            tick.last_digit
            for tick in self.tick_buffer
        ]

    # ========================================================
    # TICK BUFFER
    # ========================================================

    @property
    def ticks(self) -> list[Tick]:

        return list(self.tick_buffer)

    # ========================================================
    # LATEST TICK
    # ========================================================

    @property
    def latest_tick(self) -> Optional[Tick]:

        if not self.tick_buffer:

            return None

        return self.tick_buffer[-1]

    # ========================================================
    # EXTRACT LAST DIGIT
    # ========================================================

    @staticmethod
    def extract_last_digit(quote) -> int:
        """
        Extract the final decimal digit from the quote.

        Example:

            123.4567 -> 7
            123.4500 -> 0
            99.1     -> 1

        The quote is converted to a string so that we do
        not introduce floating-point rounding into the
        digit extraction process.
        """

        quote_string = str(quote).strip()

        if not quote_string:

            raise ValueError(
                "Empty quote received."
            )

        # Handle scientific notation.
        if "e" in quote_string.lower():

            quote_string = format(
                float(quote),
                "f"
            )

        # We want the final digit of the displayed quote.
        digits = [
            character
            for character in quote_string
            if character.isdigit()
        ]

        if not digits:

            raise ValueError(
                f"Unable to extract last digit "
                f"from quote: {quote}"
            )

        return int(digits[-1])

    # ========================================================
    # NORMALIZE DERIV MESSAGE
    # ========================================================

    def normalize_tick(
        self,
        message: dict,
    ) -> Optional[Tick]:
        """
        Convert a Deriv WebSocket tick message into
        our internal Tick object.
        """

        if message.get("msg_type") != "tick":

            return None

        tick_data = message.get("tick")

        if not isinstance(
            tick_data,
            dict,
        ):

            return None

        symbol = tick_data.get("symbol")

        quote = tick_data.get("quote")

        epoch = tick_data.get("epoch")

        if symbol is None:

            return None

        if quote is None:

            return None

        if epoch is None:

            return None

        try:

            quote_float = float(quote)

            last_digit = (
                self.extract_last_digit(
                    quote
                )
            )

            return Tick(
                symbol=str(symbol),

                quote=quote_float,

                epoch=int(epoch),

                last_digit=last_digit,

                raw=message,
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            logger.warning(
                "Tick normalization failed: %s",
                error,
            )

            return None

    # ========================================================
    # ADD TICK
    # ========================================================

    def add_tick(
        self,
        tick: Tick,
    ) -> None:

        self.tick_buffer.append(
            tick
        )

    # ========================================================
    # REQUEST AUTHENTICATED OTP URL
    # ========================================================

    async def request_otp_url(self) -> str:
        """
        Request the authenticated Options API
        WebSocket URL.

        This is NOT used for public tick streaming.

        The credentials are supplied at runtime and are
        never hard-coded into this module.
        """

        if not self.account_id:

            raise ValueError(
                "account_id is required "
                "for authenticated mode."
            )

        if not self.bearer_token:

            raise ValueError(
                "bearer_token is required "
                "for authenticated mode."
            )

        endpoint = (
            f"{REST_BASE_URL}"
            f"/trading/v1/options/accounts/"
            f"{self.account_id}/otp"
        )

        headers = {
            "Authorization":
                f"Bearer {self.bearer_token}",
        }

        if self.app_id:

            headers[
                "Deriv-App-ID"
            ] = self.app_id

        timeout = aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT
        )

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:

            async with session.post(
                endpoint,
                headers=headers,
            ) as response:

                response_text = (
                    await response.text()
                )

                if response.status >= 400:

                    raise RuntimeError(
                        "OTP request failed "
                        f"with HTTP "
                        f"{response.status}: "
                        f"{response_text}"
                    )

                try:

                    payload = json.loads(
                        response_text
                    )

                except json.JSONDecodeError:

                    raise RuntimeError(
                        "Invalid JSON received "
                        "from OTP endpoint."
                    )

        data = payload.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):

            raise RuntimeError(
                "OTP response does not "
                "contain a valid data object."
            )

        websocket_url = data.get(
            "url"
        )

        if not websocket_url:

            raise RuntimeError(
                "OTP response does not "
                "contain a WebSocket URL."
            )

        return str(
            websocket_url
        )

    # ========================================================
    # CONNECT
    # ========================================================

    async def connect(
        self,
        authenticated: bool = False,
    ) -> None:
        """
        Establish the WebSocket connection.

        authenticated=False:
            Public market data.

        authenticated=True:
            Obtain an authenticated WebSocket URL
            through the Options API OTP endpoint.
        """

        self.stop_requested = False

        if authenticated:

            websocket_url = (
                await self.request_otp_url()
            )

        else:

            websocket_url = (
                PUBLIC_WEBSOCKET_URL
            )

        logger.info(
            "Connecting to Deriv..."
        )

        self.websocket = (
            await websockets.connect(
                websocket_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            )
        )

        self.connected = True

        self.authenticated = (
            authenticated
        )

        self.reconnect_attempts = 0

        logger.info(
            "Connected to Deriv."
        )

        await self.subscribe_ticks()

    # ========================================================
    # SUBSCRIBE TO TICKS
    # ========================================================

    async def subscribe_ticks(self) -> None:

        if not self.websocket:

            raise RuntimeError(
                "WebSocket is not connected."
            )

        request = {
            "ticks": self.market,
            "subscribe": 1,
        }

        await self.websocket.send(
            json.dumps(request)
        )

        logger.info(
            "Tick subscription sent: %s",
            self.market,
        )

    # ========================================================
    # RECEIVE MESSAGE
    # ========================================================

    async def receive_message(
        self,
    ) -> Optional[dict]:

        if not self.websocket:

            return None

        raw = await self.websocket.recv()

        if raw is None:

            return None

        if isinstance(
            raw,
            bytes,
        ):

            raw = raw.decode(
                "utf-8"
            )

        try:

            return json.loads(
                raw
            )

        except json.JSONDecodeError:

            logger.warning(
                "Received invalid JSON."
            )

            return None

    # ========================================================
    # STREAM TICKS
    # ========================================================

    async def stream(
        self,
        authenticated: bool = False,
    ) -> AsyncIterator[Tick]:
        """
        Continuously yield normalized live ticks.

        Automatic reconnect is enabled.
        """

        while not self.stop_requested:

            try:

                await self.connect(
                    authenticated=authenticated
                )

                while (
                    self.connected
                    and not self.stop_requested
                ):

                    message = (
                        await self.receive_message()
                    )

                    if message is None:

                        continue

                    tick = (
                        self.normalize_tick(
                            message
                        )
                    )

                    if tick is None:

                        continue

                    self.add_tick(
                        tick
                    )

                    yield tick

            except asyncio.CancelledError:

                logger.info(
                    "Tick stream cancelled."
                )

                break

            except Exception as error:

                self.connected = False

                logger.error(
                    "WebSocket error: %s",
                    error,
                )

                if self.stop_requested:

                    break

                self.reconnect_attempts += 1

                if (
                    self.reconnect_attempts
                    > MAX_RECONNECT_ATTEMPTS
                ):

                    logger.error(
                        "Maximum reconnect attempts reached."
                    )

                    break

                logger.info(
                    "Reconnecting in %s seconds "
                    "(attempt %s/%s)",
                    RECONNECT_DELAY,
                    self.reconnect_attempts,
                    MAX_RECONNECT_ATTEMPTS,
                )

                await asyncio.sleep(
                    RECONNECT_DELAY
                )

            finally:

                await self.close()

    # ========================================================
    # CLOSE
    # ========================================================

    async def close(self) -> None:

        self.connected = False

        self.authenticated = False

        if self.websocket:

            try:

                await self.websocket.close()

            except Exception as error:

                logger.debug(
                    "WebSocket close error: %s",
                    error,
                )

        self.websocket = None

    # ========================================================
    # STOP
    # ========================================================

    async def stop(self) -> None:

        self.stop_requested = True

        await self.close()


# ============================================================
# SIMPLE TERMINAL TEST
# ============================================================

async def test_live_ticks(
    market: str = DEFAULT_MARKET,
    number_of_ticks: int = 10,
) -> None:
    """
    Diagnostic test.

    This uses PUBLIC market data.

    No API token is required.
    """

    client = DerivWebSocketClient(
        market=market,
        buffer_size=50,
    )

    received = 0

    try:

        async for tick in client.stream(
            authenticated=False
        ):

            print(
                "----------------------------------------"
            )

            print(
                f"Market:     {tick.symbol}"
            )

            print(
                f"Quote:      {tick.quote}"
            )

            print(
                f"Last Digit: {tick.last_digit}"
            )

            print(
                f"Epoch:      {tick.epoch}"
            )

            received += 1

            if received >= number_of_ticks:

                break

    finally:

        await client.stop()


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    asyncio.run(
        test_live_ticks()
    )
