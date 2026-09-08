"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
DIGIT SCANNER
VERSION 1.0
============================================================

Purpose:
    Convert an AI candidate into a controlled DigitMatch
    confirmation signal.

Core logic:

        AI CANDIDATE
              |
              v
        LOCK DIGIT
              |
              v
        WAIT FOR NEXT TICK
              |
        +-----+------+
        |            |
      MATCH        NO MATCH
        |            |
        v            v
    CONFIRMED      REJECT
        |            |
        |            v
        |       RESUME SCAN
        |
        v
    READY SIGNAL

Important:
    This module does NOT place trades.

    A confirmed match means:

        next_digit == locked_digit

    It does not mean that a profitable trade is guaranteed.

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ==========================================================
# STATES
# ==========================================================

class ScannerState(str, Enum):

    WAITING = "WAITING"

    CANDIDATE = "CANDIDATE"

    LOCKED = "LOCKED"

    CONFIRMED = "CONFIRMED"

    REJECTED = "REJECTED"


# ==========================================================
# SCAN RESULT
# ==========================================================

@dataclass
class ScanResult:

    state: str

    candidate_digit: Optional[int]

    locked_digit: Optional[int]

    observed_digit: Optional[int]

    probability: float

    confirmed: bool

    reason: str


# ==========================================================
# DIGITMATCH SCANNER
# ==========================================================

class DigitMatchScanner:
    """
    Controls the candidate -> lock -> next-tick confirmation
    lifecycle.
    """

    def __init__(
        self,
        minimum_probability: float = 0.0,
        require_model_agreement: bool = False,
    ):

        self.minimum_probability = float(
            minimum_probability
        )

        self.require_model_agreement = bool(
            require_model_agreement
        )

        self.state = ScannerState.WAITING

        self.candidate_digit: Optional[int] = None

        self.locked_digit: Optional[int] = None

        self.candidate_probability = 0.0

        self.model_agreement = False

        self.candidate_tick_index: Optional[int] = None

        self.confirmation_tick_index: Optional[int] = None

        self.tick_index = 0

        self.last_observed_digit: Optional[int] = None

        self.confirmation_count = 0

        self.rejection_count = 0


    # ======================================================
    # VALIDATE DIGIT
    # ======================================================

    @staticmethod
    def validate_digit(
        digit: int,
    ) -> int:
        """
        Validate a digit in the range 0-9.
        """

        try:

            digit = int(digit)

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "Digit must be an integer."
            )

        if not 0 <= digit <= 9:

            raise ValueError(
                "Digit must be between 0 and 9."
            )

        return digit


    # ======================================================
    # START CANDIDATE
    # ======================================================

    def set_candidate(
        self,
        digit: int,
        probability: float = 0.0,
        model_agreement: bool = False,
    ) -> ScanResult:
        """
        Register a new AI candidate.

        The candidate is NOT considered confirmed.

        The next tick must be observed before confirmation.
        """

        digit = self.validate_digit(
            digit
        )

        probability = float(
            probability
        )

        if probability < 0.0:

            probability = 0.0

        if probability > 1.0:

            probability = 1.0

        if (
            probability
            < self.minimum_probability
        ):

            self.reset()

            return ScanResult(
                state=ScannerState.REJECTED.value,
                candidate_digit=None,
                locked_digit=None,
                observed_digit=None,
                probability=probability,
                confirmed=False,
                reason=(
                    "Candidate probability is below "
                    "the configured minimum."
                ),
            )

        if (
            self.require_model_agreement
            and not model_agreement
        ):

            self.reset()

            return ScanResult(
                state=ScannerState.REJECTED.value,
                candidate_digit=None,
                locked_digit=None,
                observed_digit=None,
                probability=probability,
                confirmed=False,
                reason=(
                    "Candidate rejected because "
                    "LightGBM and XGBoost do not agree."
                ),
            )

        self.candidate_digit = digit

        self.locked_digit = digit

        self.candidate_probability = (
            probability
        )

        self.model_agreement = bool(
            model_agreement
        )

        self.candidate_tick_index = (
            self.tick_index
        )

        self.confirmation_tick_index = None

        self.state = ScannerState.LOCKED

        return ScanResult(
            state=ScannerState.LOCKED.value,
            candidate_digit=self.candidate_digit,
            locked_digit=self.locked_digit,
            observed_digit=None,
            probability=self.candidate_probability,
            confirmed=False,
            reason=(
                "Candidate locked. "
                "Waiting for the next tick."
            ),
        )


    # ======================================================
    # PROCESS NEXT TICK
    # ======================================================

    def process_tick(
        self,
        digit: int,
    ) -> ScanResult:
        """
        Process one incoming tick.

        If a digit is locked:

            next_digit == locked_digit
                -> CONFIRMED

            next_digit != locked_digit
                -> REJECTED

        If no candidate is locked, the tick is simply
        observed and the scanner remains in WAITING state.
        """

        digit = self.validate_digit(
            digit
        )

        self.tick_index += 1

        self.last_observed_digit = digit

        # --------------------------------------------------
        # No active candidate
        # --------------------------------------------------

        if self.locked_digit is None:

            self.state = (
                ScannerState.WAITING
            )

            return ScanResult(
                state=ScannerState.WAITING.value,
                candidate_digit=None,
                locked_digit=None,
                observed_digit=digit,
                probability=0.0,
                confirmed=False,
                reason=(
                    "No digit is locked. "
                    "Continue scanning."
                ),
            )

        # --------------------------------------------------
        # Candidate was created on the current tick.
        #
        # Never confirm against the same tick.
        # --------------------------------------------------

        if (
            self.candidate_tick_index
            == self.tick_index
        ):

            self.state = (
                ScannerState.LOCKED
            )

            return ScanResult(
                state=ScannerState.LOCKED.value,
                candidate_digit=(
                    self.candidate_digit
                ),
                locked_digit=(
                    self.locked_digit
                ),
                observed_digit=digit,
                probability=(
                    self.candidate_probability
                ),
                confirmed=False,
                reason=(
                    "Candidate locked on this tick. "
                    "Waiting for the following tick."
                ),
            )

        # --------------------------------------------------
        # MATCH
        # --------------------------------------------------

        if digit == self.locked_digit:

            self.state = (
                ScannerState.CONFIRMED
            )

            self.confirmation_tick_index = (
                self.tick_index
            )

            self.confirmation_count += 1

            return ScanResult(
                state=ScannerState.CONFIRMED.value,
                candidate_digit=(
                    self.candidate_digit
                ),
                locked_digit=(
                    self.locked_digit
                ),
                observed_digit=digit,
                probability=(
                    self.candidate_probability
                ),
                confirmed=True,
                reason=(
                    "NEXT TICK MATCHED THE "
                    "LOCKED DIGIT."
                ),
            )

        # --------------------------------------------------
        # NO MATCH
        # --------------------------------------------------

        self.rejection_count += 1

        rejected_digit = (
            self.locked_digit
        )

        self.state = (
            ScannerState.REJECTED
        )

        result = ScanResult(
            state=ScannerState.REJECTED.value,
            candidate_digit=(
                self.candidate_digit
            ),
            locked_digit=(
                self.locked_digit
            ),
            observed_digit=digit,
            probability=(
                self.candidate_probability
            ),
            confirmed=False,
            reason=(
                f"Locked digit "
                f"{rejected_digit} "
                f"did not match next digit "
                f"{digit}. Candidate rejected."
            ),
        )

        # --------------------------------------------------
        # Automatically return to scanning mode.
        # --------------------------------------------------

        self._clear_lock()

        return result


    # ======================================================
    # ACKNOWLEDGE CONFIRMATION
    # ======================================================

    def acknowledge_confirmation(
        self,
    ) -> ScanResult:
        """
        Clear a confirmed candidate after the consuming
        system has processed the confirmation.

        This keeps the scanner separate from execution.
        """

        if (
            self.state
            != ScannerState.CONFIRMED
        ):

            return ScanResult(
                state=self.state.value,
                candidate_digit=(
                    self.candidate_digit
                ),
                locked_digit=(
                    self.locked_digit
                ),
                observed_digit=(
                    self.last_observed_digit
                ),
                probability=(
                    self.candidate_probability
                ),
                confirmed=False,
                reason=(
                    "There is no pending confirmation."
                ),
            )

        confirmed_digit = (
            self.locked_digit
        )

        self._clear_lock()

        self.state = (
            ScannerState.WAITING
        )

        return ScanResult(
            state=ScannerState.WAITING.value,
            candidate_digit=None,
            locked_digit=None,
            observed_digit=(
                self.last_observed_digit
            ),
            probability=0.0,
            confirmed=False,
            reason=(
                f"Confirmation for digit "
                f"{confirmed_digit} acknowledged. "
                "Scanner is ready for another candidate."
            ),
        )


    # ======================================================
    # CLEAR LOCK
    # ======================================================

    def _clear_lock(
        self,
    ) -> None:

        self.candidate_digit = None

        self.locked_digit = None

        self.candidate_probability = 0.0

        self.model_agreement = False

        self.candidate_tick_index = None

        self.confirmation_tick_index = None


    # ======================================================
    # RESET
    # ======================================================

    def reset(
        self,
    ) -> None:
        """
        Completely reset the scanner.
        """

        self._clear_lock()

        self.state = (
            ScannerState.WAITING
        )


    # ======================================================
    # READY TO SCAN
    # ======================================================

    @property
    def ready_for_candidate(
        self,
    ) -> bool:

        return (
            self.locked_digit is None
        )


    # ======================================================
    # CONFIRMATION PENDING
    # ======================================================

    @property
    def confirmation_pending(
        self,
    ) -> bool:

        return (
            self.state
            == ScannerState.CONFIRMED
        )


    # ======================================================
    # STATUS
    # ======================================================

    def status(
        self,
    ) -> dict:

        return {
            "state": self.state.value,
            "candidate_digit": (
                self.candidate_digit
            ),
            "locked_digit": (
                self.locked_digit
            ),
            "candidate_probability": (
                self.candidate_probability
            ),
            "model_agreement": (
                self.model_agreement
            ),
            "tick_index": (
                self.tick_index
            ),
            "last_observed_digit": (
                self.last_observed_digit
            ),
            "candidate_tick_index": (
                self.candidate_tick_index
            ),
            "confirmation_tick_index": (
                self.confirmation_tick_index
            ),
            "confirmation_count": (
                self.confirmation_count
            ),
            "rejection_count": (
                self.rejection_count
            ),
            "ready_for_candidate": (
                self.ready_for_candidate
            ),
        }


# ==========================================================
# STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "================================================"
    )

    print(
        "PRINCE PAUL FX"
    )

    print(
        "DIGITMATCH SCANNER TEST"
    )

    print(
        "================================================"
    )

    scanner = DigitMatchScanner()

    # ------------------------------------------------------
    # Example:
    #
    # AI selects digit 7.
    # The scanner locks 7.
    # The NEXT tick is then tested.
    # ------------------------------------------------------

    print(
        "\n1. AI selects digit 7"
    )

    result = scanner.set_candidate(
        digit=7,
        probability=0.16,
        model_agreement=True,
    )

    print(result)

    # ------------------------------------------------------
    # Next tick = 3
    # Therefore 7 does not match.
    # Candidate is rejected.
    # ------------------------------------------------------

    print(
        "\n2. Next tick = 3"
    )

    result = scanner.process_tick(
        3
    )

    print(result)

    # ------------------------------------------------------
    # Scanner is now waiting again.
    # ------------------------------------------------------

    print(
        "\n3. Scanner status"
    )

    print(
        scanner.status()
    )

    # ------------------------------------------------------
    # New candidate = 4
    # ------------------------------------------------------

    print(
        "\n4. AI selects digit 4"
    )

    result = scanner.set_candidate(
        digit=4,
        probability=0.15,
        model_agreement=True,
    )

    print(result)

    # ------------------------------------------------------
    # Next tick = 4
    # Match confirmed.
    # ------------------------------------------------------

    print(
        "\n5. Next tick = 4"
    )

    result = scanner.process_tick(
        4
    )

    print(result)

    # ------------------------------------------------------
    # Acknowledge confirmation.
    # ------------------------------------------------------

    print(
        "\n6. Acknowledge confirmation"
    )

    result = scanner.acknowledge_confirmation()

    print(result)

    print(
        "\n================================================"
    )

    print(
        "SCANNER TEST COMPLETE"
    )

    print(
        "================================================"
    )
