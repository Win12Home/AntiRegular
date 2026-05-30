# Environment Variable and Const Variables
from rich import print
import os

__all__ = [  # I worried that
    "ANTI_REGULAR__LORA_RANK",
    "ANTI_REGULAR__VERBOSE",
]

# NOTE: The prefix of Environment Variables is ANTI_REGULAR__

"""
ANTI_REGULAR__LORA_RANK
Usage:
  Value: 0, 2, 4, 8, ... (Any unsigned int)
  Explanation:
    This option decide AntiRegular use what rank to QLoRa
    If you set this value to 0 or empty, AntiRegular will find the best lora
"""

ANTI_REGULAR__LORA_RANK = str(os.environ.get("ANTI_REGULAR__LORA_RANK", ""))

if ANTI_REGULAR__LORA_RANK.strip() == "":  # Check empty
    ANTI_REGULAR__LORA_RANK = 0  # From empty to 0

try:
    ANTI_REGULAR__LORA_RANK = abs(
        int(ANTI_REGULAR__LORA_RANK)
    )  # Oh I said I need a unsigned integer.
except:  # NOQA, Except anything.
    print(
        "[yellow]WARNING: [/yellow]ANTI_REGULAR__LORA_RANK Variable might not an integer! Automatic set it to Auto Decide Mode!"
    )
    ANTI_REGULAR__LORA_RANK = 0  # Set it to zero.

"""
ANTI_REGULAR__VERBOSE
Usage:
  Value: 0, 1 (Boolean)
  Explanation:
    This option decide AntiRegular run with Verbose Mode or Normal Mode
"""

ANTI_REGULAR__VERBOSE = str(os.environ.get("ANTI_REGULAR__VERBOSE", ""))

if ANTI_REGULAR__VERBOSE.strip() == "":  # Check empty
    ANTI_REGULAR__VERBOSE = "false"  # Oh, empty -> false

if ANTI_REGULAR__VERBOSE.strip().lower() == "true":
    ANTI_REGULAR__VERBOSE = 1
elif ANTI_REGULAR__VERBOSE.strip().lower() == "false":
    ANTI_REGULAR__VERBOSE = 0

try:
    ANTI_REGULAR__VERBOSE = int(ANTI_REGULAR__VERBOSE)  # Boolean? Integer?

    if (
        ANTI_REGULAR__VERBOSE < 0 or ANTI_REGULAR__VERBOSE > 1
    ):  # Convert Integer to Boolean
        print(
            "[yellow]WARNING: [/yellow]ANTI_REGULAR__VERBOSE Variable might not a boolean! Automatic set it to Verbose Mode!"
        )
        ANTI_REGULAR__VERBOSE = 1
except:  # NOQA, also
    print(
        "[yellow]WARNING: [/yellow]ANTI_REGULAR__VERBOSE Variable might not a boolean! Automatic set it to Normal Mode!"
    )
    ANTI_REGULAR__VERBOSE = 0  # False!!!!
