# I'm lazy to create wheel by myself, geez.
from argparse import ArgumentParser
from rich_argparse import RichHelpFormatter
from utils.variables import ANTI_REGULAR__LORA_RANK
import sys

# Ohh shit, I don't wanna write comment

parser = ArgumentParser(
    prog="AntiRegular",
    usage=f"{sys.argv[0]} -m PATH -tm PATH -o PATH ...",
    description="Make a model from Regular to Uncensored quickly.",
    formatter_class=RichHelpFormatter,
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    dest="model_path",
    required=True,
    help="Path to Model",
    metavar="PATH/NAME",
)
parser.add_argument(
    "-tm",
    "--teacher-model",
    type=str,
    dest="teacher_model_path",
    required=True,
    help="Path to Teacher Model (Only GGUF)",
    metavar="PATH",
)
parser.add_argument(
    "-o",
    "--outdir",
    type=str,
    dest="outdir",
    required=True,
    help="Path to output directory",
    metavar="PATH",
)
parser.add_argument(
    "--lora-rank",
    type=int,
    default=ANTI_REGULAR__LORA_RANK,
    dest="lora_rank",
    required=False,
    help="Decide AntiRegular use what rank to QLoRa (Default 0)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--auto-find-rank-start",
    type=int,
    default=64,
    dest="auto_find_rank_start",
    required=False,
    help="Decide AntiRegular use what rank to iter (Default 64)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--lora-alpha",
    type=int,
    default=None,
    dest="lora_alpha",
    required=False,
    help="Decide AntiRegular use what alpha to QLoRa (Default rank*2)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--dpo-epoch",
    type=int,
    default=15,
    required=False,
    help="Epoches of DPO (Default 15)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--lora-epoch",
    type=int,
    default=5,
    dest="lora_epoch",
    required=False,
    help="Epoches of LoRa Fine-Tuning (Default 5)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--lora-restore-epoch",
    type=int,
    default=3,
    dest="lora_restore_epoch",
    required=False,
    help="Epoches of The Last Fine-tuned to normal (Default 3)",
    metavar="+INTEGER",
)
parser.add_argument(
    "-tc",
    "--teacher-ctx-size",
    type=int,
    default=16384,
    dest="teacher_ctx_size",
    required=False,
    help="Teacher ctx size to check valid (Default 8192)",
    metavar="+INTEGER",
)
parser.add_argument(
    "-ct",
    "--ctx-type",
    type=str,
    default="q8_0",
    choices=["f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"],
    dest="ctx_type",
    required=False,
    help="Quantize CTX Type (Default q8_0)",
    metavar="+QUANT",
)
parser.add_argument(
    "-t",
    "--thread",
    type=int,
    default=0,
    dest="thread",
    required=False,
    help="Number of threads for llama (Default 0)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--harmless-prompts",
    type=str,
    default="./resources/data/harmless_prompts.json",
    dest="harmless_prompts",
    required=False,
    help="Path to harmless prompts file",
    metavar="+PATH",
)
parser.add_argument(
    "--harmful-prompts",
    type=str,
    default="./resources/data/harmful_prompts.json",
    dest="harmful_prompts",
    required=False,
    help="Path to harmful prompts file",
    metavar="+PATH",
)
parser.add_argument(
    "--select-prompts",
    type=int,
    default=0,
    dest="select_prompts",
    required=False,
    help="Select What Prompts to Select (Default 0)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--select-prompts-to-check-refusals",
    type=int,
    default=20,
    dest="select_prompts_to_check",
    required=False,
    help="Select The First What Prompts from Harmful Prompts to check Refusals (Default 20)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    dest="max_seq_length",
    required=False,
    help="Maximum Sequence Length (Default 2048)",
    metavar="+INTEGER",
)
parser.add_argument(
    "--load-with-16bit",
    dest="load_with_16bit",
    required=False,
    help="Load Student Model with 16bit",
    action="store_true",
)
parser.add_argument(
    "--no-restore",
    dest="no_restore",
    required=False,
    help="Do not restore Student Model with original output for harmless data",
    action="store_true",
)
parser.add_argument(
    "-cref-sft",
    "--check-refusals-while-fine-tuned",
    dest="check_refusals_while_fine_tuned",
    required=False,
    help="Check Refusals While Fine Tuned (Default false)",
    action="store_true",
)
arguments = parser.parse_args()
