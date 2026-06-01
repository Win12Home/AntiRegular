from transformers import (
    PreTrainedTokenizerFast,
    LlamaForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from threading import Thread
from utils.variables import ANTI_REGULAR__VERBOSE
from utils.logger import Logger
from datetime import datetime
import torch


def generate_from_CasualLM(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int,
    only_debug_streaming_output: bool,
    enable_thinking: bool = False,
) -> str:
    Logger.debug("Tokenizing Message...")

    input_msg = tokenizer(
        (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False
            )
            + "assistant\n"
            if enable_thinking
            else "assistant\n<think>\n</think>\n"
        ),
        return_tensors="pt",
    ).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Tokenize

    Logger.debug("Tokenized Message!")
    Logger.debug("Generating Streamer...")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    Logger.debug("Generated Streamer!")
    Logger.debug("Generating Message...")

    with torch.inference_mode():
        thread = Thread(
            target=lambda: model.generate(
                **input_msg,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                do_sample=True,
                generation_config=GenerationConfig(
                    enable_thinking=enable_thinking,
                    repetition_penalty=1.1,
                    top_p=0.95,
                    top_k=40,
                ),
            ),
            daemon=True,
        )
        thread.start()

        before_datetime = datetime.now()
        texts = ""
        for chunk in streamer:
            texts += chunk
            if not only_debug_streaming_output or ANTI_REGULAR__VERBOSE:
                print(chunk, end="", flush=True)

        Logger.debug(
            f"Information: {len(texts)/(datetime.now()-before_datetime).total_seconds():.2f} char/s (Average), Used {(datetime.now()-before_datetime).total_seconds():.2f} seconds."
        )

        thread.join()  # Wait for thread

    return texts
