from transformers import LlamaForCausalLM, AutoTokenizer
from core.generate import generate_from_CasualLM
from utils.logger import Logger
from core.get_model import get_peft_model, get_normal_model
from core.train import dpo_train, lora_train
from core.refuse import check_refused
from utils.variables import ANTI_REGULAR__VERBOSE
from datetime import datetime
from llama_cpp import Llama
import torch
import gc


def find_best_rank(
    model: LlamaForCausalLM,
    teacher: Llama,
    harmful_prompts: list[str],
    max_seq_length: int,
    start_rank: int,
) -> tuple[int, LlamaForCausalLM]:
    Logger.info("Finding the best rank")
    now_rank = start_rank

    target_prompt = harmful_prompts[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model.name_or_path,
    )

    Logger.debug("Target prompt:" + target_prompt)

    Logger.info("Generating Student Response...")
    rejected = generate_from_CasualLM(
        model,
        AutoTokenizer.from_pretrained(
            model.name_or_path,
        ),
        target_prompt,
        16384,
        True,
        True,
    )

    Logger.info("Generating Teacher Response...")
    chosen = ""

    before_datetime = datetime.now()

    for chunk in teacher(
        f"<|im_start|>user\n{target_prompt}<|im_end|>\n<|im_start|>assistant\n",
        temperature=0.6,
        presence_penalty=1.0,
        repeat_penalty=1.1,
        max_tokens=teacher.n_ctx(),
        stream=True,
    ):
        if "choices" in chunk:
            chosen += chunk["choices"][0]["text"]

            if ANTI_REGULAR__VERBOSE:
                print(chunk["choices"][0]["text"], end="", flush=True)

    print()

    Logger.debug(
        f"Information: {len(chosen)/(datetime.now()-before_datetime).total_seconds():.2f} char/s (Average), Used {(datetime.now()-before_datetime).total_seconds():.2f} seconds."
    )

    path = model.name_or_path

    del model

    while True:
        Logger.info(f"Testing rank={now_rank}...")

        torch.cuda.empty_cache()

        model, _ = get_normal_model(
            path,
            False,
        )

        model = get_peft_model(  # Get Peft Model
            model,
            now_rank,
            max_seq_length=max_seq_length,
        )

        model = dpo_train(model, [(target_prompt, chosen, rejected)], 15)

        model = lora_train(model, [(target_prompt, chosen)], 5)

        if not check_refused(teacher, model, target_prompt, 2048):
            break

        del model
        torch.cuda.empty_cache()
        gc.collect()

        now_rank *= 2

    del model
    torch.cuda.empty_cache()
    gc.collect()

    model = get_normal_model(
        path,
        False,
    )

    return (now_rank, model)
