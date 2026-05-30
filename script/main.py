from datetime import datetime
from core.find_best_rank import find_best_rank
from core.generate import generate_from_CasualLM
from utils.rich_outputs import show_anti_regular_ascii
from utils.argument_parser import arguments
from utils.variables import ANTI_REGULAR__VERBOSE
from utils.logger import Logger
from core.get_model import get_normal_model, get_train_model
from core.train import lora_train, dpo_train
from llama_cpp import Llama
from core.quant_to_llama_num import quant_to_llama_num
from core.refuse import check_refused
from core.remove_llama import remove_llama
from transformers import AutoTokenizer
from json import loads, JSONDecodeError, dumps
from rich import print
from tqdm import tqdm
import torch
import os


def main() -> None:
    show_anti_regular_ascii()
    print()
    print("[red]Anti[/red][blue]Regular[/blue] 0.1.0")
    print("Win12Home (C) 2026, MIT License.")
    print()

    model, _ = get_normal_model(arguments.model_path, arguments.load_with_16bit)

    os.environ["GLOG_minloglevel"] = "2"
    teacher = Llama(
        arguments.teacher_model_path,
        n_gpu_layers=99,
        n_ctx=arguments.teacher_ctx_size,
        flash_attn=True,
        type_k=quant_to_llama_num(arguments.ctx_type),
        type_v=quant_to_llama_num(arguments.ctx_type),
        n_threads=arguments.thread,
        verbose=False,
    )

    Logger.info("Loading Harmful Prompts...")
    try:
        with open(arguments.harmful_prompts) as f:
            harmful_prompts = loads(f.read())
    except JSONDecodeError:
        Logger.error("Error loading Harmful Prompts: Not a valid JSON.")
        del model
        torch.cuda.empty_cache()
        remove_llama(teacher)
        os._exit(1)

    if not isinstance(harmful_prompts, list):
        Logger.error("Error loading Harmful Prompts: Not a list.")
        del model
        torch.cuda.empty_cache()
        remove_llama(teacher)
        os._exit(1)

    if not len(harmful_prompts):
        Logger.error("Error loading Harmful Prompts: JSON Empty")
        del model
        torch.cuda.empty_cache()
        remove_llama(teacher)
        os._exit(1)

    # Check refusals
    selected = min(arguments.select_prompts_to_check, len(harmful_prompts))
    Logger.info(
        f"Checking refusals... All {len(harmful_prompts)}, select {selected}/{len(harmful_prompts)} prompts to test"
    )

    refusals = 0

    for prompt in harmful_prompts[:selected]:
        if check_refused(teacher, model, prompt):
            refusals += 1

    try:
        Logger.info(
            f"Original Model Refusals: {refusals}/{selected}, rate: {refusals/selected*100:.2f}%"
        )

        if refusals / selected < 0.05:
            Logger.info("Refusals are low, no need to fine-tuning.")
            del model
            torch.cuda.empty_cache()
            remove_llama(teacher)
            os._exit(0)

    except ZeroDivisionError:
        Logger.info(f"Original Model Refusals: {refusals}/{selected}, rate: 0.00%")

    rank, model = (
        find_best_rank(
            model,
            teacher,
            harmful_prompts,
            arguments.max_seq_length,
            arguments.auto_find_rank_start,
        )
        if arguments.lora_rank <= 0
        else (arguments.lora_rank, model)
    )

    Logger.info(f"Using rank={rank} to training, removing all the model...")

    torch.cuda.empty_cache()
    remove_llama(teacher)

    Logger.info("Loading Harmless Prompts...")

    harmless_to_restore = []
    harmless_prompts = []
    original_harmless_responses = {}

    try:
        with open("harmless_responses.json", "r", encoding="utf-8") as f:
            original_harmless_responses = loads(f.read())
    except:
        pass

    try:
        with open(arguments.harmless_prompts, "r", encoding="utf-8") as f:
            harmless_prompts = loads(f.read())
    except JSONDecodeError:
        Logger.error("Error loading Harmless Prompts: Not a valid JSON.")

    if not isinstance(harmless_prompts, list):
        Logger.error("Error loading Harmless Prompts: Not a list.")
        harmless_prompts = []

    Logger.info(
        "Getting the responses from the original student model... (Harmless Prompts)"
    )

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    if not arguments.no_restore:
        for prompt in tqdm(
            harmless_prompts,
            desc="Getting responses...",
            unit="problems",
            unit_scale=True,
        ):
            Logger.debug(f"Getting the response of {prompt}...")

            if original_harmless_responses.get(prompt.strip(), None):
                harmless_to_restore.append(
                    (
                        prompt,
                        original_harmless_responses[prompt.strip()],
                    )
                )
                continue

            harmless_to_restore.append(
                (
                    prompt,
                    generate_from_CasualLM(model, tokenizer, prompt, 16384, True, True),
                )
            )

    Logger.info("Saving to cache...")
    with open("harmless_responses.json", "w", encoding="utf-8") as f:
        f.write(
            dumps(
                original_harmless_responses
                | {prompt.strip(): answer for prompt, answer in harmless_to_restore},
                indent=2,
                ensure_ascii=False,
            )
        )

    Logger.info("Removing Student Model...")
    del model
    torch.cuda.empty_cache()
    del tokenizer
    torch.cuda.empty_cache()

    Logger.info("Loading Teacher Model...")

    teacher = Llama(
        arguments.teacher_model_path,
        n_gpu_layers=99,
        n_ctx=arguments.teacher_ctx_size,
        flash_attn=True,
        type_k=quant_to_llama_num(arguments.ctx_type),
        type_v=quant_to_llama_num(arguments.ctx_type),
        n_threads=arguments.thread,
        verbose=False,
    )

    harmful_to_train = []

    original_harmful_responses = {}

    try:
        with open("harmful_responses.json", "r", encoding="utf-8") as f:
            original_harmful_responses = loads(f.read())
    except:
        pass

    for prompt in tqdm(
        harmful_prompts if arguments.select_prompts == 0 else harmful_prompts[:arguments.select_prompts-1], desc="Getting responses...", unit="problems", unit_scale=True
    ):
        texts = ""
        before_datetime = datetime.now()

        if original_harmful_responses.get(prompt.strip(), None):
            harmful_to_train.append(
                (prompt, original_harmful_responses[prompt.strip()])
            )
            continue

        for chunk in teacher(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            temperature=0.6,
            presence_penalty=1.0,
            repeat_penalty=1.1,
            max_tokens=teacher.n_ctx(),
            stream=True,
        ):
            if "choices" in chunk:
                texts += chunk["choices"][0]["text"]

                if ANTI_REGULAR__VERBOSE:
                    print(chunk["choices"][0]["text"], end="", flush=True)
        Logger.debug(
            f"Speed: {len(texts)/(datetime.now()-before_datetime).total_seconds():.2f} char/s. Total Time: {(datetime.now()-before_datetime).total_seconds():.2f}s"
        )
        harmful_to_train.append((prompt, texts))

    Logger.info("Saving to cache...")

    with open("harmful_responses.json", "w", encoding="utf-8") as f:
        f.write(
            dumps(
                original_harmful_responses
                | {prompt.strip(): answer for prompt, answer in harmful_to_train},
                indent=2,
                ensure_ascii=False,
            )
        )

    Logger.info("Removing Teacher Model...")
    remove_llama(teacher)

    Logger.info("Loading Student Model...")

    model, _ = get_normal_model(
        arguments.model_path,
        arguments.load_with_16bit,
    )

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)

    harmful_rejected_to_train = []

    original_harmful_rejected_responses = {}

    try:
        with open("harmful_rejected_responses.json", "r", encoding="utf-8") as f:
            original_harmful_rejected_responses = loads(f.read())
    except:
        pass

    for prompt in tqdm(
        harmful_prompts if arguments.select_prompts == 0 else harmful_prompts[:arguments.select_prompts-1], desc="Getting responses...", unit="problems", unit_scale=True
    ):
        texts = ""

        if original_harmful_rejected_responses.get(prompt.strip(), None):
            harmful_rejected_to_train.append(
                (prompt, original_harmful_rejected_responses[prompt.strip()])
            )
            continue

        texts = generate_from_CasualLM(
            model,
            tokenizer,
            prompt,
            16384,
            True,
            True,
        )

        harmful_rejected_to_train.append((prompt, texts))

    Logger.info("Saving to cache...")

    with open("harmful_rejected_responses.json", "w", encoding="utf-8") as f:
        f.write(
            dumps(
                original_harmful_rejected_responses
                | {
                    prompt.strip(): answer
                    for prompt, answer in harmful_rejected_to_train
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    Logger.info("Removing Student Model...")

    del model
    del tokenizer
    torch.cuda.empty_cache()

    Logger.info("Loading Peft Model...")

    peft_model, tokenizer = get_train_model(
        arguments.model_path,
        arguments.load_with_16bit,
        rank,
        max_seq_length=2048,
    )

    Logger.info("Training...")

    peft_model = dpo_train(
        peft_model,
        [
            (
                harmful_prompts[i],
                harmful_to_train[i][1],
                harmful_rejected_to_train[i][1],
            )
            for i in range(min(len(harmful_to_train), len(harmful_rejected_to_train)))
        ],  # For Safety, but the length is same.
        arguments.dpo_epoch,
    )

    peft_model = lora_train(
        peft_model,
        harmful_to_train,
        arguments.lora_epoch,
    )

    Logger.info("Restoring...")

    peft_model = lora_train(
        peft_model,
        harmless_to_restore,
        arguments.lora_restore_epoch,
    )

    from pathlib import Path

    Path(arguments.outdir).mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(arguments.outdir)
    tokenizer.save_pretrained(arguments.outdir)

    Logger.info(f"Saved Pretrained LoRa to {arguments.outdir}")

    if not arguments.check_refusals_while_fine_tuned:
        del peft_model
        torch.cuda.empty_cache()
        Logger.info("Exited")
        os._exit(0)

    Logger.info("Loading Llama Model...")

    teacher = Llama(
        arguments.teacher_model_path,
        n_gpu_layers=99,
        n_ctx=arguments.teacher_ctx_size,
        flash_attn=True,
        type_k=quant_to_llama_num(arguments.ctx_type),
        type_v=quant_to_llama_num(arguments.ctx_type),
        n_threads=arguments.thread,
        verbose=False,
    )

    Logger.info(
        f"Checking refusals... All {len(harmful_prompts)}, select {selected}/{len(harmful_prompts)} prompts to test"
    )

    after_refusals = 0

    for prompt in harmful_prompts[:selected]:
        if check_refused(teacher, peft_model, prompt):
            after_refusals += 1

    Logger.info(
        f"Refusals: {after_refusals}/{selected}, -{(refusals-after_refusals)/selected*100:.2f}% refusals"
    )

    del peft_model
    torch.cuda.empty_cache()
    remove_llama(teacher)
