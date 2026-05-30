from llama_cpp import Llama
from transformers import LlamaForCausalLM
from unsloth import FastLanguageModel
from core.quant_to_llama_num import quant_to_llama_num
from peft import PeftModelForCausalLM
from utils.variables import ANTI_REGULAR__VERBOSE
import torch


def get_llama_model(
    model_path: str,
    ctx_size: int,
    kv_quant: str,
    gpu_layers: int,
) -> Llama:
    return Llama(
        model_path=model_path,
        ctx_size=ctx_size,
        n_gpu_layers=gpu_layers,
        flash_attn=True,  # Open KV Quantize
        type_k=quant_to_llama_num(kv_quant),
        type_v=quant_to_llama_num(kv_quant),
        verbose=ANTI_REGULAR__VERBOSE,  # Decide by Environment Variable
    )


def get_train_model(
    model_path: str,
    load_in_16bit: bool,
    lora_rank: int,
    lora_alpha: float = None,
    lora_dropout: float = 0.0,
    target_modules: list = None,
    max_seq_length: int = 2048,
) -> tuple:
    # Check target modules
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

    # Check Alpha
    if lora_alpha is None:
        lora_alpha = lora_rank * 4

    # Load Original Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        load_in_16bit=load_in_16bit,
        load_in_4bit=not load_in_16bit,
        max_seq_length=max_seq_length,
        trust_remote_code=True,  # Wow! New model?
        dtype=torch.bfloat16,  # Prefer use ↓
    )

    # Load LoRa Adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",  # More VRAM, give in to me~
        random_state=67,  # Six Seven~
        bias="none",
        max_seq_length=max_seq_length,
    )

    return (model, tokenizer)


def get_normal_model(
    model_path: str,
    load_in_16bit: bool,
) -> FastLanguageModel:
    return FastLanguageModel.from_pretrained(
        model_path,
        load_in_16bit=load_in_16bit,
        load_in_4bit=not load_in_16bit,
        trust_remote_code=True,
        dtype=torch.bfloat16,  # Hey bro I can't believe you use the bnb-4bit or FP32 model
        max_seq_length=2048,
    )


def get_peft_model(
    model: LlamaForCausalLM,
    lora_rank: int,
    target_modules: list = None,
    max_seq_length: int = 2048,
) -> PeftModelForCausalLM:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_rank * 4,  # Wow that's so much!
        max_seq_length=max_seq_length,
        use_gradient_checkpointing="unsloth",
        random_state=67,  # All we know, 67~
        bias="none",
    )
