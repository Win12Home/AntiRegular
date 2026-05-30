from unsloth import is_bf16_supported, PatchDPOTrainer
from utils.variables import ANTI_REGULAR__VERBOSE
from transformers import AutoProcessor
from peft import PeftModelForCausalLM
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
from datasets import Dataset

try:
    PatchDPOTrainer()  # I love patching!
except:
    pass


def dpo_train(
    model: PeftModelForCausalLM,
    dataset: list[
        tuple[str, str, str]
    ],  # [("user from harmful", "chosen", "rejected")]
    epoch: int,
    batch_size: int = 1,
) -> PeftModelForCausalLM:

    if len(dataset) == 0:
        return model

    if not isinstance(dataset, list):
        return model

    # Convert!!!!

    dataset = Dataset.from_dict(
        {
            "prompt": [i[0] for i in dataset],
            "chosen": [i[1] for i in dataset],
            "rejected": [i[2] for i in dataset],
        }
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,  # No way i no no no ref models
        train_dataset=dataset,
        processing_class=AutoProcessor.from_pretrained(model.base_model.name_or_path),
        args=DPOConfig(
            output_dir="./temporary",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epoch,
            optim="adamw_8bit",
            seed=67,  # Six Seven~
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1 if ANTI_REGULAR__VERBOSE else 5,
            learning_rate=5e-5,
            gradient_accumulation_steps=4,
            beta=0.05,
        ),
    )

    trainer.train()

    return model


def lora_train(
    model: PeftModelForCausalLM,
    dataset: list[
        tuple[str, str]
    ],  # [("user from harmless", "assistant from original model")]
    epoch: int,
    batch_size: int = 1,
) -> PeftModelForCausalLM:

    if len(dataset) == 0:
        return model

    if not isinstance(dataset, list):
        return model

    def conversations_to_sft_text(example) -> dict[str, str]:
        user_msg = [
            example["conversations"][i * 2]["content"]
            for i in range(int(len(example["conversations"]) / 2))
        ]
        assistant_msg = [
            example["conversations"][i * 2 + 1]["content"]
            for i in range(int(len(example["conversations"]) / 2))
        ]

        text = ""
        for i in range(int(len(example["conversations"]) / 2)):
            text += f"<|user|>\n{user_msg[i]}\n<|assistant|>\n{assistant_msg[i]}\n"

        return {"text": text}

    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            for user, assistant in dataset  # From list[tuple[str, str]] to list[dict[str, list[dict[str, str]]]]
        ]
    )

    dataset = dataset.map(conversations_to_sft_text)

    trainer = SFTTrainer(
        model=model,  # NOQA
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="./temporary",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epoch,
            optim="adamw_8bit",  # NOQA, adamw is adamw
            seed=67,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1 if ANTI_REGULAR__VERBOSE else 5,
            learning_rate=2e-4,
            gradient_accumulation_steps=4,
        ),
    )

    trainer.train()

    return model
