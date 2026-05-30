from llama_cpp import (
    LLAMA_FTYPE_MOSTLY_Q8_0,
    LLAMA_FTYPE_MOSTLY_F16,
    LLAMA_FTYPE_ALL_F32,
    LLAMA_FTYPE_MOSTLY_Q4_0,
    LLAMA_FTYPE_MOSTLY_BF16,
    LLAMA_FTYPE_MOSTLY_Q4_1,
    LLAMA_FTYPE_MOSTLY_Q5_0,
    LLAMA_FTYPE_MOSTLY_Q5_1,
)


def quant_to_llama_num(quant: str) -> int:
    match quant:
        case "f32":
            return LLAMA_FTYPE_ALL_F32
        case "f16":
            return LLAMA_FTYPE_MOSTLY_F16
        case "bf16":
            return LLAMA_FTYPE_MOSTLY_BF16
        case "q8_0":
            return LLAMA_FTYPE_MOSTLY_Q8_0
        case "q4_0":
            return LLAMA_FTYPE_MOSTLY_Q4_0
        case "q4_1":
            return LLAMA_FTYPE_MOSTLY_Q4_1
        case "q5_0":
            return LLAMA_FTYPE_MOSTLY_Q5_0
        case "q5_1":
            return LLAMA_FTYPE_MOSTLY_Q5_1

    raise ValueError(f"Unknown quant: {quant}")
