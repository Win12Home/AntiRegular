from llama_cpp import Llama
from utils.logger import Logger
import torch
import gc


def remove_llama(llama: Llama) -> None:
    if isinstance(llama, Llama):
        Logger.debug(f"Removing a Llama Object, ID: {id(llama)}")
        llama.close()
        del llama
        torch.cuda.empty_cache()
        gc.collect()
        Logger.debug("Removed this Llama Object")
