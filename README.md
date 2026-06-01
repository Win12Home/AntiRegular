# AntiRegular
## Introduction
A Project to jailbreak LLMs via **JBOWFT** (Jailbreak Only With Fine-Tuning).<br>
## How to use it
Ensure you have a python environment and version $\ge$ 3.13.<br>
Then, step by step.<br>
```bash
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130  # Or /whl/cpu
export ANTI_REGULAR__VERBOSE=1  # $env:ANTI_REGULAR__VERBOSE=1 in Powershell, set ANTI_REGULAR__VERBOSE=1 in CMD
python main.py -m /path/to/your/instruct/model -tm /path/to/your/uncensored/model.gguf -o /path/to/lora/directory
```
Then wait when jailbreak finished.
## Research Paper
viXra link: Waiting...<br>
Paper: See `paper.pdf`