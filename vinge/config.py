import os

VINGE_DIR = os.path.join(os.environ["HOME"], ".vinge")

VINGE_JOBS = os.path.join(VINGE_DIR, "jobs.json")

MISTRAL_WEIGHTS_URI = "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q5_K_M.gguf?download=true"

MISTRAL_WEIGHTS_FILE = "mistral.gguf"

MISTRAL_DIM = 4096

OPENAI_MODEL = "gpt-4o-mini"

OPENAI_MAX_TOKENS = 10000
