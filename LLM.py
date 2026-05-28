import csv
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, logging as hf_logging
import re

hf_logging.set_verbosity_error()

MODEL_PATH = "./trained_model"
MODEL_NAME = "gpt2"
MAX_LEN    = 256
EPOCHS     = 3
BATCH_SIZE = 2
LR         = 5e-5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Globale variabler — modellen lastes bare én gang
_model     = None
_tokenizer = None


def debug(msg):
    print(f"[DEBUG] {msg}")

def _looks_like_valid_reply(text):
    text = text.strip()
    if len(text) < 8: 
        return False
    if re.fullmatch(r"[\d\.\s,\-\[\]]+", text): 
        return False
    letters = sum(c.isalpha() for c in text)
    digits  = sum(c.isdigit() for c in text)
    if letters < 6:
        return False
    if letters > 0 and digits > letters * 0.5:
        return False
    if not re.search(r"\b[A-Za-z]{3,}\b", text): 
        return False
    return True

# ======================================================================
# KALLBAR FUNKSJON — importer denne i user_interface.py
# ======================================================================

def call_llm(inpu, value=None):
    _ensure_model_loaded()
    for _ in range(2):
        response = generate_response(_model, _tokenizer, inpu, value=value)
        if _looks_like_valid_reply(response):
            return response
    return "(No reply available right now.)"

# ======================================================================
# INTERN: last og tren modellen (kjøres bare én gang)
# ======================================================================

def _ensure_model_loaded():
    global _model, _tokenizer
    if _model is not None:
        return

    debug("Loading tokenizer ...")
    _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    _tokenizer.pad_token = _tokenizer.eos_token

    if os.path.exists(MODEL_PATH):
        debug(f"Found saved model at {MODEL_PATH}, loading ...")
        _model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
    else:
        debug("No saved model found, training from scratch ...")
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
        QA_DATA = load_dataset()
        dataset = FineTuneDataset(QA_DATA, _tokenizer)
        train_model(_model, dataset)

    debug("Model ready!")


# ======================================================================
# DATASET
# ======================================================================

def load_dataset():
    debug("Loading dataset.csv ...")
    clean_lines = []
    with open("dataset.csv", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            in_quote, result = False, []
            for ch in line:
                if ch == '"': in_quote = not in_quote
                if ch == '#' and not in_quote: break
                result.append(ch)
            clean_lines.append("".join(result).strip().rstrip(','))

    QA_DATA = []
    header = True
    for row in csv.reader(clean_lines):
        if header:
            header = False
            continue
        if len(row) < 2:
            continue
        question = row[0].strip()
        answer   = row[1].strip()
        val      = row[2].strip() if len(row) >= 3 else ""
        if not question or not answer:
            continue
        prompt = f"{question} [value: {val}]" if val else question
        QA_DATA.append((prompt, answer))

    debug(f"Loaded {len(QA_DATA)} training pairs")
    return QA_DATA


# ======================================================================
# DATASET KLASSE
# ======================================================================

class FineTuneDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
        self.items = []
        for question, answer in pairs:
            prompt_text = f"User: {question}\nAssistant:"
            full_text   = f"{prompt_text} {answer}{tokenizer.eos_token}"
            prompt_ids  = tokenizer.encode(prompt_text)
            full_ids    = tokenizer.encode(full_text)[:max_len]
            prompt_len  = min(len(prompt_ids), len(full_ids))
            labels      = [-100] * prompt_len + full_ids[prompt_len:]
            labels      = labels[:max_len]
            pad_len     = max_len - len(full_ids)
            self.items.append({
                "input_ids":      torch.tensor(full_ids + [tokenizer.eos_token_id] * pad_len),
                "attention_mask": torch.tensor([1] * len(full_ids) + [0] * pad_len),
                "labels":         torch.tensor(labels + [-100] * pad_len),
            })

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ======================================================================
# TRENING
# ======================================================================

def train_model(model, dataset):
    debug("Starting training ...")
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch in loader:
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["labels"].to(DEVICE),
            )
            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += outputs.loss.item()
        avg = total_loss / len(loader)
        debug(f"Epoch {epoch}/{EPOCHS}  loss={avg:.4f}")
    debug("Training complete. Saving model ...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    _tokenizer.save_pretrained(MODEL_PATH)
    debug("Model saved!")


# ======================================================================
# GENERERING
# ======================================================================

@torch.no_grad()
def generate_response(model, tokenizer, question, value=None,
                      max_new_tokens=150, temperature=0.5):
    model.eval()

    if value is not None:
        prompt = (
            f"Context: the user's recent posture score is {value}.\n"
            f"User: {question}\n"
            f"Assistant:"
        )
    else:
        prompt = f"User: {question}\nAssistant:"

    encoded   = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = encoded["input_ids"]
    attn_mask = encoded["attention_mask"]

    output_ids = model.generate(
        input_ids,
        attention_mask = attn_mask,
        max_new_tokens = max_new_tokens,
        temperature    = temperature,
        do_sample      = True,
        top_p          = 0.85,
        pad_token_id   = tokenizer.eos_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    for stop in ["User:", "\nUser", "\n\nUser", "Context:"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()


# ======================================================================
# KJØR DIREKTE — bare for testing
# ======================================================================

if __name__ == "__main__":
    _ensure_model_loaded()
    print("\nTesting call_llm direkte:\n")
    print(call_llm("What is the current state of my posture?", value=3))
    print(call_llm("Make a summary of my postures current development", value=[4.2, 3.1, 3.8, 4.5]))
    print(call_llm("Do you have a recomedation for something i can do right now to improve my posture?"))