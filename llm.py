
def run_llm(inp):

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("\n  transformers is not installed. Run:")
        print("    pip install transformers accelerate\n")
        raise SystemExit(1)


    # ======================================================================
    # CONFIG
    # ======================================================================

    MODEL_NAME = "gpt2"           # 124M params — downloads ~500MB once
    MAX_LEN    = 256               # max tokens per training example
    EPOCHS     = 3                 # few epochs is enough for a real model
    BATCH_SIZE = 2                 # small batch, fits any GPU
    LR         = 5e-5              # standard fine-tuning learning rate
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


    # ======================================================================
    # DATASET — 60 Q&A pairs for fine-tuning
    # ======================================================================

    df = pd.read_csv("dataset.csv")
    df.head()

    QA_DATA = list(zip(df['questions'], df['answers']))

    # ======================================================================
    # FORMAT DATA FOR INSTRUCTION TUNING
    # ======================================================================

    def format_conversation(question, answer):
        return f"User: {question}\nAssistant: {answer}{tokenizer.eos_token}"

    # ======================================================================
    # DATASET WITH LOSS MASKING
    # ======================================================================

    class FineTuneDataset(Dataset):
    
        def __init__(self, pairs, tokenizer, max_len=MAX_LEN):
            self.items = []

            for question, answer in pairs:
                # Encode the prompt part (everything up to and including "Assistant:")
                prompt_text = f"User: {question}\nAssistant:"
                full_text = f"{prompt_text} {answer}{tokenizer.eos_token}"

                prompt_ids = tokenizer.encode(prompt_text)
                full_ids = tokenizer.encode(full_text)

                # Truncate to max_len
                full_ids = full_ids[:max_len]
                prompt_len = min(len(prompt_ids), len(full_ids))

                # Labels: -100 on prompt tokens, real ids on response tokens
                labels = [-100] * prompt_len + full_ids[prompt_len:]
                labels = labels[:max_len]

                # Pad to max_len
                pad_len = max_len - len(full_ids)
                input_ids = full_ids + [tokenizer.eos_token_id] * pad_len
                labels = labels + [-100] * pad_len

                attention_mask = [1] * len(full_ids) + [0] * pad_len

                self.items.append({
                    'input_ids': torch.tensor(input_ids),
                    'attention_mask': torch.tensor(attention_mask),
                    'labels': torch.tensor(labels),
                })

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    # ======================================================================
    # TRAINING
    # ======================================================================

    def train(model, dataset, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        model.train()

        print(f"\n  Training for {epochs} epochs on {DEVICE} ...\n")

        for epoch in range(1, epochs + 1):
            total_loss = 0
            for step, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg = total_loss / len(loader)
            print(f"    Epoch {epoch}/{epochs}  loss={avg:.4f}")

        print(f"\n  Done! Final loss: {avg:.4f}")


    # ======================================================================
    # GENERATION
    # ======================================================================

    @torch.no_grad()
    def generate_response(model,
                        tokenizer, 
                        question, 
                        max_new_tokens=150,
                        temperature=0.7):

        """Generate a response to a user question."""
        model.eval()

        prompt = f"User: {question}\nAssistant:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the new tokens (skip the prompt)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up: stop at "User:" if the model tries to continue the conversation
        for stop in ["User:", "\nUser", "\n\nUser"]:
            if stop in response:
                response = response[:response.index(stop)]

        return response.strip()


    # ======================================================================
    # MAIN
    # ======================================================================

    print("=" * 60)
    print("  HAPose")
    print("=" * 60)

    # --- 1. Load pretrained model ---
    print(f"\n1. Loading pretrained {MODEL_NAME} from Hugging Face ...")
    print(f"   (first run downloads ~500MB, then it's cached)\n")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {MODEL_NAME}")
    print(f"   Parameters: {n_params:,}")
    print(f"   Device: {DEVICE}")

    # --- 2. Test BEFORE fine-tuning ---
    print("\n2. Testing BEFORE fine-tuning:\n")
    for q in ["Hey", "Hey!"]:

        r = generate_response(model, tokenizer, q)
        print(f"   You:  {q}")
        print(f"   Bot:  {r[:200]}")
        print()

    print("   (Notice: the base model doesn't follow our Q&A format)\n")

    # --- 3. Prepare dataset ---
    print("3. Preparing fine-tuning dataset ...")
    dataset = FineTuneDataset(QA_DATA, tokenizer, max_len=MAX_LEN)
    print(f"   {len(dataset)} training pairs  |  max_len={MAX_LEN}")

    sample = dataset[0]
    masked = (sample['labels'] == -100).sum().item()
    total = sample['labels'].shape[0]
    print(f"   Loss masking: {masked}/{total} tokens masked in sample")

    # --- 4. Fine-tune ---
    print("\n4. Fine-tuning ...")
    train(model, dataset, epochs=EPOCHS)

    # --- 5. Test AFTER fine-tuning ---
    print("\n5. Testing AFTER fine-tuning:\n")
    test_questions = [
        "Hey",
        "Hei",
        "Hello"]

    for q in test_questions:
        r = generate_response(model, tokenizer, q, temperature=0.5)
        print(f"   You:  {q}")
        print(f"   Bot:  {r}")
        print()

    # --- 6. Interactive chat --- #AI 16.04 Claude 13.32
    print(f"n6. Responding to input: '{inp}'\n")
    response = generate_response(model, tokenizer, inp, temperature=0.7)
    print(f" Bot: {response}\n")

    return response