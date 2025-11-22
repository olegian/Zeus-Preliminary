from train import train_model, get_tokens
import tiktoken
import pathlib
import json
import os

# LRS = [10e-5, 10e-4, 10e-3]
LRS = [0.0005, 0.005]
N_EPOCHS = 2500
CHECKPOINT_FREQ = 250
DEVICE = "cuda"
BATCH_SIZE = 32
BETA_1 = 0.9
BETA_2 = 0.999
W = 0.1
D_MODEL = 512
D_FF = 1344
R_THETA = 10_000
N_LAYERS = 4
N_HEADS = 16
CONTEXT_LENGTH = 256

TRAINING_DATASET = "./data/TinyStoriesV2-GPT4-train.txt"
VALIDATION_DATASET = "./data/TinyStoriesV2-GPT4-valid.txt"

def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    train_tokens = get_tokens(pathlib.Path(TRAINING_DATASET), tokenizer)
    val_tokens = get_tokens(pathlib.Path(VALIDATION_DATASET), tokenizer)

    for lr in LRS:
        checkpoint_path = pathlib.Path(os.getcwd()) / "experiments" / f"lr-{lr}"
        md = train_model(
            train_tokens=train_tokens, 
            val_tokens=val_tokens, 
            tokenizer=tokenizer,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            model_params={
                 "d_model": D_MODEL,
                 "num_heads": N_HEADS,
                 "d_ff": D_FF,
                 "theta": R_THETA,
                 "context_length": CONTEXT_LENGTH,
                 "num_layers": N_LAYERS,
            },
            optimizer_params={
                'lr': lr,
                'b1': BETA_1,
                'b2': BETA_2,
                'w': W,
            },
            checkpoint_path=checkpoint_path,
            checkpoint_freq=CHECKPOINT_FREQ,
            checkpoint_cutoff=2000,
            device=DEVICE,
            capture_metadata=True,
        )

        md_path = checkpoint_path / f"{lr}-run_data.json"
        with open(md_path, "w") as f:
            json.dump(md, f, indent=4)
    

if __name__ == "__main__":
    main()
