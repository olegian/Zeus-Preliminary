from train import train_model, get_tokens
from cse599o_basics.tokenizer import BPETokenizer
import pathlib
import json
import os
import itertools

LRS = [0.0001, 0.0005, 0.001]
N_EPOCHS = 2000
CHECKPOINT_FREQ = 250
DEVICE = "cuda"
BATCH_SIZE = 32
BETA_1 = 0.9
BETA_2 = 0.999
W = 0.1
D_MODEL = 512
D_FF = 3072
D_S = [(256, 1344), (512, 3072)]
R_THETA = 10_000
N_LAYERS = [4, 12]
N_HEADS = [16, 32]
CONTEXT_LENGTH = 256
EPS = 1e-8

TRAINING_DATASET = "./data/TinyStoriesV2-GPT4-train.txt"
VALIDATION_DATASET = "./data/TinyStoriesV2-GPT4-valid.txt"

def main():
    tokenizer = BPETokenizer(vocab={}, merges={})
    train_tokens = get_tokens(pathlib.Path(TRAINING_DATASET), tokenizer)
    val_tokens = get_tokens(pathlib.Path(VALIDATION_DATASET), tokenizer)

    for i, (lr, ds, n_layers, n_heads) in enumerate(itertools.product(LRS, D_S, N_LAYERS, N_HEADS)):
        if i < 8:
            continue

        checkpoint_path = pathlib.Path(os.getcwd()) / "experiments" / f"lr-{lr}-d-{ds[0]}-nl-{n_layers}-nh-{n_heads}"
        md = train_model(
            train_tokens=train_tokens, 
            val_tokens=val_tokens, 
            tokenizer=tokenizer,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            model_params={
                 "d_model": ds[0],
                 "num_heads": n_heads,
                 "d_ff": ds[1],
                 "theta": R_THETA,
                 "context_length": CONTEXT_LENGTH,
                 "num_layers": n_layers,
            },
            optimizer_params={
                'lr': lr,
                'b1': BETA_1,
                'b2': BETA_2,
                'w': W,
            },
            checkpoint_path=checkpoint_path,
            checkpoint_freq=CHECKPOINT_FREQ,
            checkpoint_cutoff=4000,
            device=DEVICE,
            capture_metadata=True,
            eps=EPS,
        )

        md_path = checkpoint_path / f"run_data.json"
        with open(md_path, "w") as f:
            json.dump(md, f, indent=4)
    

if __name__ == "__main__":
    main()
