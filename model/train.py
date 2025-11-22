import os
import pathlib
from typing import Optional, Tuple
import numpy as np
import tiktoken
import torch
import tqdm
import wandb
import sys
import argparse
import timeit
import json

from cse599o_basics.adamw import AdamWOptimizer
from cse599o_basics.batch import data_loader
from cse599o_basics.checkpoint import save_checkpoint
from cse599o_basics.cross_entropy import cross_entropy_loss
from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.gradient_clipping import clip_gradient
from cse599o_basics.tokenizer import BPETokenizer
from cse599o_basics.lr_schedule import cosine_lr_schedule
from tests.common import FIXTURES_PATH

# ====================================
# Run this script with
# uv run python train.py "../data/TinyStoriesV2-GPT4-train.txt" "../data/TinyStoriesV2-GPT4-valid.txt" -l 0.001 -b 16 -b1 0.9 -b2 0.999 -w 0.1 -e 10 -o "../checkpoints" -of 2 -cl 256 -dm 64 -dff 171 -r 10000 -nl 4 -nh 16 -d "cpu"
# ====================================

parser = argparse.ArgumentParser(description="Script for training the assignment 1 model")
# required parameters
parser.add_argument('trainingdataset', type=str, help="path to the training dataset")
parser.add_argument('validationdataset', type=str, help="path to the validation dataset")
parser.add_argument('-l', '--lr', type=float, help="learning rate to use", required=True)
parser.add_argument('-b', '--batchsize', type=int, help="batch size", required=True)
parser.add_argument('-b1', '--beta1', type=float, help="beta 1 value for AdamW", required=True)
parser.add_argument('-b2', '--beta2', type=float, help="beta 2 value for AdamW", required=True)
parser.add_argument('-w', '--weightdecay', type=float, help="weight decay value for AdamW", required=True)
parser.add_argument('-e', '--epochs', type=int, help="number of epochs to train for", required=True)

# optional specifications
parser.add_argument('--eps', type=float, help="epsilon value", default=1e-5)
parser.add_argument('-o', '--checkpoint', type=str, help="path to folder which contains checkpoints", default=None)
parser.add_argument('-of', '--checkpointfreq', type=int, help="how many epochs before a checkpoint is made", default=10)
parser.add_argument('-oc', '--autocheckpointcutoff', type=int, help="how many epochs until every best validation loss model is saved", default=0)
parser.add_argument('-cl', '--contextlength', type=int, help="max context length", default=256)
parser.add_argument('-dm', '--dmodel', type=int, help="number of feature dimensions", default=512)
parser.add_argument('-dff', '--dff', type=int, help="number of dimensions in the feed forward network", default=1344)
parser.add_argument('-r', '--ropetheta', type=int, help="theta for RoPE", default=10_000)
parser.add_argument('-nl', '--numlayers', type=int, help="number of transformer layers", default=4)
parser.add_argument('-nh', '--numheads', type=int, help="number of heads to use for self-attention", default=16)
parser.add_argument('-d', '--device', type=str, help="device to use for training", default='cpu')

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
EOT_TOKEN = "<|endoftext|>"

def main(args: argparse.Namespace):
    tokenizer = BPETokenizer(vocab={}, merges={})
    train_tokens = get_tokens(pathlib.Path(args.trainingdataset), tokenizer)
    val_tokens = get_tokens(pathlib.Path(args.validationdataset), tokenizer)
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint is not None else None
    md = train_model(
        train_tokens=train_tokens, 
        val_tokens=val_tokens, 
        tokenizer=tokenizer,
        epochs=args.epochs,
        batch_size=args.batchsize,
        model_params={
             "d_model": args.dmodel,
             "num_heads": args.numheads,
             "d_ff": args.dff,
             "theta": args.ropetheta,
             "context_length": args.contextlength,
             "num_layers": args.numlayers,
        },
        optimizer_params={
            'lr': args.lr,
            'b1': args.beta1,
            'b2': args.beta2,
            'w': args.weightdecay,
        },
        checkpoint_path=checkpoint_path,
        checkpoint_freq=args.checkpointfreq,
        checkpoint_cutoff=args.autocheckpointcutoff,
        device=args.device,
        capture_metadata=True,
    )

    md_path = checkpoint_path / "run_data.json"
    with open(md_path, "w") as f:
        json.dump(md, f, indent=4)
    

def get_tokens(dataset_path: pathlib.Path, tokenizer: BPETokenizer) -> np.memmap:
    if not os.path.exists(dataset_path):
        raise Exception(f"While attempting to open dataset: dataset '{dataset_path.resolve()}' does not exist")

    cached_dataset_path = dataset_path.with_suffix(".tokens.npy")
    if not os.path.exists(cached_dataset_path):
        # need to create tokenized numpy file, count tokens in dataset
        # and generate the appropriate mmap np file
        print(f"|> Generating numpy mmap of tokens from {dataset_path}")

        total_tokens = 0
        with open(dataset_path, mode='r', encoding='utf-8') as f:
            print("|> Precomputing size...")
            for line in tqdm.tqdm(f):
                total_tokens += len(tokenizer.encode(line))
        
        mmap = np.memmap(cached_dataset_path, dtype=np.int32, mode='write', shape=(total_tokens,))

        # write to mmap file, each line tokenized seperately and then written
        current_pos = 0
        with open(dataset_path, mode='r', encoding='utf-8') as f:
            print(f"|> Tokenizing file into {total_tokens:,} tokens...")
            for line in tqdm.tqdm(f):
                token_ids = tokenizer.encode(line)
                len_token_ids = len(token_ids)
                # TODO: pretty sure this check is completely optional given the precomptation
                if current_pos + len_token_ids <= total_tokens:
                    mmap[current_pos : (current_pos+len_token_ids)] = token_ids
                    current_pos += len_token_ids

    else :
        print(f"|> Found numpy mmap of tokens from {cached_dataset_path}")
        mmap = np.memmap(cached_dataset_path)

    return mmap


def train_model(
    train_tokens: np.memmap,
    val_tokens: np.memmap,
    tokenizer: BPETokenizer,
    epochs: int,
    batch_size: int,
    model_params: dict,
    optimizer_params: dict,
    checkpoint_freq: int,
    checkpoint_path: Optional[pathlib.Path] = None,
    checkpoint_cutoff: int = 0,
    max_grad: float = 6,
    eps: float = 1e-8,
    device:Optional[torch.device] = "cpu",
    capture_metadata: Optional[bool] = False,
):
    metadata = dict() if capture_metadata else None

    if metadata is not None:
        metadata['val_losses'] = []
        metadata['train_losses'] = []
        metadata['n_epochs'] = epochs
        metadata['optimizer_params'] = optimizer_params
        metadata['optimizer_params']['eps'] = eps
        metadata['model_params'] = model_params
        metadata['batch_size'] = batch_size

    if checkpoint_path is None:
        print("! Warning: checkpointing is disabled")
    else: 
        if not os.path.exists(checkpoint_path):
            checkpoint_path.mkdir(parents=True)

        print(f"! Checkpoints are going to be saved every {checkpoint_freq} epochs, to {checkpoint_path.resolve()}")
        print(f"! Auto checkpoints are enabled after {checkpoint_cutoff} epochs")

    print(f"! Using device: {device}")
    print(f"! Using batch_size: {batch_size}")
    print(f"! Using model_params: {model_params}")
    print(f"! Using optimizer_params: {optimizer_params}")

    # run = wandb.init(
    #     # Set the wandb entity where your project will be logged (generally your team name).
    #     entity="olegian-university-of-washington",
    #     # Set the wandb project where this run will be logged.
    #     project="cse599o-assignment-1",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": optimizer_params['lr'],
    #         "beta1": optimizer_params['b1'],
    #         "beta2": optimizer_params['b2'],
    #         "w": optimizer_params['w'],
    #         "epochs": epochs,
    #     },
    # )

    model = TransformerLM(
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        d_ff=model_params['d_ff'],
        theta=model_params['theta'],
        max_seq_len=model_params['context_length'],
        vocab_size=tokenizer.tokenizer.n_vocab,
        context_length=model_params['context_length'],
        num_layers=model_params['num_layers'],
        device=device,
    ).to(device)
    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=optimizer_params['lr'],
        betas=(optimizer_params['b1'], optimizer_params['b2']),
        weight_decay=optimizer_params['w'],
        eps=eps,
    )
    
    best_val_loss = float("inf")
    for epoch in range(epochs):
        st = timeit.default_timer()

        print(f"===== Training Epoch {epoch+1}/{epochs} =====")
        model.train()
        train_inputs, train_labels = data_loader(
            train_tokens,
            batch_size=batch_size,
            context_length=model_params['context_length'],
            device=device
        )

        # update lr of optimizer based on schedule
       # lr = cosine_lr_schedule(epoch, optimizer_params['lr'], 1e-4, int(0.03 * epochs), int(epochs / (2.5)))
       # print(f"!!!\t Changing LR to: {lr}")
       # for group in optimizer.param_groups:
       #     group['lr'] = lr 

       # for inputs, labels in zip(batched_inputs, batched_labels):
       #     inputs, labels = inputs.to(device), labels.to(device)

       #     optimizer.zero_grad()
       #     outputs = model.forward(batched_inputs)
       #     loss = cross_entropy_loss(outputs, batched_labels)
       #     loss.backward()
       #     optimizer.step()

       #     training_loss = loss.item() * batched_labels.numel()
       #     total_tokens += batched_labels.numel()

        outputs = model(train_inputs)
        loss = cross_entropy_loss(outputs, train_labels)
        loss.backward()
        # clip_gradient(model.parameters(), max_grad)
        optimizer.step()
        train_loss = loss.item()
        
        print(f">>>\t Finished epoch {epoch + 1}, training loss: {train_loss:.2f}")
        # run.log({"avg-training-loss": avg_training_loss})

        model.eval()
        val_inputs, val_labels = data_loader(
            val_tokens,
            batch_size=batch_size,
            context_length=model_params['context_length'],
            device=device
        )

        val_loss = 0.0
        with torch.no_grad():
            outputs = model(val_inputs)
            val_loss = cross_entropy_loss(outputs, val_labels).item()


        print(f">>>\t Val loss: {val_loss:.4f}")
        if checkpoint_path is not None and val_loss < best_val_loss and checkpoint_cutoff - 1 < epoch:
            print("!!!\t Found better validation loss...")
            best_val_loss = val_loss

            if metadata is not None:
                metadata['best_epoch_idx'] = epoch

            checkpoint_file_path = checkpoint_path / f"best-model"
            save_checkpoint(model, optimizer, epoch, checkpoint_file_path)

            print(">>> Saving checkpoint...")
            checkpoint_file_path = checkpoint_path / f"epoch-{epoch + 1}-train-{train_loss:.2f}-val-{val_loss:.2f}"
            save_checkpoint(model, optimizer, epoch, checkpoint_file_path)

        if metadata is not None:
            metadata['train_losses'].append(train_loss)
            metadata['val_losses'].append(val_loss)

        et = timeit.default_timer()
        print(f"---\t Ellapsed time in epoch: {et-st}s")


    print("   ===  DONE ===   ")
    return metadata

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
