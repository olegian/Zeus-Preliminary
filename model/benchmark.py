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

parser = argparse.ArgumentParser(description="Script for training the assignment 1 model")
# required parameters
parser.add_argument('trainingdataset', type=str, help="path to the training dataset")
parser.add_argument('validationdataset', type=str, help="path to the validation dataset")
parser.add_argument('-l', '--lr', type=float, help="learning rate to use", required=True)
parser.add_argument('-b', '--batchsize', type=int, help="batch size", required=True)
parser.add_argument('-b1', '--beta1', type=float, help="beta 1 value for AdamW", required=True)
parser.add_argument('-b2', '--beta2', type=float, help="beta 2 value for AdamW", required=True)
parser.add_argument('-w', '--weightdecay', type=float, help="weight decay value for AdamW", required=True)
parser.add_argument('-nw', '--nwarmup', type=int, help="number of steps before timing starts", required=True)
parser.add_argument('-nt', '--ntimed', type=int, help="number of steps timed", required=True)

# optional specifications
parser.add_argument('-tf', '--timeforwardonly', action='store_true', help="true to time only forward")
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

def main(args: argparse.Namespace):
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint is not None else None
    md = benchmark(
        n_warmup=args.nwarmup,
        n_timed=args.ntimed,
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
        device=args.device,
        only_forward=args.timeforwardonly,
    )
    

def benchmark(
    n_warmup:int,
    n_timed: int,
    batch_size: int,
    model_params: dict,
    optimizer_params: dict,
    eps: float = 1e-8,
    device:Optional[torch.device] = "cpu",
    only_forward: Optional[bool] = True,
):
    print(f"=== Benchmark, timing {n_timed} after {n_warmup} steps ===")
    print(f"! Using model_params: {model_params}")
    print(f"! Using optimizer_params: {optimizer_params}")
    print(f"! Using batch_size: {batch_size}")

    tokenizer = BPETokenizer(vocab={}, merges={})
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

    model.train()
    
    n_vocab = tokenizer.tokenizer.n_vocab
    inputs = torch.randint(0, n_vocab, (batch_size, model_params['context_length']), device=device)
    target = torch.randint(0, n_vocab, (batch_size, model_params['context_length']), device=device)

    for _ in range(n_warmup):
        optimizer.zero_grad(set_to_none=True)
        out = model(inputs)
        loss = cross_entropy_loss(out, target)
        loss.backward()
        optimizer.step()

    torch.cuda.memory._record_memory_history(max_entries=1000000)
    print("|||||   START MONITORED RUNS  |||||")
    for _ in range(n_timed):
        # optimizer.zero_grad(set_to_none=True)
        out = model(inputs)
        # loss = cross_entropy_loss(out, target)
        # loss.backward()
        # optimizer.step()

    torch.cuda.memory._dump_snapshot(f"mem_snap_fwd_{model_params['context_length']}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
