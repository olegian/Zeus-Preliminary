from __future__ import print_function

import argparse
import os
import functools

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset
from torch.utils.data.dataset import Dataset
import torch
import nvtx
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import deepspeed
from deepspeed.pipe import PipelineModule
from zeus.monitor import ZeusMonitor
from zeus.utils.lr_scaler import LinearScaler
from zeus.device import get_gpus

N_COMM_REPEATS = 1

# WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))

# TO RUN: srun --gpus-per-node=2 uv run deepspeed test_qwen_ds_pp_dispatch.py

# TO RUN: srun --gpus-per-node=2 uv run nsys profile -o qwen_ds_pp deepspeed test_qwen_ds_pp_dispatch.py

# def wrap_all():
#     functions = [
#         dist.recv,
#         dist.irecv,
#         dist.send]
#     for func in functions:
#         func_name = func.__name__
#         orig = func
#         def wrapped(*args, **kwargs):
#             print(f"[Rank {dist.get_rank()}] WRAPPED {func_name}")
#             ret = orig(*args, **kwargs)
#             torch.cuda.synchronize()
#             return ret
#         func = wrapped
# orig = dist.recv
# def wrapped_recv(*args, **kwargs):
#     print(f"[Rank {dist.get_rank()}] WRAPPED recv")
#     ret = orig(*args, **kwargs)
#     torch.cuda.synchronize()
#     return ret
# dist.recv = wrapped_recv

# orig_ = dist.send
# def wrapped_send(*args, **kwargs):
#     print(f"[Rank {dist.get_rank()}] WRAPPED send")
#     ret = orig_(*args, **kwargs)
#     torch.cuda.synchronize()
#     return ret
# dist.send = wrapped_send

def train(args, model: deepspeed.PipelineEngine, ds, optimizer: optim.Optimizer, epoch, batch_size, monitor: ZeusMonitor):
    model.train()
    energy_measurements = []
    n = 5
    print("warmup")

    for _ in range(n):
        monitor.begin_window("training")

        loss = model.train_batch()
        if model.is_last_stage():
            print(f"loss: {loss}")

        train_mes = monitor.end_window("training")
        energy_measurements.append(train_mes.total_energy)

    measurements = {}
    def wrapped_closure(func):
        def wrapped(*args, **kwargs):
            print(f"[Rank {dist.get_rank()}] WRAPPED {func.__name__}")
            # cloned_args = []
            # for arg in args:
            #     if isinstance(arg, torch.Tensor):
            #         cloned_args.append(arg.clone().detach())
            #     else:
            #         cloned_args.append(arg)

            monitor.begin_window(f"{func.__name__}")
            for i in range(N_COMM_REPEATS):
                ret = func(*args, **kwargs)
                # if hasattr(func, 'wait'):
                #     ret.wait() # force s
            # ret = func(*args, **kwargs)
            torch.cuda.synchronize()
            res = monitor.end_window(f"{func.__name__}")
            if func.__name__ not in measurements:
                measurements[func.__name__] = []
            measurements[func.__name__].append(res.total_energy)
            return ret
        return wrapped

    functions = [dist.recv, dist.send]

    for func in functions:
        wrapped_ = wrapped_closure(func)
        setattr(dist, func.__name__, wrapped_)

    print("measuring...")
    with nvtx.annotate("measured", color='blue'):
        energy_measurements = []
        
        for i in range(n):
            with nvtx.annotate(f"energy-{i}", color='red'):
                monitor.begin_window("training")

                loss = model.train_batch()
                if model.is_last_stage():
                    print(f"loss: {loss}")

                train_mes = monitor.end_window("training")
                energy_measurements.append(train_mes.total_energy)
        
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"{local_rank}e{epoch}|> Avg energy consumption per step: {np.array(energy_measurements).mean(-1)}")
    # print(f"[Rank {local_rank}] FINAL MEASUREMENTS: {len(measurements)} calls, total energy: {sum(measurements)}J")
    for key in measurements:
        vals = measurements[key]
        print(f"[Rank {local_rank}] FINAL MEASUREMENTS for {key}: {len(vals)} calls, total energy: {sum(vals)}")

def is_distributed():
    return dist.is_available() and dist.is_initialized()

class ShittyWrapper(nn.Module):
    def __init__(self, decoder_layer, rotary_emb):
        super().__init__()
        self.layer = decoder_layer
        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

        return self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=self.rotary_emb(hidden_states, position_ids),
            **kwargs,
        )

def dummy_loss_fn(outputs,labels):
        # outputs is a CausalLMOutputWithPast for some reason
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        return nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dir",
        default="logs",
        metavar="L",
        help="directory where summary logs are stored",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.5,
        help="Target accuracy (default: 0.5)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local rank for distributed training",
    )
    if dist.is_available():
        parser.add_argument(
            "--backend",
            type=str,
            help="Distributed backend",
            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
            default=dist.Backend.GLOO,
        )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    local_rank = int(getattr(args, "local_rank", os.environ.get("LOCAL_RANK", "0")))

    deepspeed.init_distributed()

    ########################## Zeus ##########################
    monitor = ZeusMonitor(gpu_indices=[local_rank])

    # Fetch the batch size from the BSO server. (Oleg: There used to be more
    # zeus stuff here which used this BSO server to optimize the batch size,
    # i've removed it as it seemed irrelevant).
    batch_size = 4
    print("Chosen batch_size:", batch_size)
    # Scale the learning rate accordingly.
    # Default was batch size 64 and learing rate 0.01, and we use the linear
    # scaling rule since the optimizer is SGD.
    args.lr = LinearScaler(bs=64, lr=0.01).compute_lr(new_bs=batch_size)
    ##########################################################

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("loaded model and tokenizer")

    layers = []
    target = model.model
    rotary_emb = target.rotary_emb
    layers.append(target.embed_tokens)
    for layer in target.layers:
        layers.append(ShittyWrapper(layer, rotary_emb))
    layers.append(target.norm)
    layers.append(model.lm_head)

    world_size = dist.get_world_size()

    model = PipelineModule(
        layers=layers,
        num_stages=world_size if world_size > 0 else 1,
        loss_fn=dummy_loss_fn,
        partition_method="parameters",
    )

    print(f"PP = {world_size if world_size > 0 else 1} stages")

    # need simple text dataset instead of image dataset for language model

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:5%]")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    # text, input_ids, attention_mask

    assert isinstance(train_dataset, HFDataset)
    assert isinstance(test_dataset, HFDataset)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": args.lr,
                "momentum": args.momentum
            }
        },
        "fp16": {
            "enabled": False
        }
    }

    class ProperDataset(Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = input_ids.clone()
            return (
                input_ids, labels
            )

    model, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=ProperDataset(train_dataset),
        config=ds_config
    )

    ########################## Zeus ##########################
    for epoch in range(1, args.epochs + 1):
        # ds = train_dataset.shuffle(seed=args.seed + epoch)
        train(args, model, train_loader, optimizer, epoch, batch_size, monitor)
    ##########################################################

    if args.save_model and local_rank == 0:
        torch.save(model.state_dict(), "qwen_ds_2.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
