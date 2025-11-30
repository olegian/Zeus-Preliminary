from __future__ import print_function

import argparse
import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.distributed.fsdp as fsdp
from torch.distributed.fsdp import FullyShardedDataParallel
from zeus.monitor import ZeusMonitor
from zeus.utils.lr_scaler import LinearScaler
import nvtx

# TO RUN: 
# srun --gpus-per-node=2 uv run torchrun --nproc_per_node=2 test_qwen.py

def train(args, model: FullyShardedDataParallel, device, ds, optimizer: optim.Optimizer, epoch, batch_size, monitor: ZeusMonitor):
    model.train()
    energy_measurements = []
    n = 0
    print("warmup")
    for start_idx in range(0, len(ds), batch_size):
        batch = ds[start_idx : start_idx + batch_size]
        data = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        monitor.begin_window("training")
        # output = model(data)
        output = model(input_ids=data, attention_mask=mask, labels=data)
        # loss = F.nll_loss(output, target)
        loss = output.loss
        loss.backward()
        # optimizer.step()

        train_mes = monitor.end_window("training")
        energy_measurements.append(train_mes.total_energy)

        if start_idx % (args.log_interval * batch_size) == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    start_idx,
                    len(ds),
                    100.0 * start_idx / len(ds),
                    loss.item(),
                )
            )
        
        if n == 5:
            break

        n += 1

    print("measuring...")
    with nvtx.annotate("measured", color='blue'):
        energy_measurements = []
        n = 0
        for start_idx in range(0, len(ds), batch_size):
            batch = ds[start_idx : start_idx + batch_size]
            data = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with model.no_sync():
                with nvtx.annotate(f"energy-{n}", color='red'):
                    monitor.begin_window("training")
                    # output = model(data)
                    output = model(input_ids=data, attention_mask=mask, labels=data)
                    # loss = F.nll_loss(output, target)
                    loss = output.loss
                    loss.backward()
                    # optimizer.step()

                    train_mes = monitor.end_window("training")
                    energy_measurements.append(train_mes.total_energy)

            if start_idx % (args.log_interval * batch_size) == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                        epoch,
                        start_idx,
                        len(ds),
                        100.0 * start_idx / len(ds),
                        loss.item(),
                    )
                )
            
            if n == 5:
                break

            n += 1
        
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"{local_rank}e{epoch}|> Avg energy consumption per step: {np.array(energy_measurements).mean(-1)}")

def is_distributed():
    return dist.is_available() and dist.is_initialized()

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
        default=10,
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

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    # if should_distribute():
    args.backend = dist.Backend.NCCL  # workaround
    print("Using distributed PyTorch with {} backend".format(args.backend))
    dist.init_process_group(backend=args.backend)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

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
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("loaded model and tokenizer")
    # model = FullyShardedDataParallel(model).to(device)

    # need simple text dataset instead of image dataset for language model

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10%]")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    # text, input_ids, attention_mask

    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # exit for now
    model = FullyShardedDataParallel(model, sharding_strategy=fsdp.ShardingStrategy.NO_SHARD, device_id=local_rank)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    ########################## Zeus ##########################
    for epoch in range(1, args.epochs + 1):
        # ds = train_dataset.shuffle(seed=args.seed + epoch)
        train(args, model, device, train_dataset, optimizer, epoch, batch_size, monitor)
    ##########################################################

    if args.save_model and local_rank == 0:
        torch.save(model.state_dict(), "qwen_2.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
