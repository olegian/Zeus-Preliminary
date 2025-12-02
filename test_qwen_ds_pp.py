from __future__ import print_function

import argparse
import os

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

# WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))

# TO RUN: srun --gpus-per-node=2 uv run deepspeed test_qwen_ds.py

# TO RUN: srun --gpus-per-node=2 uv run nsys profile -o qwen_ds_2 deepspeed test_qwen_ds.py

class MyDatasetIterator:
    def __init__(self, ds, model):
        self.ds = ds
        self.model = model
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.index < len(self.ds):
            batch = self.ds[self.index]
            loss = self.model.train_batch(batch)
            data = batch["input_ids"].to(self.model.device)
            mask = batch["attention_mask"].to(self.model.device)
            self.index += 1
            return (loss, data, mask)
        else:
            raise StopIteration

def train(args, model: deepspeed.PipelineEngine, ds, optimizer: optim.Optimizer, epoch, batch_size, monitor: ZeusMonitor):
    model.train()
    energy_measurements = []
    n = 0
    print("warmup")

    loss = model.train_batch()
    print("loss:", loss)
    # loss = model.train_batch(data_iter=train_iter)
    # print("loss:", loss)
    #     loss = model.train_batch(batch)
    #     data = batch["input_ids"].to(model.device)
    #     mask = batch["attention_mask"].to(model.device)

    #     monitor.begin_window("training")

    #     loss = model(input_ids=data, attention_mask=mask, labels=data).loss
        
    #     model.backward(loss)
    #     model.step()

    #     train_mes = monitor.end_window("training")
    #     energy_measurements.append(train_mes.total_energy)

    
    
    #     if i % args.log_interval == 0:
    #         print(
    #             "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
    #                 epoch,
    #                 i,
    #                 len(ds),
    #                 100.0 * i / len(ds),
    #                 loss.item(),
    #             )
    #         )

    #     if n == 5:
    #         break

    #     n += 1

    print("measuring...")
    with nvtx.annotate("measured", color='blue'):
        energy_measurements = []
        n = 0
        for i, batch in enumerate(ds):
            break
            data = batch["input_ids"].to(model.device)
            mask = batch["attention_mask"].to(model.device)

            # with model.no_sync():
            with nvtx.annotate(f"energy-{n}", color='red'):
                monitor.begin_window("training")
                loss = model(input_ids=data, attention_mask=mask, labels=data).loss
                model.backward(loss)
                model.step()

                train_mes = monitor.end_window("training")
                energy_measurements.append(train_mes.total_energy)

            if i % (args.log_interval * batch_size) == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                        epoch,
                        i,
                        len(ds),
                        100.0 * i / len(ds),
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

    model = PipelineModule(
        layers=[model],
        num_stages=1,
        loss_fn=nn.CrossEntropyLoss(),
        partition_method="parameters",
    )

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

    model, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=train_dataset,
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
