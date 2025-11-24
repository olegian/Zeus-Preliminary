from __future__ import print_function

import argparse
import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorboardX import SummaryWriter
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from torchvision import transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed.fsdp as fsdp
from torch.distributed.fsdp import FullyShardedDataParallel
from zeus.monitor import ZeusMonitor
from zeus.utils.lr_scaler import LinearScaler
from zeus.device import get_gpus

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))

# TO RUN: srun --gpus-per-node=2 uv run torchrun --nproc_per_node=2 test_qwen.py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model: nn.Module, device, ds, optimizer: optim.Optimizer, epoch, batch_size, writer, monitor: ZeusMonitor):
    model.train()
    energy_measurements = []
    for start_idx in range(0, len(ds), batch_size):
        batch = ds[start_idx : start_idx + batch_size]
        data = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        monitor.begin_window("training")
        # output = model(data)
        output = model(input_ids=data, attention_mask=mask, labels=data)
        train_mes = monitor.end_window("training")
        energy_measurements.append(train_mes.total_energy)
        # loss = F.nll_loss(output, target)
        loss = output.loss
        loss.backward()
        optimizer.step()

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
            niter = epoch * (len(ds) // args.batch_size) + (start_idx // args.batch_size)
            writer.add_scalar("loss", loss.item(), niter)
        
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"{local_rank}e{epoch}|> Avg energy consumption per step: {np.array(energy_measurements).mean(-1)}")


def test(args, model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar("accuracy", float(correct) / len(test_loader.dataset), epoch)

    return float(correct) / len(test_loader.dataset)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


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

    writer = SummaryWriter(args.dir)

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
    batch_size = 32
    print("Chosen batch_size:", batch_size)
    # Scale the learning rate accordingly.
    # Default was batch size 64 and learing rate 0.01, and we use the linear
    # scaling rule since the optimizer is SGD.
    args.lr = LinearScaler(bs=64, lr=0.01).compute_lr(new_bs=batch_size)
    ##########################################################

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(
    #         "../data",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     ),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     **kwargs,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.FashionMNIST(
    #         "../data",
    #         train=False,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     ),
    #     batch_size=args.test_batch_size,
    #     shuffle=False,
    #     **kwargs,
    # )

    # model = Net().to(device)
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("loaded model and tokenizer")
    # model = FullyShardedDataParallel(model).to(device)

    # need simple text dataset instead of image dataset for language model

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

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
    # if is_distributed():
    # Distributor = (
    #     # nn.parallel.DistributedDataParallel
    #     FullyShardedDataParallel
    #     if use_cuda
    #     else nn.parallel.DistributedDataParallelCPU
    # )
    model = FullyShardedDataParallel(model, sharding_strategy=fsdp.ShardingStrategy.FULL_SHARD, device_id=local_rank)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    ########################## Zeus ##########################
    for epoch in range(1, args.epochs + 1):
        # ds = train_dataset.shuffle(seed=args.seed + epoch)
        train(args, model, device, train_dataset, optimizer, epoch, batch_size, writer, monitor)
        
    ##########################################################
        
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
