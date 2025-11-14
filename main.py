from __future__ import print_function

import argparse
import os

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel
from zeus.monitor import ZeusMonitor
from zeus.utils.lr_scaler import LinearScaler

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 2))

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


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


def train(model: nn.Module, device, train_loader: DataLoader, optimizer: optim.Optimizer, epoch, writer, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)


def test(model, device, test_loader, writer, epoch):
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

def worker(rank, world_size, model: nn.Module, train_loader, test_loader, optimizer, epochs, writer):
    """Runs a single ranks workload"""
    setup(rank, world_size)

    device = f"cuda:{rank}"
    model = model.to(device)

    ########################## Zeus ##########################
    monitor = ZeusMonitor(gpu_indices=[rank])
    for epoch in range(1, epochs + 1):
        monitor.begin_window("training")
        train(model, device, train_loader, optimizer, epoch, writer, 10)
        train_measure = monitor.end_window("training")

        monitor.begin_window("validation")
        acc = test(model, device, test_loader, writer, epoch)
        val_measure = monitor.end_window("validation")

        print(f"Zeus Measurements: train={train_measure.total_energy}, val={val_measure.total_energy}")
    ##########################################################


    cleanup()

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
            default=dist.Backend.NCCL,
        )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(args.backend))
        # dist.init_process_group(backend=args.backend)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    ########################## Zeus ##########################
    # Fetch the batch size from the BSO server. (Oleg: this doesn't happen anymore,
    # i've disabled it. It used to be a Zeus optimzation feature to find this batch_size)
    batch_size = 32
    print("Chosen batch_size:", batch_size)
    # Scale the learning rate accordingly.
    # Default was batch size 64 and learing rate 0.01, and we use the linear
    # scaling rule since the optimizer is SGD.
    args.lr = LinearScaler(bs=64, lr=0.01).compute_lr(new_bs=batch_size)
    ##########################################################

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=False,
        **kwargs,
    )

    model = Net()
    # model = FullyShardedDataParallel(model).to(device)

    if is_distributed():
        Distributor = (
            nn.parallel.DistributedDataParallel
            if use_cuda
            else nn.parallel.DistributedDataParallelCPU
        )
        model = Distributor(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    mp.spawn(
        worker,
        args=(WORLD_SIZE, model, train_loader, test_loader, optimizer, args.epochs, writer),
        nprocs=WORLD_SIZE,
        join=True,
    )
        
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
