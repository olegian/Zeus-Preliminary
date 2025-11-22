import argparse
import queue
import numpy as np
from regex import W
import torch
import torch.distributed as dist
# Any other necessary imports can be added here.
import os
import torch.multiprocessing as mp

from ddp import DDPBucketedParameters
from model.adamw import AdamWOptimizer
from model.cross_entropy import cross_entropy_loss
from model.transformer_lm import TransformerLM
from zeus.monitor import ZeusMonitor
import nvtx


# You can change the function and variable names as needed.
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ============================================================
# MARK: (3) Bucketed DDP
# ============================================================
# You can change the function and variable names as needed.
def run_bucketed(
        world_size: int,
        model: torch.nn.Module,
        bucket_size_mb: int,
        data: torch.Tensor,        
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        num_warmup: int,
        num_timed: int,
        iteration_times: queue.Queue,
    ):
    """Group gradients into buckets and all-reduce each bucket."""
    print(f"===== Training Bucketed DDP Model for {num_warmup} + {num_timed} Steps =====")
    print(f"> Using bucket size: {bucket_size_mb}MB")
    mp.spawn(
        run_bucketed_ddp_worker,
        args=(world_size, model, bucket_size_mb, data, labels, optimizer, num_warmup, num_timed, iteration_times),
        nprocs=world_size,
        join=True,
    )

def run_bucketed_ddp_worker(
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        bucket_size_mb: int,
        data: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        num_warmups: int,
        num_timed: int,
        per_iteration: queue.Queue,
    ):
    assert data.shape[-2] % world_size == 0, "data batch dimension does not evenly split by world size"

    setup(rank, world_size)

    # move relevant stuff to device, rank initialized in setup()
    model.to("cuda")
    data = data.to("cuda")
    labels = labels.to("cuda")

    monitor = ZeusMonitor(gpu_indices=[rank])
    model = DDPBucketedParameters(model, bucket_size_mb)
    model.train()

    # work out splits for this batch
    batch_len = data.shape[-2] // world_size
    split_data = data[rank * batch_len : (rank+1) * batch_len, :]
    split_labels = labels[rank * batch_len : (rank+1) * batch_len, :]

    print(f"\t> {rank}: Warming up...")
    for i in range(num_warmups):
        optimizer.zero_grad(set_to_none=True)
        out = model.forward(split_data)
        loss = cross_entropy_loss(out, split_labels)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
    print(f"\t> {rank}: Finished Warmup.")

    recorded_step_energies = []
    print(f"\t> {rank}: Start step-timed training loop on batches [{rank * batch_len}, {(rank+1) * batch_len})...")
    for i in range(num_timed):
        optimizer.zero_grad(set_to_none=True)

        n_mults = 10_000
        # xys = [
        #     (torch.rand(size=(128, 128)).to("cuda"),
        #      torch.rand(size=(128, 128)).to("cuda"))
        #     for _ in range(n_mults)
        # ]
        n_params = 20_000
        xs = [ torch.rand(size=(n_params, )).to("cuda") for _ in range(n_mults)]
        if rank == 0:
            xs = [-x for x in xs]
        with nvtx.annotate("step", color='green'):
            # # out = model.forward(split_data)
            # # loss = cross_entropy_loss(out, split_labels)
            # # loss.backward()
            # # model.finish_gradient_synchronization()
            # # optimizer.step()
            # for i in range(n_mults):
            #     # a = xys[i][0] @ xys[i][1]
            monitor.begin_window("train")
            for i in range(n_mults):
                dist.all_reduce(xs[i], op=dist.ReduceOp.SUM)
            mes = monitor.end_window("train")

        recorded_step_energies.append(mes.total_energy)
    
    print(f"\t> {rank}: Finished step-timed training loop.")

    # have every rank send their respective times
    per_iteration.put(recorded_step_energies)
    
    cleanup()


# ============================================================
# Benchmark Function
# ============================================================
# You can change the function and variable names as needed.
def main():
    """Benchmark DDP variants on the Transformer model."""
    parser = argparse.ArgumentParser(description="Benchmark optimized DDP variants.")
    parser.add_argument(
        "--bucket-size-mb",
        type=int,
        default=10,
        help="Bucket size (in MB) for the bucketed DDP variant.",
    )
    args = parser.parse_args()

    # Run parameters
    num_iters, num_warmup = 20, 5
    world_size = 2
    vocab_size = 100
    # context_length = 100
    # batch_dim = 20
    context_length = 10
    batch_dim = 90
    
    # DDP setup
    # TODO: Initialize distributed process group
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    iteration_time_queue = manager.Queue()

    # Construct model and move to GPU
    # TODO: Define model parameters
    model = TransformerLM(
        d_model=1280,
        d_ff=5120,
        num_layers=36,
        num_heads=20,
        theta=10_000,
        max_seq_len=context_length,
        vocab_size=vocab_size,
        context_length=context_length,
    )

    # Construct optimizer
    optimizer = AdamWOptimizer(
        params=model.parameters(), 
        lr=0.001, 
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
    
    # Dummy data
    data = torch.randint(low=0, high=vocab_size, size=(batch_dim, context_length), dtype=torch.int64)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_dim, context_length,), dtype=torch.int64)

    run_bucketed(
        world_size,
        model,
        args.bucket_size_mb,
        data,
        labels,
        optimizer,
        num_warmup,
        num_iters, 
        iteration_time_queue,
    )

    total = []
    while not iteration_time_queue.empty():
        measurements_on_device = iteration_time_queue.get()
        for mes in measurements_on_device:
            total.append(mes)

    print(total)
    total = sum(total)
    print("Total Energy Consumed", total)
    print("Per Device", total / 2)
    print("Per Iteration", (total / 2) / num_iters)


if __name__ == "__main__":
    main()