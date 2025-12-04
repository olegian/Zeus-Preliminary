import argparse
import functools
import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from zeus.monitor import ZeusMonitor
from zeus.utils.lr_scaler import LinearScaler
import nvtx
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh

# Hook into functional collectives instead!
try:
    import torch.distributed._functional_collectives as funcol
    HAS_FUNCOL = True
except ImportError:
    HAS_FUNCOL = False
    print("Warning: functional collectives not available")

N_COMM_REPEATS = 60

class FunctionalCollectiveHookManager:
    """Hooks into functional collectives used by DTensor/TP"""
    
    def __init__(self, before_hook=None, after_hook=None):
        self.before_hook = before_hook or (lambda op, *args, **kwargs: None)
        self.after_hook = after_hook or (lambda op, result: None)
        self.original_functions = {}
        self.installed = False
    
    def install(self):
        """Install hooks on functional collective operations"""
        if self.installed or not HAS_FUNCOL:
            return
        
        # collectives used by tensor parallelism (although im pretty sure its all all_reduce 
        # based on logs)
        collective_functions = [
            'all_reduce',
            'all_gather_tensor',
            'reduce_scatter_tensor',
            'all_gather_into_tensor',
            'reduce_scatter_tensor',
            'recv',
            'irecv',
            'send'
        ]
        
        print(f"Installing hooks on functional collectives...")
        for func_name in collective_functions:
            if hasattr(funcol, func_name):
                original = getattr(funcol, func_name)
                self.original_functions[func_name] = original
                print(f"  Found function: {func_name}")
                
                # Create closure
                def make_wrapper(orig, name, before_hook, after_hook):
                    @functools.wraps(orig)
                    def wrapped(*args, **kwargs):
                        rank = dist.get_rank() if dist.is_initialized() else -1

                        # Clone the input tensor(s) for measurement
                        # cause you cant have it accumulate every time
                        # cloned_args = []
                        # for arg in args:
                        #     if isinstance(arg, torch.Tensor):
                        #         cloned_args.append(arg.clone().detach())
                        #     else:
                        #         cloned_args.append(arg)
                        
                        before_hook(name, *args, **kwargs)
                        result = orig(*args, **kwargs)
                        if hasattr(result, 'wait'):
                            result.wait()
                        # for i in range(N_COMM_REPEATS - 1):
                        #     res = orig(*cloned_args, **kwargs)
                        #     if hasattr(result, 'wait'):
                        #         res.wait() # force syncronous
                        torch.cuda.synchronize()
                        after_hook(name, result)
                        
                        return result
                    return wrapped
                
                wrapped = make_wrapper(original, func_name, self.before_hook, self.after_hook)
                setattr(funcol, func_name, wrapped)
            else:
                print(f"  Function NOT found: {func_name}")
        
        self.installed = True
        print("Functional collective hooks installed successfully")
    
    def uninstall(self):
        """Remove hooks"""
        if not self.installed or not HAS_FUNCOL:
            return
        
        for func_name, original_func in self.original_functions.items():
            setattr(funcol, func_name, original_func)
        
        self.original_functions.clear()
        self.installed = False
        print("Functional collective hooks uninstalled")


def train(args, model, device, ds, optimizer: optim.Optimizer, epoch, batch_size, monitor: ZeusMonitor):
    model.to("cuda")
    model.train()
    energy_measurements = []
    n = 0
    print("warmup")
    for start_idx in range(0, len(ds), batch_size):
        batch = ds[start_idx : start_idx + batch_size]
        data = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()

        monitor.begin_window("training")
        output = model(input_ids=data, attention_mask=mask, labels=data)
        loss = output.loss
        loss.backward()

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

    # INSTALL HOOKS on functional collectives
    # measurements = []
    def pre_comm_hook(op_name, *args, **kwargs):
        pass
        # rank = dist.get_rank()
        # print(f"[Rank {rank}] → PRE {op_name}")
        # monitor.begin_window("nccl_comm")
    
    def post_comm_hook(op_name, result):
        print(f"[Rank {dist.get_rank()}] ← POST {op_name}")
        pass
        # rank = dist.get_rank()
        # res = monitor.end_window('nccl_comm')
        # measurements.append(res.total_energy / N_COMM_REPEATS)

    hooks = FunctionalCollectiveHookManager(
        before_hook=pre_comm_hook,
        after_hook=post_comm_hook,
    )
    hooks.install()
    print("measuring...")
    with nvtx.annotate("measured", color='blue'):
        energy_measurements = []
        n = 0
        for start_idx in range(0, len(ds), batch_size):
            batch = ds[start_idx : start_idx + batch_size]
            data = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()

            with nvtx.annotate(f"energy-{n}", color='red'):
                monitor.begin_window("training")
                output = model(input_ids=data, attention_mask=mask, labels=data)
                loss = output.loss
                loss.backward()

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
    # print(f"[Rank {local_rank}] FINAL MEASUREMENTS: {len(measurements)} calls, total energy: {sum(measurements)}J")
    
    # hooks.uninstall()


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N")
    parser.add_argument("--epochs", type=int, default=1, metavar="N")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--dir", default="logs", metavar="L")
    parser.add_argument("--target-accuracy", type=float, default=0.5)
    if dist.is_available():
        parser.add_argument("--backend", type=str, choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI], default=dist.Backend.GLOO)
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    args.backend = dist.Backend.NCCL
    print("Using distributed PyTorch with {} backend".format(args.backend))
    dist.init_process_group(backend=args.backend)

    monitor = ZeusMonitor(gpu_indices=[local_rank])
    batch_size = 4
    print("Chosen batch_size:", batch_size)
    args.lr = LinearScaler(bs=64, lr=0.01).compute_lr(new_bs=batch_size)

    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("loaded model and tokenizer")

    # Load datasets
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:10%]")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Setup tensor parallelism
    tp_plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    print(f"[Rank {rank}] Creating device mesh...")
    device_mesh = init_device_mesh("cuda", (world_size,))
    
    print(f"[Rank {rank}] Applying tensor parallelism...")
    for layer_id, layer in enumerate(model.model.layers):
        parallelize_module(
            module=layer,
            device_mesh=device_mesh,
            parallelize_plan=tp_plan,
        )

    print(f"[Rank {rank}] Setting up optimizer...")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print(f"[Rank {rank}] Starting training...")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataset, optimizer, epoch, batch_size, monitor)

    # if args.save_model and local_rank == 0:
    #     torch.save(model.state_dict(), "qwen_2.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()