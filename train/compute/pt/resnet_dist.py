import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data.distributed
from torch.profiler import ExecutionTraceObserver, ProfilerActivity, record_function

# print("PyTorch version: ", torch.__version__)

rank_id = 0

def profiler_trace_handler(p):
    global rank_id
    p.export_chrome_trace(f'/zhang-x3/users/ml2585/eg_logs/resnet_dist_trace_{rank_id}.json')

def train(rank, world_size, warmups, steps, eg, profile, compile):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if eg:
        eg_file = f'/zhang-x3/users/ml2585/eg_logs/resnet_dist_eg_{rank}.json'
        eg = ExecutionTraceObserver()
        eg.register_callback(eg_file)

        eg.start()
        # Initialize the process group
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        eg.stop()
    else:
        # Initialize the process group
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    # Set up the ResNet-18 model
    model = models.resnet18()

    if compile:
        model = torch.compile(model)

    device = torch.device('cuda', rank + 1)

    global rank_id
    rank_id = rank

    model.to(device)

    # Wrap the model with DDP
    model = nn.parallel.DistributedDataParallel(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 3000

    # Set up the data loading for distributed training
    train_dataset = datasets.ImageFolder(root='/work/zhang-x3/common/datasets/imagenet-pytorch/train', transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                               shuffle=False, num_workers=4, sampler=train_sampler)

    if profile:
        with torch.profiler.profile(
                activities=[ProfilerActivity.CPU,
                            ProfilerActivity.CUDA],
                on_trace_ready=profiler_trace_handler,
                schedule=torch.profiler.schedule(
                    # skip_first=0,
                    wait=warmups + 5,
                    warmup=5,
                    active=steps
                ),
            ) as pf:
            iterations = 0
            # Training loop
            for epoch in range(5):  # Run for 5 epochs
                train_sampler.set_epoch(epoch)  # Update the sampler for distributed shuffling

                for batch_idx, (data, target) in enumerate(train_loader):
                    if eg:
                        if batch_idx == warmups:
                            eg.start()
                        if batch_idx == warmups + 1:
                            eg.stop()
                            eg.unregister_callback()

                    # with record_function(f"iteration#{iterations}"):
                    optimizer.zero_grad()  # Clear gradients
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)  # Forward pass
                    loss = criterion(output, target)  # Compute loss
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update weights
                    
                    pf.step()
                    iterations += 1
                    if iterations == warmups + steps + 10:
                        return
    else:
        iterations = 0
        # Training loop
        for epoch in range(5):  # Run for 5 epochs
            train_sampler.set_epoch(epoch)  # Update the sampler for distributed shuffling

            for batch_idx, (data, target) in enumerate(train_loader):
                if eg:
                    if batch_idx == warmups:
                        eg.start()
                    if batch_idx == warmups + 1:
                        eg.stop()
                        eg.unregister_callback()

                optimizer.zero_grad()  # Clear gradients
                data = data.to(device)
                target = target.to(device)
                output = model(data)  # Forward pass
                loss = criterion(output, target)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
            
                iterations += 1
                if iterations == warmups + steps:
                    return

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ResNet18")
    parser.add_argument("--eg", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compile", default=False, action="store_true")

    args = parser.parse_args()
    # Spawn the worker processes
    mp.spawn(train, args=(args.workers, args.warmups, args.steps, args.eg, args.profile, args.compile), nprocs=args.workers, join=True)
