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

# os.environ["TORCH_COMPILE_DEBUG"] = "1"

rank_id = 0
pt2 = False
trace_folder_path = ''

def profiler_trace_handler(p):
    global rank_id
    global pt2
    global trace_folder_path
    
    if pt2:
        p.export_chrome_trace(f'{trace_folder_path}/resnet_dist_trace_{rank_id}_pt2.json')
    else:
        p.export_chrome_trace(f'{trace_folder_path}/resnet_dist_trace_{rank_id}.json')


def train(rank, world_size, warmups, steps, eg, profile, pt2, trace_folder):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if eg:
        if pt2:
            eg_file = f'{trace_folder}/resnet_dist_et_{rank}_pt2.json'
        else:
            eg_file = f'{trace_folder}/resnet_dist_et_{rank}.json'

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

    device = torch.device('cuda', rank)

    global rank_id
    rank_id = rank
    global trace_folder_path
    trace_folder_path = trace_folder

    model.to(device)

    # Wrap the model with DDP
    model = nn.parallel.DistributedDataParallel(model)

    if pt2:
        model = torch.compile(model)

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
    parser.add_argument("--pt2", default=False, action="store_true")

    args = parser.parse_args()

    trace_folder = f'/zhang-x3/users/ml2585/eg_logs/resnet_dist_{args.workers}'

    if not os.path.exists(trace_folder):
        os.makedirs(trace_folder)

    # Set the pt2 global variable based on command-line argument
    pt2 = args.pt2

    # Spawn the worker processes
    mp.spawn(train, args=(args.workers, args.warmups, args.steps, args.eg, args.profile, args.pt2, trace_folder), nprocs=args.workers, join=True)
