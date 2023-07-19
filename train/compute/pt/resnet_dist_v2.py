import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.utils.data.distributed
from torch.profiler import ExecutionGraphObserver, ProfilerActivity, record_function

rank_id = 0

def profiler_trace_handler(p):
    global rank_id
    p.export_chrome_trace(f'/zhang-x3/users/ml2585/eg_logs/resnet_dist_trace_{rank_id}.json')


def distributed_train(rank, world_size, iter, profile, data_dir, debug):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    global rank_id
    rank_id = rank

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_datasets = datasets.ImageFolder(
        os.path.join(data_dir, "train"), data_transforms["train"]
    )
    val_datasets = datasets.ImageFolder(
        os.path.join(data_dir, "val"), data_transforms["val"]
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_datasets)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=128, num_workers=4, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_datasets, batch_size=128, num_workers=4, shuffle=False
    )

    device = torch.device("cuda:{}".format(rank))

    model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model = model.to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        ddp_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5
    )

    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        break

    iter_count = 0

    if profile:
        eg_fn = f'/zhang-x3/users/ml2585/eg_logs/resnet_dist_eg_{rank}.json'
        egob = ExecutionGraphObserver()
        egob.register_callback(eg_fn)
        print("start EG recording")

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=profiler_trace_handler,
        ) as pf:
            # Loop over the dataset multiple times
            for epoch in range(iter + 10):
                if debug:
                    print("Local Rank: {}, Epoch: {}, Training ...".format(rank, epoch))

                # # Save and evaluate model routinely
                # if epoch % 10 == 0:
                #     if rank == 0:
                #         accuracy = evaluate(
                #             model=ddp_model, device=device, test_loader=val_loader
                #         )
                #         print("-" * 75)
                #         print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                #         print("-" * 75)

                ddp_model.train()

                # for data in train_loader:
                iter_count += 1
                if iter_count == 20:
                    egob.start()
                if iter_count == 21:
                    egob.stop()
                    egob.unregister_callback()
                    print(f"Finish EG recording {eg_fn=}")

                if iter_count == iter:
                    return
                # inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                if debug:
                    print("loss: ", loss)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(device=device)
                pf.step()
    else:
        for epoch in range(iter + 10):
            if debug:
                print("Local Rank: {}, Epoch: {}, Training ...".format(rank, epoch))
            ddp_model.train()
            # for data in train_loader:
            iter_count += 1
            if iter_count == iter:
                return
            # inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            if debug:
                print("loss: ", loss)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device=device)
    # cleanup
    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ResNet18")
    parser.add_argument("--iter", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default='/work/zhang-x3/common/datasets/imagenet-pytorch')
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()
    # Spawn the worker processes
    mp.spawn(distributed_train, args=(args.workers, args.iter, args.profile, args.data_dir, args.debug), nprocs=args.workers, join=True)