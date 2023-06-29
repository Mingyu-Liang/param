import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.profiler import ExecutionGraphObserver, ProfilerActivity, record_function


print("PyTorch version: ", torch.__version__)

# Define transforms for the training and validation datasets
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet dataset
trainset = torchvision.datasets.ImageFolder(root='/work/zhang-x3/common/datasets/imagenet-pytorch/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

valset = torchvision.datasets.ImageFolder(root='/work/zhang-x3/common/datasets/imagenet-pytorch/val', transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)

# Define the ResNet18 model
model = torchvision.models.resnet18(pretrained=False)

torch.set_num_interop_threads(10)

def profiler_trace_handler(p):
    p.export_chrome_trace("/zhang-x3/users/ml2585/eg_logs/resnet_trace_iters.json")

def train(warmups, steps, profile, eg, cuda):
    # Use GPU if available
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    if eg:
        eg_file = f"/zhang-x3/users/ml2585/eg_logs/resnet_eg_iter{warmups}.json"
        eg = ExecutionGraphObserver()
        eg.register_callback(eg_file)
    
    if profile:
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU,
                        ProfilerActivity.CUDA],
            on_trace_ready=profiler_trace_handler,
        ) as pf:
            iterations = 0
            # Train the model
            for epoch in range(10):
                # running_loss = 0.0
                accumulated_loss = torch.cuda.FloatTensor([0.0])
                for i, data in enumerate(trainloader, 0):
                    if eg:
                        if i == warmups:
                            eg.start()
                        if i == warmups + 1:
                            eg.stop()
                            eg.unregister_callback()
                    
                    with record_function(f"iteration#{iterations}"):
                        # start_time = time.time_ns()
                        inputs, labels = data[0].to(device), data[1].to(device)

                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # running_loss += loss.item()
                        accumulated_loss += loss

                        pf.step()
                        iterations += 1

                        # print('Per step time: ', (time.time_ns() - start_time) / 1000000.0)

                        if iterations == warmups + steps:
                            return
    else:
        iterations = 0
        # Train the model
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                if eg:
                    if i == warmups:
                        eg.start()
                    if i == warmups + 1:
                        eg.stop()
                        eg.unregister_callback()
                
                # start_time = time.time_ns()
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                iterations += 1
                if iterations == warmups + steps:
                    return

            # # Validate the model
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     for data in valloader:
            #         images, labels = data[0].to(device), data[1].to(device)
            #         outputs = model(images)
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum().item()

            # print('Epoch %d: Accuracy on validation set: %d %%' % (epoch+1, 100 * correct / total))

    print('Finished Training')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="ResNet18")
    parser.add_argument("--eg", action="store_true")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cuda", type=int, default=0)

    args = parser.parse_args()

    train(args.warmups, args.steps, args.profile, args.eg, args.cuda)
