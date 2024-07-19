import torch
import torch.nn as nn
import torch.quantization
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets
import pickle
import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

model = torch.jit.load('static_.pth', map_location=device)

import time


def evaluate_model(model, device):
    model = model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            st = time.time()
            outputs = model(images)
            et = time.time()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct // total} %')
    print('Elapsed time = {:0.4f} milliseconds'.format((et - st) * 1000))
    print("====================================================================================================")


device = 'cpu'
print("=====================================AFTER WEIGHT PRUNING========================================")
evaluate_model(model, device)

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

# img, label = testset[9]

# evens = list(range(0, len(trainset), 2))
odds = list(range(1, len(test_dataset), 100))
test_data = torch.utils.data.Subset(test_dataset, odds)

for img, label in test_data:
    model.eval()
    # img_pil = transform(img)  # Convert to PIL Image
    # print(f"Image shape after ToPILImage: {img_pil.size}")
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, test = torch.max(output, 1)
        print(f"Predicted: {classes[test.item()]}, Target: {classes[label]}")