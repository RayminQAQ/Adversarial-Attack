import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model import ConvNet

def loadDataset(dataset_root: str) -> ImageFolder:
    _transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(28, 28), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    return ImageFolder(dataset_root, transform=_transform)

def splitDataset(dataset, batch) -> tuple:
    # Compute sizing
    train_size = int(0.8 * len(dataset))  
    test_size = len(dataset) - train_size
    
    # split
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)
    return train_loader, test_loader

def makeModel(class_num, device):
    model = ConvNet(class_num)
    model.to(device)
    return model

def train(model, train_loader, loss_fn, optimizer, idx, device) -> tuple:  # Returning history_accuracy, history_loss
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # Move images and labels to the appropriate device (e.g., GPU or CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Clear the gradients before each instance
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute the loss
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Compute the number of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy over an epoch
    average_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f"Epoch {idx}, Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Return the accuracy and average loss
    return accuracy, average_loss

def test(model, test_loader, device) -> tuple: # history_accuracy, history_loss
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {(correct/total)*100:.2f}%")
    
    return (correct/total), 

def saveResult(root_path, model, epoch, batch_size, history : dict):
    hisFolder = 1 # initial value for the history info directory
    savedDirPath = ""
    
    while True:
        if os.path.isdir(f"{root_path}/his-{hisFolder}"):
            hisFolder += 1
        else:
            savedDirPath = f"{root_path}/his-{hisFolder}"
            os.mkdir(savedDirPath)        
            break
    
    # draw plot into path
    _drawPlot(history, savedDirPath)
    
    # save hyperParameter.txt
    with open(f"{savedDirPath}/hyperParameter.txt", 'w') as file:
        file.write(f"Epoch: {epoch}\n")
        file.write(f"Batch size: {batch_size}\n")
        
    # save model
    model_path = f"{savedDirPath}/model.pth"
    torch.save(model, model_path)

def _drawPlot(history, path):
    """
    Generate and save a single plot with multiple metrics data over training epochs.

    Parameters:
        history (dict): A dictionary containing lists of metrics data. Each key is a metric name.
        path (str): Directory path to save the generated plot.

    The function plots each metric on the same graph using different colors and markers.
    """
    # Assuming all lists are the same length and non-empty, calculate the number of epochs from the first list
    if history:
        epochs = list(range(1, len(next(iter(history.values()))) + 1))

        for key, values in history.items():
            if values:  # Only plot if there are data points
                plt.figure()
                plt.title(f"{key} over {len(values)} epochs")
                plt.plot(epochs, values, 'b-o')  # Assume the values are accuracy data
                
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                
                plt.savefig(f'{path}/{key}.png')    
                plt.close()
                
                print(f"History saved in {path}/{key}.png")
    
    # Loop through each metric in the dictionary to plot them on the same figure
    plt.figure()
    for key, values in history.items():
        if values:  # Only plot if there are data points
            plt.plot(epochs, values, marker='o', label=key)  # Plot each key with a unique line and marker

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()  # Add a legend to differentiate the lines

    # Save the combined plot to the specified path
    plt.savefig(f'{path}/combined_metrics.png')    
    plt.close()
    
    print(f"Combined metrics history saved in {path}/combined_metrics.png")

def main(dataPath, savedPath, batch_size, epoch_size):
    # Initialize
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = {
        "history_train_acc" : [],
        "history_train_loss" : [],
        "history_test_acc" : [],
        # "history_test_loss" : [],
    }
    
    # Load Data
    dataset = loadDataset(dataPath)
    trainLoader, testLoader = splitDataset(dataset, batch_size)
    
    # Load Model
    model = makeModel(len(dataset.classes), DEVICE)
    
    # Define epochs_size, loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Train & Test
    for round in range(1, epoch_size + 1):
        train_acc, train_loss = train(model, trainLoader, loss_fn, optimizer, round, DEVICE)
        test_acc = test(model, testLoader, DEVICE)
        # append
        history["history_train_acc"].append(train_acc)
        history["history_train_loss"].append(train_loss)
        history["history_test_acc"].append(test_acc)
        # history["history_test_loss"].append(test_loss)

    # save result
    if not os.path.isdir(savedPath):
        os.makedirs(savedPath)
    saveResult(root_path=savedPath, model=model, epoch=epoch_size, batch_size=batch_size, history=history)

if __name__ == '__main__':
    dataPath = "image_fromTA"
    savedPath = "history"
    batch_size = 32
    epoch_size = 2
    main(dataPath, savedPath, batch_size, epoch_size)    