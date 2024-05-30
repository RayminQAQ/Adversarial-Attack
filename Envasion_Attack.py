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
from utils import NoisyDataset
     
def loadDataset(dataset_root: str) -> ImageFolder:
    _transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(28, 28), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    return ImageFolder(dataset_root, transform=_transform)

def splitDataset(dataset) -> tuple:
    # Compute sizing
    train_size = int(0.8 * len(dataset))  
    test_size = len(dataset) - train_size
    
    # split
    train_set, test_set = random_split(dataset, [train_size, test_size])
    return train_set, test_set

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

def test(model, test_loader, device) -> tuple:  # Returns overall accuracy and per-class accuracies
    model.eval()
    correct = 0
    total = 0

    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update per-class accuracy stats
            for label, prediction in zip(labels, predicted):
                if label.item() in class_correct:
                    class_correct[label.item()] += (prediction == label).item()
                else:
                    class_correct[label.item()] = (prediction == label).item()

                if label.item() in class_total:
                    class_total[label.item()] += 1
                else:
                    class_total[label.item()] = 1

    # Compute overall accuracy
    overall_accuracy = (correct / total) * 100
    print(f"Test Accuracy: {overall_accuracy:.2f}%")

    # Compute per-class accuracies
    class_accuracies = {cls: 100 * class_correct[cls] / class_total[cls] for cls in class_correct if class_total[cls] > 0}

    return overall_accuracy, class_accuracies

def saveResult(root_path, model, epoch, batch_size, history, histogram_info):
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
    _drawHistogram(histogram_info, savedDirPath)
    
    # save hyperParameter.txt
    with open(f"{savedDirPath}/hyperParameter.txt", 'w') as file:
        file.write(f"Epoch: {epoch}\n")
        file.write(f"Batch size: {batch_size}\n")
        file.write(f"Train Accuracy: {history['history_train_acc']}\n")
        file.write(f"Train Loss: {history['history_train_loss']}\n")
        file.write(f"Test Accuracy: {history['history_test_acc']}\n")
         
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

def _drawHistogram(class_accuracies, path):
    """
    Plot and save a histogram of class-wise accuracies.

    Parameters:
        class_accuracies (dict): A dictionary where keys are class indices and values are accuracies.
        path (str): Directory path to save the generated histogram.
    """
    classes = list(class_accuracies.keys())
    accuracies = [class_accuracies[cls] for cls in classes]
    plt.figure()
    plt.bar(classes, accuracies, color='blue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.savefig(f'{path}/class_accuracies_histogram.png')
    plt.close()
    print(f"Class-wise accuracy histogram saved in {path}/class_accuracies_histogram.png")

def main(dataPath, savedPath, batch_size, epoch_size, poison_fraction=0.1, noise_level=0):
    # Initialize
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = {
        "history_train_acc" : [],
        "history_train_loss" : [],
        "history_test_acc" : [],
        # "history_test_loss" : [],
    }
    histogram = {}
    
    # Load Data
    dataset = loadDataset(dataPath)
    train_set, test_set = splitDataset(dataset)
    
    # Envasion Attack
    test_set = NoisyDataset(test_set, poison_fraction, noise_level)
    
    # Load 
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Load Model
    model = makeModel(len(dataset.classes), DEVICE)
    
    # Define epochs_size, loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Train & Test
    print(f"==========================================================")
    for round in range(1, epoch_size + 1):
        train_acc, train_loss = train(model, trainLoader, loss_fn, optimizer, round, DEVICE)
        test_acc, _histogram = test(model, testLoader, DEVICE)
        # append
        history["history_train_acc"].append(train_acc)
        history["history_train_loss"].append(train_loss)
        history["history_test_acc"].append(test_acc)
        # history["history_test_loss"].append(test_loss)

        if round == epoch_size:
            histogram = _histogram
            
        print(f"==========================================================")
        

    # save result
    if not os.path.isdir(savedPath):
        os.makedirs(savedPath)
    saveResult(root_path=savedPath, model=model, epoch=epoch_size, batch_size=batch_size, 
               history=history, histogram_info=histogram)

if __name__ == '__main__':
    dataPath = "image_fromTA"
    savedPath = "result_EnvAttack"
    batch_size = 32
    epoch_size = 20
    
    # For poison attack: poison_labels
    poison_fraction = 0.1
    noise_level = 0.1
    
    main(dataPath, savedPath, batch_size, epoch_size, poison_fraction, noise_level)    