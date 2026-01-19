from Data_Preparation.data_preparation import test_loader
from Model.model import ClassifierCNN
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch

model = ClassifierCNN()

def evaluate_model():
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.001)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        bar = tqdm(test_loader)
        for batch, labels in bar:
            logits = model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            test_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {100 * (correct / total):.2f}%")           

evaluate_model()