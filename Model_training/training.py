from Data_Preparation.data_preparation import train_loader
from Model.model import ClassifierCNN
import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim import SGD



def train_model(model, train_loader, epochs=10):

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001)   
    model.train()
    bar = tqdm(train_loader)
    for epoch in range(epochs):
        train_loss = 0.0
        for batch, labels in bar:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")

model = ClassifierCNN()
train_model(model, train_loader)