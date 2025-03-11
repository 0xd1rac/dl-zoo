import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.datamanger import DataManager
from ebm.models.simple_ebm import SimpleEBM
from typing import Tuple

# Assuming the generalized EBM model (with support for cnn, mlp, linear, quadratic, boltzmann)
# is defined above or imported, for example:
# from simple_ebm import EBM

# Training function for one epoch.
def train(model: nn.Module, device: torch.device, train_loader: DataLoader, 
          optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Forward pass: get energy outputs for each class.
        energies = model(data)
        # Convert energies into logits: lower energy means higher probability.
        logits = -energies
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    return avg_loss

# Evaluation function.
def test(model: nn.Module, device: torch.device, test_loader: DataLoader, 
         criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            energies = model(data)
            logits = -energies
            loss = criterion(logits, target)
            test_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = DataManager.get_mnist_dataloaders()

    # Instantiate the EBM.
    # You can choose: 'cnn', 'mlp', 'linear', 'quadratic', or 'boltzmann'
    energy_function = 'cnn'
    num_classes = 10
    model = SimpleEBM(energy_function=energy_function, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy * 100:.2f}%")

    # Save the trained model.
    torch.save(model.state_dict(), f"simple_ebm_{energy_function}.pth")

if __name__ == "__main__":
    main()
