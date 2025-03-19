import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
from utils import create_directories, get_transforms, save_model
import argparse

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Fonction d'entraînement du modèle."""
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Phase d'entraînement
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Phase de validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Sauvegarde du meilleur modèle
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_model(model, 'models/best_model.pth')
            print(f'Nouveau meilleur modèle sauvegardé avec une précision de {best_acc:.4f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille du batch')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Taux d\'apprentissage')
    args = parser.parse_args()

    # Création des répertoires
    create_directories()

    # Configuration du device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Chargement des données
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder('data/Training', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/Validation', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Création du modèle
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes : fumeur et non-fumeur
    model = model.to(device)

    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Entraînement du modèle
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device)

if __name__ == '__main__':
    main()
