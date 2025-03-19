import os
import torch
from torchvision import transforms
from PIL import Image

def create_directories():
    """Crée les répertoires nécessaires pour le projet."""
    directories = ['data/Training', 'data/Validation', 'data/Testing', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_transforms():
    """Définit les transformations pour l'augmentation des données."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def load_image(image_path):
    """Charge et prépare une image pour la prédiction."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

def save_model(model, path):
    """Sauvegarde le modèle."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Charge le modèle."""
    model.load_state_dict(torch.load(path))
    return model
