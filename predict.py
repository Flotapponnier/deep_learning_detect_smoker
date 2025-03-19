import torch
import torch.nn as nn
from torchvision import models
import argparse
from utils import load_image, load_model
import json
import os

def predict_image(model, image_path, class_names, device):
    """Fait une prédiction sur une image."""
    model.eval()
    image = load_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Obtenir les probabilités pour chaque classe
        results = []
        for i, (prob, class_name) in enumerate(zip(probabilities[0], class_names)):
            results.append({
                'class': class_name,
                'probability': prob.item()
            })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Chemin vers l\'image à classifier')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Chemin vers le modèle sauvegardé')
    args = parser.parse_args()

    # Vérifier si le modèle existe
    if not os.path.exists(args.model):
        print(f"Erreur: Le modèle {args.model} n'existe pas.")
        return

    # Configuration du device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Charger les noms des classes
    try:
        with open('data/class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print("Erreur: Le fichier class_names.json n'existe pas.")
        return

    # Créer et charger le modèle
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes : fumeur et non-fumeur
    model = load_model(model, args.model)
    model = model.to(device)

    # Faire la prédiction
    results = predict_image(model, args.image, class_names, device)

    # Afficher les résultats
    print("\nPrédictions:")
    print("-" * 30)
    for result in results:
        print(f"{result['class']}: {result['probability']*100:.2f}%")

    # Afficher la prédiction finale
    prediction = max(results, key=lambda x: x['probability'])
    print("\nPrédiction finale:")
    print(f"La personne est un {prediction['class']} avec une confiance de {prediction['probability']*100:.2f}%")

if __name__ == '__main__':
    main()
