import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
import os
from train import train_model
from evaluate import evaluate_model
from preprocess import preprocess_images

def adjust_mobilenet(model):

    for param in model.parameters():
        param.requires_grad = False
    
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout al 50%
        nn.Linear(model.last_channel, 2)  # ultimo layer per classificare in 2 classi
    )

def adjust_resnet(model):
   
    for param in model.parameters():
        param.requires_grad = False
    
    # Scongela solo l'ultimo strato completamente connesso (fc)
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  # Dropout 50%
    nn.Linear(model.fc.in_features, 2)  # ultimo layer per classificare in 2 classi
)


def test_model(model_name = 'mobilenet', preprocess=False):
    # Configurazione
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 12

    # Trasformazioni: normalizzazione e ridimensionamento
    transform = transforms.Compose([
        transforms.Resize(256),  # Ridimensiona l'immagine
        transforms.CenterCrop(224),  # Crop centrale
        transforms.ToTensor(),  # Converte l'immagine in un tensore
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione 
    ])

    # Caricamento dei dati di addestramento
    data_dir = 'drawings'
    spirals_train_dir = os.path.join(data_dir, 'spiral', 'training')
    waves_train_dir = os.path.join(data_dir, 'wave', 'training')

    if preprocess == True:
        preprocess_images(spirals_train_dir)
        preprocess_images(waves_train_dir)

    train_data = datasets.ImageFolder(spirals_train_dir, transform=transform)
    train_data = datasets.ImageFolder(waves_train_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Caricamento dei dati di test
    spirals_test_dir = os.path.join(data_dir, 'spiral', 'testing')
    waves_test_dir = os.path.join(data_dir, 'wave', 'testing')

    test_data = datasets.ImageFolder(spirals_test_dir, transform=transform)
    test_data = datasets.ImageFolder(waves_test_dir, transform=transform)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    if model_name == "mobilenet":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        adjust_mobilenet(model)
        model_save_path="model_mobilenet.pth"
        metrics_save_path="metrics_mobilenet.csv"
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-4)

    if model_name == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        adjust_resnet(model)
        model_save_path="model_resnet.pth"
        metrics_save_path="metrics_resnet.csv"
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Spostamento del modello su GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)

    # Definizione della funzione di perdita
    criterion = nn.CrossEntropyLoss()

    # Funzione di addestramento
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    torch.save(model.state_dict(), model_save_path)
    print(f"Modello salvato in: {model_save_path}")

    # Test del modello
    evaluate_model(model, test_loader, device, model_save_path, metrics_save_path)