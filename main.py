import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
import os
from train import train_model
from preprocess import preprocess_images
from evaluate import evaluate_model

# Impostazioni di configurazione
batch_size = 32
learning_rate = 0.001
num_epochs = 15

# Trasformazioni: normalizzazione e ridimensionamento
transform = transforms.Compose([
    transforms.Resize(256),  # Ridimensiona l'immagine
    transforms.CenterCrop(224),  # Crop centrale per MobileNet
    transforms.ToTensor(),  # Converte l'immagine in un tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione per MobileNet
])

# Caricamento dei dati di addestramento
data_dir = 'drawings'
spirals_train_dir = os.path.join(data_dir, 'spiral', 'training')
waves_train_dir = os.path.join(data_dir, 'wave', 'training')

train_data = datasets.ImageFolder(spirals_train_dir, transform=transform)
train_data = datasets.ImageFolder(waves_train_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Caricamento dei dati di test
spirals_test_dir = os.path.join(data_dir, 'spiral', 'testing')
waves_test_dir = os.path.join(data_dir, 'wave', 'testing')

test_data = datasets.ImageFolder(spirals_test_dir, transform=transform)
test_data = datasets.ImageFolder(waves_test_dir, transform=transform)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Creazione del modello MobileNet preaddestrato
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Congeliamo i parametri preaddestrati e li aggiorniamo solo per i nuovi strati
for param in model.parameters():
    param.requires_grad = False
# Sblocca solo gli strati finali
for param in model.classifier.parameters():
    param.requires_grad = True 

# Modifica l'ultimo strato per il nostro problema (2 classi: sano vs malato)
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.5),  # Aggiungi dropout con probabilità di 0.5
    nn.Linear(model.last_channel, 2)  # L'ultimo layer per classificare in 2 classi
)

# Spostiamo il modello su GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.to(device)

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-4)

# Funzione di addestramento
train_model(model, train_loader, criterion, optimizer, num_epochs, device)
model_save_path="model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Modello salvato in: {model_save_path}")
# Testiamo il modello sui dati di test
evaluate_model(model, test_loader, device)