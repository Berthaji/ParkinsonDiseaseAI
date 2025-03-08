import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import csv
import numpy as np

def evaluate_model(model, test_loader, device, model_save_path="model.pth", metrics_save_path="metrics.csv"):
    model.eval()  # modalità di valutazione
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcolo delle metriche
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted') * 100  
    recall = recall_score(all_labels, all_predictions, average='weighted') * 100 
    f1 = f1_score(all_labels, all_predictions, average='weighted') * 100 

    # Calcolo della matrice di confusione
    cm = confusion_matrix(all_labels, all_predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # matrice di confusione per percentuali

    # Stampa delle metriche
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # Salvataggio le metriche in un file CSV
    with open(metrics_save_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["F1 Score", f1])

       
        writer.writerow([]) 
        writer.writerow(["Confusion Matrix"])
        writer.writerow(["Predicted 0", "Predicted 1"])
        for row in cm:
            writer.writerow([f"{value:.2f}" for value in row])

    print(f"Metriche salvate in: {metrics_save_path}")