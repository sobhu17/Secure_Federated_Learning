import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import os
from core.NeuralNetwork import NeuralNetwork
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform

def get_train_test_datasets(is_train):
    return torchvision.datasets.MNIST(root="./data", train=is_train, download=True, transform=get_transform())

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = 100. * correct / total
    return test_loss, accuracy

def get_global_model(save_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(save_path):
        print(f"Existing model found at {save_path}. Loading the model.")
        global_model = NeuralNetwork().to(device)
        global_model.load_state_dict(torch.load(save_path))
        return global_model
    else:
        print(f"No existing model found at {save_path}. Please train a model first.")
        return None

def get_client_data_loaders(data_set, batch_size):
    client_data_loaders = []
    for ds in data_set:
        client_data_loaders.append(torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True))
    return client_data_loaders

def visualize_image(image_tensor):
    image = image_tensor.squeeze().numpy() 
    plt.imshow(image, cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()

# Load and preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))                
    img = np.array(img) / 255.0              
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
    return img_tensor

# Predict digit
def predict_digit(image_tensor, model, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        predicted_digit = output.argmax(dim=1).item()
        return predicted_digit
