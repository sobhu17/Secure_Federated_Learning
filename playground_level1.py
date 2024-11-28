import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from utilities.utils import get_train_test_datasets
from core.NeuralNetwork import NeuralNetwork
from utilities.utils import preprocess_image, visualize_image, predict_digit


def train(model, train_loader, device, epochs=10, learning_rate=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    print("Training completed!")
    return model

# Evaluate the model
def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Main training and evaluation function
def main(pretrained=False, batch_size=64, save_path="mnist_cnn.pth"):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = get_train_test_datasets(is_train=True)
    test_dataset = get_train_test_datasets(is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNetwork().to(device)
    if pretrained:
        # Load pre-trained parameters and move to device
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        print("Loaded pre-trained model parameters.")
    else:
        # Train the model
        model.to(device)
        train(model, train_loader, device)
        # Save the trained model
        torch.save(model.state_dict(), save_path)
        print(f"Model parameters saved to {save_path}.")

    # Evaluate the model
    evaluate(model, test_loader, device)
    return model




if __name__ == "__main__":
    # Change `pretrained` to True to load pre-trained parameters
    if os.path.exists("mnist_cnn.pth"):
        model = main(pretrained=True)
    else:
        model = main(pretrained=False)
    print(type(model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "images/7-2.png"
    image_tensor = preprocess_image(image_path)
    visualize_image(image_tensor)
    predicted_digit = predict_digit(image_tensor, model, device)
    print(f"Predicted Digit: {predicted_digit}")