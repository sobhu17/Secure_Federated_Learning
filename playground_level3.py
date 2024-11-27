import torch
import os
from utilities.utils import get_global_model, get_train_test_datasets, get_client_data_loaders, evaluate
from torch.utils.data import DataLoader, random_split
from core.NeuralNetwork import NeuralNetwork
from core.Client import Client
from core.FederatedServer import FederatedServer
from utilities.attackutils import is_malicious

def federated_training(
        num_clients=10, num_rounds=10, 
        local_epochs=5, batch_size=64, 
        save_path="./secure_federated_learning.pth"):
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    train_dataset = get_train_test_datasets(is_train=True)
    test_dataset = get_train_test_datasets(is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split training data among clients
    data_per_client = len(train_dataset) // num_clients
    client_datasets = random_split(train_dataset, [data_per_client] * num_clients)
    client_loaders = get_client_data_loaders(client_datasets, batch_size)

    global_model = NeuralNetwork().to(device)
    server = FederatedServer(global_model)

    # Initialize clients
    clients = []
    for i in range(num_clients):
        client = Client(global_model, client_loaders[i], device, i)
        clients.append(client)

    clients_data = {
        "clients": clients,
    }

    torch.save(clients_data, "./secure_clients_data.pth")

    # Check if model exists
    if os.path.exists(save_path):
        print(f"Existing model found at {save_path}. Loading the model.")
        global_model.clients = clients
        return get_global_model(save_path=save_path, batch_size=batch_size)
    
    # Training rounds
    for round in range(num_rounds):
        print(f"\nRound {round + 1}/{num_rounds}")

        # Client updates
        client_parameters = []
        for client_idx, client in enumerate(clients):
            parameters = client.train(epochs=local_epochs)
            client_parameters.append(parameters)
            print(f"Client {client_idx + 1}/{num_clients} completed local training")

        global_model = server.aggregate_parameters(client_parameters, security=True)

        # Evaluate global model
        test_loss, accuracy = evaluate(global_model, test_loader, device)
        print(f"Round {round + 1} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Update all clients with new global model
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
    
    # Save the final model
    torch.save(global_model.state_dict(), save_path) 
    print(f"Final robust model saved to {save_path}")
    return global_model


if __name__ == "__main__":
    model = federated_training()
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("secure_federated_learning.pth", map_location=device))
    test_dataset = get_train_test_datasets(is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=64)
    test_loss, accuracy = evaluate(model, test_loader, device)
    print(f"Pre-trained model - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


    