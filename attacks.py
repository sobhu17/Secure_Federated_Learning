import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utilities.attackutils import detect_random_noise_injection, detect_model_poisoning, detect_targeted_model_poisoning, is_malicious

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_data = torch.load("./secure_clients_data.pth")
clients = clients_data["clients"]
client = clients[0]
parameters = client.train(epochs=5)


def get_noisy_parameters(parameters):
    noisy_parameters = {}
    for key, value in parameters.items():
        noisy_parameters[key] = value + torch.randn_like(value) * 0.1
    return noisy_parameters

noisy_parameters = get_noisy_parameters(parameters)
attack_detected = detect_random_noise_injection(noisy_parameters, client.model)
print(f"Is attack detected? {attack_detected}")


def get_poisoned_parameters(poison_factor=10, noise_level=0.1):
    poisoned_parameters = {}
    for key, value in parameters.items():
        # Add noise to weights
        noise = torch.randn_like(value) * noise_level
        poisoned_value = value + noise

        # Scale the weights (mimic Byzantine behavior)
        poisoned_value *= poison_factor

        poisoned_parameters[key] = poisoned_value
    
    return poisoned_parameters

poisoned_parameters = get_poisoned_parameters()
attack_detected = detect_model_poisoning(poisoned_parameters, client.model)
print(f"Is attack detected? {attack_detected}")


def targeted_poisoning_attack(model, target_class, poison_factor=10, noise_level=0.1):
    poisoned_parameters = {}

    # Identify the layers related to output (e.g., final classification layer)
    for name, param in model.named_parameters():
        # Target the final layer weights and biases (classification layer)
        if "fc2" in name:
            # Focus the attack on the target class by modifying the final output layer's weights
            if "weight" in name:  # Modifying the weights related to output
                poisoned_param = param.clone()
                poisoned_param[target_class] += torch.randn_like(param[target_class]) * noise_level  # Inject noise
                poisoned_param[target_class] *= poison_factor  # Amplify the attack
                poisoned_parameters[name] = poisoned_param
            elif "bias" in name:  # Modifying the bias term for target class
                poisoned_param = param.clone()
                poisoned_param[target_class] += torch.randn_like(param[target_class]) * noise_level  # Inject noise
                poisoned_param[target_class] *= poison_factor  # Amplify the attack
                poisoned_parameters[name] = poisoned_param
        else:
            # Keep other layers intact, as we're targeting the final layer for class manipulation
            poisoned_parameters[name] = param

    return poisoned_parameters

poisoned_parameters = targeted_poisoning_attack(client.model, target_class=5)

for key, value in parameters.items():
    parameters[key] = value.to(device)

for key, value in poisoned_parameters.items():
    poisoned_parameters[key] = value.to(device)


def detect_targeted_poisoning(poisoned_parameters, original_parameters, threshold=100):
    diff = 0
    for key in poisoned_parameters:
        diff += torch.sum((poisoned_parameters[key] - original_parameters[key]) ** 2).item()

    if diff > threshold:
        return True  # Attack detected
    return False

# Detect if the poisoning attack has been applied successfully
attack_detected = detect_targeted_poisoning(poisoned_parameters, parameters)
print(f"Is attack detected? {attack_detected}")