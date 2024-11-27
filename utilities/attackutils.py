import numpy as np
from scipy.spatial.distance import cosine

def extract_model_weights(model):
    if isinstance(model, dict):
        weights = [np.array(w).flatten() for w in model.values()]
        return np.concatenate(weights)
    
    if hasattr(model, 'state_dict'):
        weights = []
        for param in model.state_dict().values():
            weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(weights)
    
    if hasattr(model, 'get_weights'):
        return np.concatenate([w.flatten() for w in model.get_weights()])
    try:
        return np.array(model).flatten()
    except:
        raise TypeError("Unable to extract weights from the given model type")

def detect_random_noise_injection(client_update, global_model):
    client_weights = extract_model_weights(client_update)
    global_weights = extract_model_weights(global_model)
    
    variance_ratio = np.var(client_weights) / np.var(global_weights)
    if variance_ratio < 1 or variance_ratio > 2.5:
        return True
    return False

def detect_model_poisoning(client_update, global_model, threshold=10.0):
    client_weights = extract_model_weights(client_update)
    global_weights = extract_model_weights(global_model)
    
    update_magnitude = np.linalg.norm(client_weights - global_weights)
    return update_magnitude > threshold

def detect_targeted_model_poisoning(client_update, global_model, threshold=0.7):
    client_weights = extract_model_weights(client_update)
    global_weights = extract_model_weights(global_model)
    
    similarity = 1 - cosine(client_weights, global_weights)
    return similarity < threshold


def is_malicious(client_update, global_model):
    if detect_model_poisoning(client_update, global_model):
        print("Malicious client update detected: Model Poisoning")
        return True
    
    if detect_random_noise_injection(client_update, global_model):
        print("Malicious client update detected: Random Noise Injection")
        return True
    
    if detect_targeted_model_poisoning(client_update, global_model):
        print("Malicious client update detected: Targeted Model Poisoning")
        return True
    return False


