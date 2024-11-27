import torch
from core.NeuralNetwork import NeuralNetwork
from utilities.utils import preprocess_image, visualize_image, predict_digit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "images/7-2.png"
image_tensor = preprocess_image(image_path)
visualize_image(image_tensor)

# Load model and predict
model = NeuralNetwork()
model.load_state_dict(torch.load("federated_learning.pth", map_location=device))
model.to(device)

predicted_digit = predict_digit(image_tensor, model, device)
print(f"Predicted Digit: {predicted_digit}")
