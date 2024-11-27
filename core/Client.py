import torch
import copy
import torch.nn as nn

class Client:
    def __init__(self, model, train_data, device, client_id):
        self.model = copy.deepcopy(model)
        self.train_data = train_data
        self.device = device
        self.client_id = client_id
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_data:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters()

    def get_parameters(self):
        return {key: value.cpu() for key, value in self.model.state_dict().items()}