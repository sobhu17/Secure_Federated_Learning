import torch
import copy
from utilities.attackutils import is_malicious


class FederatedServer:
    def __init__(self, global_model):
        self.global_model = global_model
        self.update_history = []
    
    def aggregate_parameters(self, client_parameters, security=False):
        if security:
            valid_client_parameters = []
            for client_update in client_parameters:
                if not is_malicious(client_update, self.global_model):
                    valid_client_parameters.append(client_update)
            client_parameters = valid_client_parameters
        averaged_params = {}
        for name in client_parameters[0].keys():
            averaged_params[name] = torch.stack([params[name] for params in client_parameters]).mean(dim=0)
        self.global_model.load_state_dict(averaged_params)
        return copy.deepcopy(self.global_model)