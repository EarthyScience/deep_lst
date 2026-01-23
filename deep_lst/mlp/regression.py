import torch
import torch.nn as nn


class MultiModalMLP(nn.Module):

    def __init__(self, input_size, target_size, config):
        super().__init__()

        self.input_size = input_size
        self.target_size = target_size

        self.activation = nn.ReLU() if config['activation'] == 'relu' else None
        self.hidden_sizes = config['hidden_sizes']
        self.dropout = config['dropout']

        self.net = self.dense()

    def forward(self, *inputs):
        
        inputs = [x for x in inputs if x is not None]
        x = torch.cat(inputs, dim=-1) # assumes channel is last dim

        return self.net(x)

    def dense(self):
        layers = nn.Sequential()
        layers.add_module(f'input', nn.Linear(self.input_size, self.hidden_sizes[0]))
        layers.add_module(f'activation0', self.activation)
        layers.add_module(f'dropout0', nn.Dropout(self.dropout))
        if len(self.hidden_sizes) > 1:
            for i in range(len(self.hidden_sizes)-1):
                layers.add_module(f'hidden{i}', nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
                layers.add_module(f'activation{i}', self.activation)
                layers.add_module(f'dropout{i}', nn.Dropout(self.dropout))
        layers.add_module('output', nn.Linear(self.hidden_sizes[-1], self.target_size))
        return layers

