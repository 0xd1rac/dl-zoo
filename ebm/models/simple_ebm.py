import torch 
import torch.nn as nn
from torch import Tensor

def predict(energies: Tensor) -> Tensor:
    """
    Convert energy values into class probabilities using softmax over the negatives of the energies.
    
    :param energies: Tensor of shape [B, num_classes].
    :return: Tensor of shape [B, num_classes] representing class probabilities.
    """
    return torch.softmax(-energies, dim=1)

class SimpleEBM(nn.Module):
    """
    Energy-Based Model for classification.
    
    The model outputs energy values for each class. Lower energy indicates a higher likelihood
    for that class. Different energy functions are available by setting the `energy_function` parameter.
    
    Available energy functions:
      - 'cnn': A CNN-based energy function.
      - 'mlp': An MLP on flattened input.
      - 'linear': A logistic regression style (i.e. single linear layer).
      - 'quadratic': A diagonal quadratic energy function, i.e.,
            E(x, y) = sum_i (w[y, i] * x_i^2) + b[y]
      - 'boltzmann': A Boltzmann Machine–inspired energy function where:
            E(x, y) = - x_flat dot bias_visible[y] - sum_j log(1 + exp(x_flat @ W[y,:,j] + bias_hidden[y,j]))
    """
    def __init__(self, energy_function: str = 'cnn', num_classes:int =10):
        super().__init__()
        self.energy_function = energy_function.lower()
        self.num_classes = num_classes

        if self.energy_function == 'cnn':
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
            self.fc = nn.Linear(64 * 7 * 7, self.num_classes)
        
        elif self.energy_function == 'mlp':
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_classes)
            )

        elif self.energy_function == 'linear':
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, self.num_classes)
            )

        elif self.energy_function == "quadractic":
            # Diagonal quadratic energy function:
            # E(x, y) = sum_i (w[y, i] * (x_i)^2) + b[y]
            self.input_dim = 28 * 28 
            self.quadratic_weights = nn.Parameter(torch.randn(self.num_classes, self.input_dim))
            self.quadratic_bias = nn.Parameter(torch.zeros(self.num_classes))

        elif self.energy_function == 'boltzmann':
            # Boltzmann Machine–inspired energy function:
            # E(x, y) = - x_flat dot bias_visible[y] - sum_j log(1 + exp(x_flat @ W[y,:,j] + bias_hidden[y,j]))
            self.input_dim = 28 * 28
            self.hidden_units = 50  
            self.W = nn.Parameter(torch.randn(self.num_classes, self.input_dim, self.hidden_units) * 0.01)
            self.bias_visible = nn.Parameter(torch.zeros(self.num_classes, self.input_dim))
            self.bias_hidden = nn.Parameter(torch.zeros(self.num_classes, self.hidden_units))
            
        else:
            raise ValueError("Unknown energy function type. Choose from 'cnn', 'mlp', 'linear', 'quadratic', or 'boltzmann'.")


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass that computes energy values for each class.
        
        :param x: Input tensor of shape [B, 1, 28, 28].
        :return: Tensor of shape [B, num_classes] with energy values.
        """ 
        if self.energy_function == "cnn":
            features = self.features(x)
            energies = self.fc(features)

        elif self.energy_function in ["mlp", "linear"]:
            energies = self.fc(x)

        elif self.energy_function == "quadratic":
            x_flat = x.view(x.shape[0], -1)
            energies = torch.matmul(x_flat**2, self.quadratic_weights.t()) + self.quadratic_bias

        elif self.energy_function == 'boltzmann':
            x_flat = x.view(x.shape[0], -1)  # Flatten input: [B, 784]
            energies_list = []
            for y in range(self.num_classes):
                # Visible bias term.
                linear_term = - torch.sum(x_flat * self.bias_visible[y], dim=1)
                # Compute pre-activation for hidden units.
                pre_activation = x_flat @ self.W[y] + self.bias_hidden[y]
                # Hidden term: - sum_j log(1 + exp(pre_activation))
                hidden_term = - torch.sum(torch.log1p(torch.exp(pre_activation)), dim=1)
                energies_list.append((linear_term + hidden_term).unsqueeze(1))
            energies = torch.cat(energies_list, dim=1)

        return energies