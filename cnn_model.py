import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 32,
        kernel_size: int = 4,
        dense_layer: int = 128,
        img_rows: int = 28,
        img_cols: int = 28,
        maxpool: int = 2,
        embedding_dim: int = 50,
        target_dim: int = 10
    ):
        """
        Basic Architecture of CNN

        Attributes:
            num_filters: Number of filters, out channel for 1st and 2nd conv layers,
            kernel_size: Kernel size of convolution,
            dense_layer: Dense layer units,
            img_rows: Height of input image,
            img_cols: Width of input image,
            maxpool: Max pooling size
            embedding_dim: the dimension of embedding layer
            target_dim: the dimension of target (i.e., the number of classes in the task)
        """
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3=nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            num_filters
            * ((img_rows - 2 * kernel_size + 2) // 2)
            * ((img_cols - 2 * kernel_size + 2) // 2),
            dense_layer
        )
        self.embedding=nn.Linear(dense_layer,embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, target_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        """
        The embedding layer is just before the output layer. The embedding saved in the neural network during every propagation (soemwhat inefficient, 
        better methods is welcomed)
        """
        x = self.embedding(x)
        e = x.clone().detach()
        self.e=e
        x=F.relu(x)
        x = self.dropout3(x)
        out = self.fc2(x)
        
        return out



