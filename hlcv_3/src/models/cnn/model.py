import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # self.conv1 = nn.Conv2d(self.input_size, self.hidden_layers[0], kernnel_size=3, stride=1, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu1 = self.activation()

        # layers.append(self.conv1)
        # layers.append(self.maxpool1)
        # layers.append(self.relu1)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(self.input_size, self.hidden_layers[0], kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(self.activation())

        for i in range(1, len(self.hidden_layers) - 1):
            self.layers.append(nn.Conv2d(self.hidden_layers[i-1], self.hidden_layers[i], kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.layers.append(self.activation())

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        weight_tensor = self.layers[0].weight.data.cpu().numpy()
        num_filters = weight_tensor.shape[0]

        # Calculate the grid size for plotting (e.g., 8x16 for 128 filters)
        grid_rows = 8
        grid_cols = 16
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols, grid_rows))

        for i in range(grid_rows * grid_cols):
            ax = axs[i // grid_cols, i % grid_cols]
            if i < num_filters:
                # Normalize each filter using the provided _normalize method
                filter_img = weight_tensor[i]
                normalized_img = self._normalize(filter_img)
                ax.imshow(normalized_img.transpose(1, 2, 0))  # Transpose to put channels last
            ax.axis('off')

        plt.show()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        out = None

        for layer in self.layers:
            x = layer(x)

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
