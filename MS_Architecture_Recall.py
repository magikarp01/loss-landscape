# Architecture for Maze Solving models

import torch
from torch import nn
from torch.nn.functional import relu
import MS_Architecture

in_channels = 128
out_channels = 128

# iteration block of 2 residual blocks
class IterationBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel_size = 3 according to paper, padding = 1 to maintain size
        # next, can try bias=True
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=(3,3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=(3,3), padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=(3,3), padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=(3,3), padding=1, bias=False)
    
    def forward(self, x):
        # regular relu of convolution output
        l1_out = relu(self.conv1(x))
        # add x for shortcut
        l2_out = relu(self.conv2(l1_out) + x)
        
        l3_out = relu(self.conv3(l2_out))
        l4_out = relu(self.conv4(l3_out) + l2_out)

        return l4_out

def make_iter_block():
    return IterationBlock()

class MazeSolvingNN_Recall(nn.Module):
    def __init__(self, num_iter):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = out_channels, 
            kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU()
        )

        self.iter_block = make_iter_block()
        self.num_iter = num_iter

        # in_channels should be 120, out_channels = 60
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels = out_channels, out_channels = 32, 
            kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU()
        )

        # in_channels 60, out_channels 30
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 8, 
            kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU()
        )

        self.l4 = nn.Conv2d(in_channels = 8, out_channels = 2, 
        kernel_size=(3,3), padding=1, bias=False)

        self.end_layers = nn.Sequential(self.l2, self.l3, self.l4)

        self.recall_conv = nn.Conv2d(in_channels = out_channels+3, out_channels = out_channels, 
        kernel_size=(3,3), padding=1, bias=False)

    # method for copying existing model but expanding iterations
    def change_iter(self, new_iter):
        self.num_iter = new_iter

    def expand_iterations(self, init_nn):
        with torch.no_grad():
            init_state_dict = init_nn.state_dict()
            self.l1[0].weight.copy_(init_state_dict['l1.0.weight'].clone())
            self.l2[0].weight.copy_(init_state_dict['l2.0.weight'].clone())
            self.l3[0].weight.copy_(init_state_dict['l3.0.weight'].clone())
            self.l4.weight.copy_(init_state_dict['l4.weight'].clone())
            self.recall_conv.weight.copy_(init_state_dict['recall_conv.weight'].clone())

            self.iter_block.conv1.weight.copy_(init_state_dict['iter_block.conv1.weight'].clone())
            self.iter_block.conv2.weight.copy_(init_state_dict['iter_block.conv2.weight'].clone())
            self.iter_block.conv3.weight.copy_(init_state_dict['iter_block.conv3.weight'].clone())
            self.iter_block.conv4.weight.copy_(init_state_dict['iter_block.conv4.weight'].clone())

    def forward(self, x):
        prev_output = self.l1(x)
        for i in range(self.num_iter):
            # concatenate previous output with input
            recall_input = torch.cat((prev_output, x), dim=1)
            next_input = self.recall_conv(recall_input)
            prev_output = self.iter_block(next_input)
        return self.end_layers(prev_output)
    
    
