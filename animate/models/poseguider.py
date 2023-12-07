import torch
import torch.nn as nn

# PoseGuider is a simple CNN which is used to encode the 
class PoseGuider(nn.Module):
    def __init__(self, latent_channels):
        super(PoseGuider, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.zeroconv = nn.Conv2d(128, latent_channels, 1, 1)

        # initialize conv weights as gaussians
        torch.nn.init.normal_(self.conv1[0].weight)
        torch.nn.init.normal_(self.conv2[0].weight)
        torch.nn.init.normal_(self.conv3[0].weight)
        torch.nn.init.normal_(self.conv4[0].weight)

        # set weights and biases to zero for the zero conv projection
        self.zeroconv.weight.data.fill_(0.00)
        self.zeroconv.bias.data.fill_(0.00)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # apply final zero conv projection layer
        out = self.zeroconv(out)

        return out
