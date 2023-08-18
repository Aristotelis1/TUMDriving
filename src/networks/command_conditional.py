import torch
from torch import nn

NVIDIA_IMAGE_SIZE = (200, 66)

class CommandModule(nn.Module):
  # for concatenated conditional architeure (non-branched)
    def __init__(self, num_commands, dropout):
        super(CommandModule, self).__init__()
        self.fc1 = nn.Linear(in_features=num_commands, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.act(self.drop(self.fc1(x)))
        x = self.act(self.drop(self.fc2(x)))
        return x


class CommandNetwork(nn.Module):
    """Simple conditional network with branching based on input command."""

    def __init__(self, perception_model=None, num_commands=3, dropout=0.5):
        super(CommandNetwork, self).__init__()

        self.perception = perception_model
        self.commander = CommandModule(num_commands=num_commands, dropout=dropout) # for concatenated conditional architecture

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 18 + 128, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, img, command):
        latent_img = self.perception(img)
        latent_command = self.commander(command).unsqueeze(0)
        #print(latent_img.shape, latent_command.shape)
        stacked = torch.cat((latent_img, latent_command), 1)
        output = self.fc(stacked)
        return output

class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper (added batchnorm)."""

    def __init__(self):
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.Flatten(),
            nn.Dropout(0.5)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=64 * 18, out_features=100),
        #     nn.ELU(),
        #     nn.Linear(in_features=100, out_features=50),
        #     nn.ELU(),
        #     nn.Linear(in_features=50, out_features=10),
        #     nn.Linear(in_features=10, out_features=1)
        # )

    def forward(self, input):
        #input = input.view(input.size(0), 3, 66, 200)
        output = self.conv_layers(input)
        # print(output.shape)
        #output = self.fc(output)
        return output