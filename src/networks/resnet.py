from torch import nn
from torchvision import models

class TunedResnet50(nn.Module):
    """
    * @brief Initializes the class variables
    * @param None.
    * @return None.
    """
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(weights="IMAGENET1K_V2")
        for i, param in enumerate(self.resnet50.parameters()):
          if i <= 45:
            param.requires_grad = False
        self.resnet50.fc = nn.Sequential(
            nn.Linear(2048,512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
    """
    * @brief Function to build the model.
    * @param The image to train.
    * @return The trained prediction network.
    """
    def forward(self, input):
        input = self.resnet50(input)
        return input

if __name__ == '__main__':
    model = TunedResnet50()
    print(model)