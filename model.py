import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

class SimpleDQNNetwork(nn.Module):
    def __init__(self,  in_channels, num_actions):
        super().__init__()
        
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_features =  self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, num_actions)
        
        for module in self.cnn.modules():  # off BN
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        
    def forward(self, x):
        return self.cnn(x)


if __name__ == "__main__":
    from env import PathPlanningEnv
    model = SimpleDQNNetwork(1, 4)

    
    
    env = PathPlanningEnv()
    state, reward, done = env.reset()
    
    output = model(torch.tensor(state, dtype=torch.float).reshape(1, 1, 10, 10))
    # output = model(torch.tensor(state, dtype=torch.float).reshape(1, 1, 10, 10).repeat(4, 1, 1, 1))
    
    print(output)