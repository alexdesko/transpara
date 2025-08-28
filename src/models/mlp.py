import torch
import torch.nn as nn

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class SimpleMLP(nn.Module):
    def __init__(self, input_size=224 * 224, hidden_size=512, num_classes=2):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.classifier()
        return x
