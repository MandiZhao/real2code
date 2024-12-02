from torch import nn

class PointBatchNorm1D(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x):
        return super().forward(x.view(-1, x.shape[-1])).view(x.shape)

def MLP(channels, batch_norm=True):
    layers = list()
    for i in range(1, len(channels)):
        module_layers = [
            nn.Linear(channels[i - 1], channels[i]), 
            nn.ReLU()]
        if batch_norm:
            module_layers.append(
                PointBatchNorm1D(channels[i]))
        module = nn.Sequential(*module_layers)
        layers.append(module)
    return nn.Sequential(*layers)
