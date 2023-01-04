'''
Feed Forward Net
:param d_model: the number of the feature.
:param hidden_dim: the number of the hidden units dimensions.
'''
from torch import nn

class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FFN, self).__init__()
        self.layer1 = nn.Linear(d_model, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))
