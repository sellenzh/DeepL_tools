'''
Encoder files.
:param d_model: the features' dimention.
:param num_heads: yhe number of the Multiheadattention.
:param dff: the hidden untis' dimension.
:param rate: the rate of dropout.
:param d_input: choose to alternative multihead attention dimensions.
'''
from torch import nn

from FFN import FFN
from Multi_Head_Attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.3, d_input=None):
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, d_input)
        self.ffn = FFN(d_model, dff)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout = nn.Dropout(rate)
    def forward(self, x, y=None):
        y = x if y is None else y
        att_output = self.dropout(self.mha(x, y, y))
        output1 = self.layernorm1(att_output + x)
        ffn_output = self.dropout(self.ffn(output1))
        return self.layernorm2(output1 + ffn_output)
