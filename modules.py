import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from hyperparameters import Hyperparameters as hp
# from pdb import set_trace as bp
# from modules_ode import *

class Attention(torch.nn.Module):
    """
    Dot-product attention module.

    Args:
      inputs: A `Tensor` with embeddings in the last dimension.
      mask: A `Tensor`. Dimensions are the same as inputs but without the embedding dimension.
        Values are 0 for 0-padding in the input and 1 elsewhere.
    Returns:
      outputs: The input `Tensor` whose embeddings in the last dimension have undergone a weighted average.
        The second-last dimension of the `Tensor` is removed.
      attention_weights: weights given to each embedding.
    """
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        self.context = nn.Parameter(torch.Tensor(embedding_dim)) # context vector
        self.linear_hidden = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.context)

    def forward(self, inputs, mask):
        # Hidden representation of embeddings (no change in dimensions)
        hidden = torch.tanh(self.linear_hidden(inputs))
        # Compute weight of each embedding
        importance = torch.sum(hidden * self.context, dim=-1)
        importance = importance.masked_fill(mask == 0, -1e9)
        # Softmax so that weights sum up to one
        attention_weights = F.softmax(importance, dim=-1)
        # Weighted sum of embeddings
        weighted_projection = inputs * torch.unsqueeze(attention_weights, dim=-1)
        # Output
        outputs = torch.sum(weighted_projection, dim=-2)
        return outputs, attention_weights


# class Attention(torch.nn.Module):
#
#     def __init__(self, embedding_dim):
#         super().__init__()
#         """
#         Define the linear layer `self.a_att` for alpha-attention using `nn.Linear()`;
#
#         Arguments:
#             embedding_dim: the embedding dimension
#         """
#
#         self.attn = nn.Linear(embedding_dim, embedding_dim)
#
#     def forward(self, input, mask):
#         """
#         TODO: Implement the alpha attention.
#
#         Arguments:
#             g: the output tensor from RNN-alpha of shape (batch_size, # visits, embedding_dim)
#             rev_masks: the padding masks in reversed time of shape (batch_size, # visits, # diagnosis codes)
#
#         Outputs:
#             alpha: the corresponding attention weights of shape (batch_size, # visits, 1)
#
#         HINT:
#             1. Calculate the attention score using `self.a_att`
#             2. Mask out the padded visits in the attention score with -1e9.
#             3. Perform softmax on the attention score to get the attention value.
#         """
#
#         # your code here
#         att_score = self.attn(input)
#         rev_masks1 = torch.sum(mask,dim=2)
#         rev_masks1 = torch.unsqueeze(rev_masks1,dim=2)
#         rev_masks1 = rev_masks1 > 0
#         att_score[~rev_masks1] = -1e9
#         fcn1 = nn.Softmax(dim=-1)
#         attn_weights = fcn1(att_score)
#         out = attn_weights * g
#         out = torch.sum(out,dim=1)
#         return out,attn_weights

# Attention with Concatenated Time model
class Attention_ConcatTime(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
        super(Attention_ConcatTime, self).__init__()

        # Embedding dimensions
        self.embed_dp_dim = int(2*np.ceil(num_dp_codes**0.25))
        self.embed_cp_dim = int(2*np.ceil(num_cp_codes**0.25))

        # Embedding layers
        self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
        self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

        # Attention layers
        self.attention_dp = Attention(embedding_dim=self.embed_dp_dim+1) #+1 for the concatenated time
        self.attention_cp = Attention(embedding_dim=self.embed_cp_dim+1)

        # Fully connected output
        self.fc_dp  = nn.Linear(self.embed_dp_dim+1, 1)
        self.fc_cp  = nn.Linear(self.embed_cp_dim+1, 1)
        self.fc_all = nn.Linear(num_static + 2, 1)

        # Others
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
        # Embedding
        ## output dim: batch_size x seq_len x embedding_dim
        embedded_dp = self.embed_dp(dp)
        embedded_cp = self.embed_cp(cp)

        # Concatate with time
        ## output dim: batch_size x seq_len x (embedding_dim+1)
        concat_dp = torch.cat((embedded_dp, torch.unsqueeze(dp_t, dim=-1)), dim=-1)
        concat_cp = torch.cat((embedded_cp, torch.unsqueeze(cp_t, dim=-1)), dim=-1)
        ## Dropout
        concat_dp = self.dropout(concat_dp)
        concat_cp = self.dropout(concat_cp)

        # Attention
        ## output dim: batch_size x (embedding_dim+1)
        attended_dp, weights_dp = self.attention_dp(concat_dp, (dp > 0).float())
        attended_cp, weights_cp = self.attention_cp(concat_cp, (cp > 0).float())

        # Scores
        score_dp = self.fc_dp(self.dropout(attended_dp))
        score_cp = self.fc_cp(self.dropout(attended_cp))

        # Concatenate to variable collection
        all = torch.cat((stat, score_dp, score_cp), dim=1)

        # Final linear projection
        out = self.fc_all(self.dropout(all)).squeeze()

        return out, []



# GRU
def abs_time_to_delta(times):
    delta = torch.cat((torch.unsqueeze(times[:, 0], dim=-1), times[:, 1:] - times[:, :-1]), dim=1)
    delta = torch.clamp(delta, min=0)
    return delta

class RNN_ConcatTime(nn.Module):
    def __init__(self, num_static, num_dp_codes, num_cp_codes):
        super(RNN_ConcatTime, self).__init__()

        # Embedding dimensions
        self.embed_dp_dim = int(np.ceil(num_dp_codes**0.25))
        self.embed_cp_dim = int(np.ceil(num_cp_codes**0.25))

        # Embedding layers
        self.embed_dp = nn.Embedding(num_embeddings=num_dp_codes, embedding_dim=self.embed_dp_dim, padding_idx=0)
        self.embed_cp = nn.Embedding(num_embeddings=num_cp_codes, embedding_dim=self.embed_cp_dim, padding_idx=0)

        # GRU layers
        self.gru_dp_fw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
        self.gru_cp_fw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)
        self.gru_dp_bw = nn.GRU(input_size=self.embed_dp_dim+1, hidden_size=self.embed_dp_dim+1, num_layers=1, batch_first=True)
        self.gru_cp_bw = nn.GRU(input_size=self.embed_cp_dim+1, hidden_size=self.embed_cp_dim+1, num_layers=1, batch_first=True)

        # Fully connected output
        self.fc_dp  = nn.Linear(2*(self.embed_dp_dim+1), 1)
        self.fc_cp  = nn.Linear(2*(self.embed_cp_dim+1), 1)
        self.fc_all = nn.Linear(num_static + 2, 1)

        # Others
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, stat, dp, cp, dp_t, cp_t):
        # Compute time delta
        ## output dim: batch_size x seq_len
        dp_t_delta_fw = abs_time_to_delta(dp_t)
        cp_t_delta_fw = abs_time_to_delta(cp_t)
        dp_t_delta_bw = abs_time_to_delta(torch.flip(dp_t, [1]))
        cp_t_delta_bw = abs_time_to_delta(torch.flip(cp_t, [1]))

        # Embedding
        ## output dim: batch_size x seq_len x embedding_dim
        embedded_dp_fw = self.embed_dp(dp)
        embedded_cp_fw = self.embed_cp(cp)
        embedded_dp_bw = torch.flip(embedded_dp_fw, [1])
        embedded_cp_bw = torch.flip(embedded_cp_fw, [1])

        # Concatate with time
        ## output dim: batch_size x seq_len x (embedding_dim+1)
        concat_dp_fw = torch.cat((embedded_dp_fw, torch.unsqueeze(dp_t_delta_fw, dim=-1)), dim=-1)
        concat_cp_fw = torch.cat((embedded_cp_fw, torch.unsqueeze(cp_t_delta_fw, dim=-1)), dim=-1)
        concat_dp_bw = torch.cat((embedded_dp_bw, torch.unsqueeze(dp_t_delta_bw, dim=-1)), dim=-1)
        concat_cp_bw = torch.cat((embedded_cp_bw, torch.unsqueeze(cp_t_delta_bw, dim=-1)), dim=-1)
        ## Dropout
        concat_dp_fw = self.dropout(concat_dp_fw)
        concat_cp_fw = self.dropout(concat_cp_fw)
        concat_dp_bw = self.dropout(concat_dp_bw)
        concat_cp_bw = self.dropout(concat_cp_bw)

        # GRU
        ## output dim rnn:        batch_size x seq_len x (embedding_dim+1)
        ## output dim rnn_hidden: batch_size x 1 x (embedding_dim+1)
        rnn_dp_fw, rnn_hidden_dp_fw = self.gru_dp_fw(concat_dp_fw)
        rnn_cp_fw, rnn_hidden_cp_fw = self.gru_cp_fw(concat_cp_fw)
        rnn_dp_bw, rnn_hidden_dp_bw = self.gru_dp_bw(concat_dp_bw)
        rnn_cp_bw, rnn_hidden_cp_bw = self.gru_cp_bw(concat_cp_bw)
        ## output dim rnn_hidden: batch_size x (embedding_dim+1)
        rnn_hidden_dp_fw = rnn_hidden_dp_fw.view(-1, self.embed_dp_dim+1)
        rnn_hidden_cp_fw = rnn_hidden_cp_fw.view(-1, self.embed_cp_dim+1)
        rnn_hidden_dp_bw = rnn_hidden_dp_bw.view(-1, self.embed_dp_dim+1)
        rnn_hidden_cp_bw = rnn_hidden_cp_bw.view(-1, self.embed_cp_dim+1)
        ## concatenate forward and backward: batch_size x 2*(embedding_dim+1)
        rnn_hidden_dp = torch.cat((rnn_hidden_dp_fw, rnn_hidden_dp_bw), dim=-1)
        rnn_hidden_cp = torch.cat((rnn_hidden_cp_fw, rnn_hidden_cp_bw), dim=-1)

        # Scores
        score_dp = self.fc_dp(self.dropout(rnn_hidden_dp))
        score_cp = self.fc_cp(self.dropout(rnn_hidden_cp))

        # Concatenate to variable collection
        all = torch.cat((stat, score_dp, score_cp), dim=1)

        # Final linear projection
        out = self.fc_all(self.dropout(all)).squeeze()

        return out, []
