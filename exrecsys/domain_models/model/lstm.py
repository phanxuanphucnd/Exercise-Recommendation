import torch 
import torch.nn as nn

from torch.nn.init import constant_, xavier_uniform_
from exrecsys.domain_models.constant import PAD_INDEX

class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        n_concept,
        dropout,
    ):
        super(LSTM, self).__init__()

        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self.n_concept = n_concept
        self._encoder = nn.Embedding(
            num_embeddings=2 * n_concept + 1,
            embedding_dim=input_dim,
            padding_idx=PAD_INDEX
        )
        self._lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self._decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_concept)
        )

    def init_hidden(self, batch_size):
        """
        initialize hidden layer as zero tensor
        batch_size: single integer
        """
        weight = next(self.parameters())

        return (xavier_uniform_(weight.new_zeros(self._num_layers, batch_size, self._hidden_dim)),
                xavier_uniform_(weight.new_zeros(self._num_layers, batch_size, self._hidden_dim)))

    def forward(self, input):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, sequence_size)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1)
        """
        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)
        input = self._encoder(input)
        output, _ = self._lstm(input, (hidden[0].detach(), hidden[1].detach()))
        output = self._decoder(output)
        
        # print(f"\n- output: {output.size()} {output}")
        # output = torch.gather(output, -1, target_id)
        
        return output
