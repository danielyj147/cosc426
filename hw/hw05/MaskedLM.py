import torch
class LSTM_LM(torch.nn.Module):
    def __init__(self, vocabSize, nEmbed, nHidden, nLayers):
        super(LSTM_LM, self).__init__()
        self.vocabSize = vocabSize
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.num_directions = 2 # doubling for bidirectional lstm
        self.embed = torch.nn.Embedding(vocabSize, nEmbed)
        self.lstm = torch.nn.LSTM(nEmbed, nHidden, nLayers, batch_first=True, bidirectional=True)
        self.decoder = torch.nn.Linear(nHidden*self.num_directions, vocabSize) # since LSTM is now bidirectional, the input to this layer doubles.

    def forward(self, X, hidden, cell):
        embedded = self.embed(X.long())
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        y_pred = self.decoder(output)
        return y_pred, hidden, cell

    def init_hidden(self, batchSize):
        device = next(self.parameters()).device
        hidden = cell = torch.zeros(
            (self.nLayers * self.num_directions, batchSize, self.nHidden),
            dtype=torch.float, # matching size.  
            device=device,
        )
        return hidden, cell

    def loss(self, y_pred, y_target):
        loss_fn = torch.nn.CrossEntropyLoss()  ## takes logits not probs
        return loss_fn(y_pred, y_target)