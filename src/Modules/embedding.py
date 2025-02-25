import torch


class UserEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.ie = torch.nn.Embedding(vocab_size, embed_size)
        self.weight = self.ie.weight

    def forward(self, x):
        return self.ie(x)


class ItemEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.ie = torch.nn.Embedding(vocab_size, embed_size, 0)
        self.weight = self.ie.weight

    def forward(self, x):
        return self.ie(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, embed_size)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TimestepEmbedding(torch.nn.Module):
    def __init__(self, steps, embed_size):
        super().__init__()
        self.te = torch.nn.Embedding(steps, embed_size)
        torch.nn.init.constant_(self.te.weight, 0)

    def forward(self, x):
        return self.te(x).unsqueeze(1)
