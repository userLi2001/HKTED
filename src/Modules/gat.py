import torch


class GraphAttentionLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, dropout, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU()

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.nn.functional.dropout(
            attention, self.dropout, training=self.training
        )
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return torch.nn.functional.relu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, concat=False
        )

    def forward(self, x, adj):
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.nn.functional.relu(self.out_att(x, adj))
        return torch.nn.functional.softmax(x, dim=1)
