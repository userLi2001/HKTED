import torch

import torch.nn.functional as F
from Modules.diffusion import Diffusion, BERT
from Modules.embedding import UserEmbedding, ItemEmbedding, PositionalEmbedding
from Modules.gat import GAT


class MyModel(torch.nn.Module):
    def __init__(self, args, dataset):
        super().__init__()

        self.device = args.device

        self.hidden_size = args.hidden_size
        self.user_embedding = UserEmbedding(dataset["user_num"], self.hidden_size).to(
            args.device
        )
        self.item_embedding = ItemEmbedding(
            dataset["item_num"] + 1, self.hidden_size
        ).to(args.device)

        self.dropout = torch.nn.Dropout(args.dropout)

        self.train_graph = torch.from_numpy(dataset["train_graph"]).to(args.device)
        self.train_graph = self.train_graph / torch.linalg.norm(
            self.train_graph, ord="fro"
        )

        self.co_user_num = dataset["co_user_num"]

        self.gat = GAT(
            self.hidden_size,
            self.hidden_size,
            self.hidden_size,
            args.dropout,
            args.mult_heads,
        ).to(args.device)

        self.diffusion = Diffusion(args, dataset["user_num"])

        self.linear = torch.nn.Linear(3 * args.hidden_size, 1).to(args.device)

        self.pos_embedding = PositionalEmbedding(args.max_length, self.hidden_size).to(
            args.device
        )
        self.bert = BERT(args).to(args.device)

        self.mlp_dim = 56

    def forward(self, train_seq, item_id, train_guide, stage):
        co_user_embedding = self.dropout(self.user_embedding.weight[: self.co_user_num, :])
        item_embedding = self.item_embedding(item_id).squeeze(1)
        if stage == "train_src":
            seq_embedding = self.dropout(self.item_embedding(train_seq) + self.pos_embedding(train_seq))
            attention_mask = (
                (train_seq > 0).unsqueeze(1).repeat(1, train_seq.size(1), 1).unsqueeze(1)
            )
            seq_feature = self.bert.encoder(seq_embedding, attention_mask)
            seq_feature = torch.mean(seq_feature, dim=1)   # size_shape ([1024,64])
        elif stage == "train_tgt":
            seq_feature = self.diffusion.sampler(train_seq, train_guide)
            seq_feature = torch.mean(seq_feature, dim=1)
            # size_shape ([1024,64])
        else:
            seq_embedding = self.dropout(self.item_embedding(train_seq) + self.pos_embedding(train_seq))
            attention_mask = (
                (train_seq > 0).unsqueeze(1).repeat(1, train_seq.size(1), 1).unsqueeze(1)
            )
            seq_feature = self.bert.encoder(seq_embedding, attention_mask)
            seq_feature = torch.mean(seq_feature, dim=1)
        with torch.no_grad():
             # 修改后的
             raw_features = self.train_graph  #  假设 raw_features 是从数据集中提取的节点特征矩阵,raw_features shape:[56,56]
             weighted_features = self.fiter_modules(raw_features)
             node_weights = torch.norm(weighted_features, p=2, dim=1)
             weight_matrix = torch.diag(node_weights)
             weight_graph = self.train_graph * weight_matrix
             weight_graph = self.train_graph * node_weights.unsqueeze(1)
             graph_feature =self.gat(co_user_embedding, weight_graph)

        # 下面两行
        graph = self.train_graph
        # graph_shape ([56,56])
        graph_feature = self.gat(co_user_embedding, graph)
        # 上面两行
        graph_feature = (
            torch.mean(graph_feature, dim=0)
            .unsqueeze(0)
            .repeat(seq_feature.shape[0], 1)
        ).to(self.device)
        # size_shape ([1024,64])
        feature = torch.cat([graph_feature, seq_feature, item_embedding], dim=1)

        feature = self.linear(feature).squeeze(1)
        return feature
    
    # 过滤部分代码的定义
    def fiter_modules(self, raw_features, theta_value=0.1):
        mlp = torch.nn.Sequential(
            torch.nn.Linear(self.mlp_dim, self.mlp_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ).to(self.device)
        theta = torch.nn.Parameter(torch.tensor(theta_value)).to(self.device).type(torch.float)

        # 确保raw_features在正确的设备上，并且数据类型为torch.float
        raw_features = raw_features.to(self.device).type(torch.float)

        h = mlp(raw_features)

        tau = torch.sign(h) * torch.clamp(torch.abs(h) - theta, min=0)
        selected_features =tau * raw_features

        return selected_features
