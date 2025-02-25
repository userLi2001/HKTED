import math

import torch
from torch.nn import functional as F

from .embedding import ItemEmbedding, PositionalEmbedding, TimestepEmbedding
from Modules.unet import UNet


class Diffusion(torch.nn.Module):
    def __init__(self, args, item_count):
        super().__init__()

        self.device = args.device
        self.max_length = args.max_length
        self.hidden_size = args.hidden_size
        self.max_step = args.max_step  # 扩散步骤
        self.dropout = args.dropout
        self.item_embedding = ItemEmbedding(item_count + 1, self.hidden_size).to(self.device)
        self.position_embedding = PositionalEmbedding(self.max_length, self.hidden_size).to(self.device)
        self.time_embedding = TimestepEmbedding(self.max_step, self.hidden_size).to(self.device)
        self.bert = BERT(args).to(self.device)

        self.dropout = torch.nn.Dropout(self.dropout)
        self.layernorm = torch.nn.LayerNorm([self.max_length, self.hidden_size]).to(self.device)

        self.beta_1 = args.beta_1
        self.beta_T = args.beta_T
        self.T = args.T
        self.model = UNet().to(args.device)

    def forward(self, sequence):  # 输入sequence
        item_embedding = self.item_embedding(sequence)
        position_embedding = self.position_embedding(sequence)
        attention_mask = (
            (sequence > 0).unsqueeze(1).repeat(1, sequence.size(1), 1).unsqueeze(1)
        )
        diffusion_steps = torch.randint(0, self.max_step, size=(sequence.shape[0],)).to(
            self.device
        )
        time_embedding = self.time_embedding(diffusion_steps)
        feature_input = self.dropout(
            self.layernorm(item_embedding + position_embedding)
        )

        with torch.no_grad():
            noise = torch.randn_like(item_embedding)
            alpha = 1 - torch.sqrt((diffusion_steps + 1) / self.max_step).view(-1, 1, 1)
            noisy_item = (
                torch.sqrt(alpha) * feature_input + torch.sqrt(1 - alpha) * noise
            )

        denoise_input = self.dropout(self.layernorm(noisy_item + time_embedding))
        denoise_output = self.bert.encoder(denoise_input, attention_mask)

        return denoise_output
    def sampler(self, sequence, train_guide):
        denoise_output = None
        # 前向加噪过程-----------
        item_embedding = self.item_embedding(sequence)
        position_embedding = self.position_embedding(sequence)
        feature_input = self.dropout(
            self.layernorm(item_embedding + position_embedding)
        )
        diffusion_steps = torch.randint(0, self.max_step, size=(sequence.shape[0],)).to(
            self.device
        )
        # 指导策略
        item_guide_embedding = self.item_embedding(train_guide)
        position_guide_embedding = self.position_embedding(train_guide)
        feature_guide = self.dropout(
            self.layernorm(item_guide_embedding + position_guide_embedding)
        )  # [1024,30,64]
        # -------------------------

        time_embedding = self.time_embedding(diffusion_steps)
        with torch.no_grad():
            noise = torch.randn_like(item_embedding)
            alpha = 1 - torch.sqrt((diffusion_steps + 1) / self.max_step).view(-1, 1, 1)
            noisy_item = (
                torch.sqrt(alpha) * feature_input + torch.sqrt(1 - alpha) * noise
            )

        noisy_item = self.dropout(self.layernorm(noisy_item + time_embedding))

        sampler = GaussianDiffusionSampler(self.model, self.beta_1, self.beta_T, self.T)


        for t in range(self.max_step - 1, 0, -1):
            mean, var = sampler.p_mean_variance(x_t=noisy_item, t=t, feature_guide=feature_guide)
            eps = self.model(noisy_item, t)
            # 计算均值
            xt_prev_mean = sampler.predict_xt_prev_mean_from_eps(x_t=noisy_item, t=t, eps=eps, feature_guide=feature_guide)
            # 更新去噪结果
            noisy_item = mean + torch.sqrt(var.to("cuda")) * eps
            # 最后一步高斯噪声设置为0
            if t > 1:
                noise = torch.randn_like(noisy_item)
            else:
                noise = 0
            denoise_output = xt_prev_mean + torch.sqrt(var.to("cuda")) * noise
        return denoise_output


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    t = torch.tensor([t], dtype=torch.long, device=v.device)
    #device = t.device
    out = torch.gather(v, index=t, dim=0).float()  # 移除了.to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = F.math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device="cuda") * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
# ``GaussianDiffusionSampler``包含了Diffusion Model的后向过程 & 推理过程
class GaussianDiffusionSampler(torch.nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        """
        所有参数含义和``GaussianDiffusionTrainer``（前向过程）一样
        """
        super().__init__()

        self.model = model
        self.T = T

        # 这里获取betas, alphas以及alphas_bar和前向过程一模一样
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 这一步是方便后面运算，相当于构建alphas_bar{t-1}
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]  # 把alpha_bar的第一个数字换成1,按序后移

        # 根据公式(7)(8)，后向过程中的计算均值需要用到的系数用coeff1和coeff2表示
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        # 根据公式(4)，计算后向过程的方差
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        # MLP 创新部分
        self.hidden_size = 64
        self.diffuser = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size)).to("cuda")
        self.step_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
        ).to("cuda")

    def model_forward_uncon(self, x, step, feature_guide):
        t = self.step_mlp(step)
        res = self.diffuser(x * t * feature_guide) # 这里我直接乘了  之前是拼接 拼接的话需要改t的维度 [1024,30,64]
        return res

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps, feature_guide):
        """
        该函数用于反向过程中，条件概率分布q(x_{t-1}|x_t)的均值
        Args:
             x_t: 迭代至当前步骤的图像
             t: 当前步数
             eps: 模型预测的噪声，也就是z_t
        Returns:
            x_{t-1}的均值，mean = coeff1 * x_t + coeff2 * eps
        """
        assert x_t.shape == eps.shape
        x_t = self.model_forward_uncon(x_t, t, feature_guide)
        return (
            extract(self.coeff1.to("cuda"), t, x_t.shape) * x_t -
            extract(self.coeff2.to("cuda"), t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, feature_guide):

        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        # 模型前向预测得到eps(也就是z_t)
        eps = self.model(x_t, t)
        # 计算均值
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps, feature_guide=feature_guide)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        # 反向扩散过程，从x_t迭代至x_0
        x_t = x_T
        for time_step in reversed(range(self.T)):
            # t = [1, 1, ....] * time_step, 长度为batch_size
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # 计算条件概率分布q(x_{t-1}|x_t)的均值和方差
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            # 最后一步的高斯噪声设为0（我认为不设为0问题也不大，就本实例而言，t=0时的方差已经很小了）
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # ``torch.clip(x_0, -1, 1)``,把x_0的值限制在-1到1之间，超出部分截断
        return torch.clip(x_0, -1, 1)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attention = torch.nn.functional.softmax(scores, -1)
        if dropout is not None:
            p_attention = dropout(p_attention)
        return torch.matmul(p_attention, value), p_attention


class MultiHeadedAttention(torch.nn.Module):

    def __init__(self, head, model_dimension, dropout):
        super().__init__()
        assert model_dimension % head == 0
        self.d_k = model_dimension // head
        self.head = head
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(model_dimension, model_dimension) for _ in range(3)]
        )
        self.output_linear = torch.nn.Linear(model_dimension, model_dimension)
        self.attention = Attention()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]
        x, attention = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.output_linear(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args.hidden_size
        attention_heads = args.num_heads
        feed_forward_hidden_size = args.hidden_size * 4
        dropout = args.dropout

        self.attention = MultiHeadedAttention(attention_heads, hidden_size, dropout)
        self.feed_forward = PositionedFeedForward(
            hidden_size, feed_forward_hidden_size, dropout
        )
        self.input_sublayer = SublayerConnection(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_blocks = args.num_blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(args) for _ in range(self.num_blocks)]
        )

    def encoder(self, x, attention_mask=None):
        for transformer in self.transformer_blocks:
            x = transformer(x, attention_mask)
        return x


class PositionedFeedForward(torch.nn.Module):
    def __init__(self, model_dimension, feed_forward_dimension, dropout):
        super(PositionedFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(model_dimension, feed_forward_dimension)
        self.w_2 = torch.nn.Linear(feed_forward_dimension, model_dimension)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(torch.nn.Module):
    @staticmethod
    def forward(x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(torch.nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
