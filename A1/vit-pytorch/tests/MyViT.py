import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # REL
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Args:
        dim (int): Dimension of the input features.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout rate.
    Returns:
        out (Tensor): Output tensor after applying attention.
    """
    def __init__(self, dim, heads= 8, dim_head = 64, dropout = 0 ):
        super ().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim * dim_head #多头拼接后的总维度
        self.scale = dim_head ** -0.5 ## 1/sqrt(dim_head) 稳定点积范围

        #预归一化
        self.norm = nn.LayerNorm(dim)

        #一次线性投影出 Q, K, V    (b, n, dim) -> (b, n, inner_dim * 3)
        # b = batch size, n = token number, dim = feature dimension
        # token number = 1(CLS) + num_patches
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        #注意力概率
        self.attend = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        #输出投影 把(h * d) 拼好的向量打回 dim
        #如果只有1个头并且dim_head == dim,可以直接跳过投影
        project_out = not (heads == 1 and dim_head == dim) #这种情况dim_head = dim，不需要投影
        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.
        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, dim).
        Returns:
            out (Tensor): Output tensor after applying attention.
        """
        #预归一化
        x = self.norm(x)

        #线性投影出 Q, K, V, 并reshape成多头格式
        # qkv: 3个张量，每个是 (b, n, inner_dim) -> (b, h, n, d)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        #点积注意力: (b, h, n, d) @ (b, h, d, n) -> (b, h, n, n)
        dots = q @ k.transpose(-1, -2) * self.scale

        # Softmax 得到注意力权重
        attn = self.attend(dots)
        attn = self.attn_dropout(attn)  # 应用 dropout

        #权重加权 V: (b, h, n, n) @ (b, h, n, d) -> (b, h, n, d)
        out = attn @ v

        #合并多头: (b, h, n, d) -> (b, n, h * d)
        out = rearrange(out, 'b h n d -> b n (h d)')

        #输出线性投影回 dim
        return self.to_out(out)

class Transformer(nn.Module):
    """
    Transformer block consisting of multi-head self-attention and feed-forward layers.
    Args:
        dim (int): Dimension of the input features.
        depth (int): Number of transformer layers.
        heads (int): Number of attention heads.
        dim_head (int): Dimension of each attention head.
        mlp_dim (int): Dimension of the feed-forward layer.
        dropout (float): Dropout rate.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        """
        Forward pass through the transformer block.
        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, dim).
        Returns:
            x (Tensor): Output tensor after applying transformer layers.
        """
        #残差连接
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout =0., pool = 'cls'):
        super().__init__()
        #图像大小
        image_height, image_width = pair(image_size)
        #单patch大小
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image dimensions must be divisible by the patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        #embedding patches
        x = self.to_patch_embedding(img)
        b,n,_ = x.shape # batch size, token number, feature dimension

        #添加cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) #复制cls token到batch size
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)] #添加位置编码
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 池化操作 取平均值或者取CLS token

        x = self.to_latent(x)
        return self.mlp_head(x)  # 输出分类结果


