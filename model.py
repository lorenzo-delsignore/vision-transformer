import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split image into patches and embed them"""

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """Run forward pass
        Input
        -----
        x : torch.Tensor
            Shape (n_samples, in_chans, img_size, img_size)
        """
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Attention mechanism."""

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # same dimension as the input when concateneted
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass
        Input
        -----
        x : torch.Tensor
            Shape (n_samples, n_patches + 1, dim)
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
            q @ k_t
        ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads , head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)
        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass
        Input
        -----
        x : torch.Tensor
            Shape (n_samples, n_patches + 1, in_features)
        """
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden features)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=hidden_features, out_features=dim
        )

    def forward(self, x):
        """Run forward pass
        Input
        -----
        x : torch.Tensor
            Shape (n_samples, n_patches + 1, dim)
        """
        x = x + self.attn(self.norm1(x))  # (n_samples, n_patches + 1, dim)
        x = x + self.mlp(self.norm2(x))  # (n_samples, n_patches + 1, dim)

        return x


class VisionTransformer(nn.Module):
    """Vision transformer"""

    def __init__(
        self,
        img_size=32,
        patch_size=16,
        in_chans=3,
        n_classes=10,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Run forward pass
        Input
        ----
        x : torch.Tensor
            Shape (n_samples, in_chans, img_size, img_size)
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x
