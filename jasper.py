from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


def create_sinusoidal_embeddings(
    pos,
    dim,
    min_period=4e-3,
    max_period=4.0,
    dtype=torch.float32,
):
    assert dim % 2 == 0

    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=pos.device)
    period = min_period * (max_period / min_period) ** fraction
    scale = 2 * torch.pi / period
    theta = pos[:, None] * scale[None, :]
    emb = torch.cat([theta.cos(), theta.sin()], dim=1)
    return emb.to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    x = x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]
    return x


class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()

        scale = base ** (-torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x):
        freqs = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)[:, None] * self.scale[None, :]
        freqs = torch.cat([freqs, freqs], dim=-1)
        return freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim, attn_dropout):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout
        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim)

    def forward(self, x, pos_emb):
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b t (n nh hd) -> b (n nh) t hd", nh=self.num_heads, hd=self.head_dim)
        q, k, v = qkv.chunk(3, dim=1)

        cos, sin = pos_emb
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        x = rearrange(x, "b nh t hd -> b t (nh hd)")
        x = self.o_proj(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, num_heads, head_dim, attn_dropout):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout
        self.q_proj = nn.Linear(q_dim, num_heads * head_dim)
        self.kv_proj = nn.Linear(kv_dim, 2 * num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, q_dim)

    def forward(self, x, cond):
        q = self.q_proj(x)
        kv = self.kv_proj(cond)
        q = rearrange(q, "b t (nh hd) -> b nh t hd", nh=self.num_heads)
        kv = rearrange(kv, "b t (n nh hd) -> b (n nh) t hd", nh=self.num_heads, hd=self.head_dim)
        k, v = kv.chunk(2, dim=1)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        x = rearrange(x, "b nh t hd -> b t (nh hd)")
        x = self.o_proj(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout):
        super().__init__()

        self.w1w3 = nn.Linear(hidden_dim, ff_dim * 2)
        self.w2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1, x3 = self.w1w3(x).chunk(2, dim=-1)
        return self.w2(self.dropout(F.silu(x1) * x3))


class JasperDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_dim, ff_dim, dropout, attn_dropout):
        super().__init__()

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, num_heads, head_dim, attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2_1 = nn.RMSNorm(hidden_dim)
        self.norm2_2 = nn.RMSNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, hidden_dim, num_heads, head_dim, attn_dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.RMSNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ff_dim, dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.modulation = nn.Parameter(torch.zeros(1, 11, hidden_dim))

    def forward(self, x, temb, cond, pos_emb):
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift1_cross_attn,
            scale1_cross_attn,
            shift2_cross_attn,
            scale2_cross_attn,
            gate_cross_attn,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = (temb + self.modulation).chunk(11, dim=1)
        x = x + gate_attn * self.dropout1(self.attn(self.norm1(x) * (1.0 + scale_attn) + shift_attn, pos_emb))
        x = x + gate_cross_attn * self.dropout2(
            self.cross_attn(
                self.norm2_1(x) * (1.0 + scale1_cross_attn) + shift1_cross_attn,
                self.norm2_2(cond) * (1.0 + scale2_cross_attn) + shift2_cross_attn,
            )
        )
        x = x + gate_ffn * self.dropout3(self.ffn(self.norm3(x) * (1.0 + scale_ffn) + shift_ffn))
        return x


class JasperVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder, _ = torch.hub.load("facebookresearch/vjepa2", config.vjepa2_model)
        self.norm = nn.RMSNorm(self.encoder.blocks[-1].mlp.fc2.out_features)
        self.o_proj = nn.Linear(self.encoder.blocks[-1].mlp.fc2.out_features, config.hidden_dim)

        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.o_proj(self.norm(x))
        return x


class JasperActionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.rotary_emb = RotaryEmbedding1D(config.head_dim)

        self.layers = nn.ModuleList(
            [
                JasperDecoderLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    head_dim=config.head_dim,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                )
                for _ in range(config.decoder_num_layers)
            ]
        )
        self.action_mlp = nn.Linear(config.action_dim, config.hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim * 11),
        )
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.o_proj = nn.Linear(config.hidden_dim, config.action_dim)

        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, noise, t, cond):
        x = self.action_mlp(noise)
        temb = create_sinusoidal_embeddings(t, self.hidden_dim, dtype=x.dtype)
        temb = self.time_mlp(temb)
        temb = rearrange(temb, "b (n d) -> b n d", n=11)

        pos_emb = self.rotary_emb(x)

        for layer in self.layers:
            x = layer(x, temb, cond, pos_emb=pos_emb)

        x = self.o_proj(self.norm(x))
        return x


@dataclass
class JasperConfig:
    dtype: torch.dtype = torch.float32
    device: str = "cuda"
    action_dim: int = 6
    action_horizon: int = 30
    hidden_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    ff_dim: int = 3200
    attn_dropout: float = 0.1
    dropout: float = 0.1
    decoder_num_layers: int = 4
    vjepa2_model: str = "vjepa2_1_vit_base_384"
    # vjepa2_model: str = "vjepa2_1_vit_gigantic_384"


class Jasper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        self.vision_encoder = JasperVisionEncoder(config)
        self.action_decoder = JasperActionDecoder(config)

    @torch.inference_mode()
    def sample_action(self, images, num_steps):
        cond = self.vision_encoder(images)

        dt = 1 / num_steps
        x_t = torch.randn(images.shape[0], self.action_horizon, self.action_dim, dtype=cond.dtype, device=cond.device)
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=cond.dtype, device=cond.device)[:-1]
        timesteps = repeat(timesteps, "n -> n b", b=images.shape[0])

        for t in timesteps:
            v = self.action_decoder(x_t, t, cond)
            x_t = x_t + v * dt

        return x_t

    def forward(self, images, action):
        cond = self.vision_encoder(images)

        noise = torch.randn_like(action)
        t = torch.rand(images.shape[0], dtype=cond.dtype, device=cond.device)
        x_t = action * t[:, None, None] + noise * (1 - t)[:, None, None]
        v = self.action_decoder(x_t, t, cond)

        loss = F.mse_loss(v, action - noise)

        return loss


if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float32

    config = JasperConfig(device=device, dtype=dtype)

    images = torch.randn(4, 3, 16, 256, 256, device=config.device, dtype=config.dtype)
    action = torch.randn(4, config.action_horizon, config.action_dim, device=config.device, dtype=config.dtype)

    model = Jasper(config).to(config.dtype).to(config.device)

    loss = model(images, action)
    pred_action = model.sample_action(images, num_steps=10)

    print(loss)
    print(pred_action.shape)
