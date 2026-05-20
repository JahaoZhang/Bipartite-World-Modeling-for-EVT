"""Implementation of additional projectors for additional inputs to the VLA models."""
import torch
import torch.nn as nn


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features

class TrajMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=448, out_dim=896):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, K, in_dim)
        return self.net(x)

class PositionMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=448, out_dim=896):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, in_dim)
        return self.net(x)

class NoisyActionProjector(nn.Module):
    """
    [Diffusion] Projects noisy action inputs into the LLM's embedding space.

    Note that since each action is tokenized into 7 tokens in OpenVLA (rather
    than having 1 token per action), each noisy action token will have dimension 1
    instead of 7.
    """
    def __init__(self, llm_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.action_token_dim = 1

        self.fc1 = nn.Linear(self.action_token_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, noisy_actions: torch.Tensor = None) -> torch.Tensor:
        # noisy_actions: (bsz, num_action_tokens=chunk_len*action_dim, 1)
        projected_features = self.fc1(noisy_actions)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


class CrossAttentionDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.action_proj = nn.Linear(action_dim, latent_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.norm_ffn = nn.LayerNorm(latent_dim)

        hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, a_t):
        B, N, D = z_t.shape

        a_token = self.action_proj(a_t)

        kv = torch.cat([z_t, a_token], dim=1)

        q = self.norm_q(z_t)
        k = self.norm_kv(kv)
        v = k

        attn_out, _ = self.attn(q, k, v)
        z = z_t + attn_out

        z = z + self.mlp(self.norm_ffn(z))
        return z   # 只输出 latent

class LatentToFeature(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(latent_dim * mlp_ratio)

        self.norm = nn.LayerNorm(latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z: torch.Tensor):
        """
        z: [B, N, latent_dim]
        return: [B, N, feature_dim]
        """
        z = self.norm(z)
        z = z.mean(dim=1, keepdim=True) 
        feature = self.mlp(z)
        return feature

class TargetLatentToFeature(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(latent_dim * mlp_ratio)

        self.norm = nn.LayerNorm(latent_dim)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z: torch.Tensor):
        """
        z: [B, N, latent_dim]
        return: [B, N, feature_dim]
        """
        z = self.norm(z)
        z = z.mean(dim=1, keepdim=True) 
        feature = self.mlp(z)
        return feature