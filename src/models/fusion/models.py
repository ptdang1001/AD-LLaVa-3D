# -*- utf-8 -*-
# built-in modules

# third-party modules
import torch
import torch.nn as nn
import lightning as L

# custom modules


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_ratio=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.mlp(x)


class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = int(num_experts * 0.51) + 1

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts, bias=False), nn.Softmax(dim=-1)
        )

        # Expert networks
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Calculate gate scores
        gate_scores = self.gate(x)  # [batch_size, num_experts]

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        final_output = torch.zeros_like(x)

        # Combine expert outputs
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_score = top_k_scores[:, i].unsqueeze(-1)

            # Get expert outputs for each sample
            batch_size = x.shape[0]
            expert_outputs = torch.stack(
                [self.experts[expert_idx[j]](x[j]) for j in range(batch_size)]
            )

            final_output += expert_score * expert_outputs

        return self.layer_norm(final_output + x)  # Add residual connection


class MoE(nn.Module):
    def __init__(
        self,
        input_dim=4864,
        hidden_dim=4864,
        output_dim=4864,
        num_layers=1,
        num_experts_per_layer=5,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                MoELayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_experts=num_experts_per_layer,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =========================
# Fusion Model
# =========================
class GatedFusion(nn.Module):
    def __init__(self, d_fusion, d_multiview, d_language, dropout_ratio=0.5):
        super().__init__()
        self.d_fusion = d_fusion
        self.d_multiview = d_multiview
        self.d_language = d_language

        # Gating mechanism
        self.multiview_gate = nn.Sequential(
            nn.Linear(d_fusion, d_multiview, bias=False), nn.Sigmoid()
        )
        self.language_gate = nn.Sequential(
            nn.Linear(d_fusion, d_language, bias=False), nn.Sigmoid()
        )

        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_fusion, d_fusion, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(d_fusion, d_fusion, bias=False),
        )
        self.final_ln = nn.LayerNorm(d_fusion)

    def forward(self, pooled_multiview_emb, pooled_language_emb):
        # Concatenate for gate computation
        concatenated_emb = torch.cat(
            [pooled_multiview_emb, pooled_language_emb], dim=-1
        )

        # Compute gates
        multiview_weight = self.multiview_gate(concatenated_emb)
        language_weight = self.language_gate(concatenated_emb)

        # Apply gates
        gated_multiview = pooled_multiview_emb * multiview_weight
        gated_language = pooled_language_emb * language_weight

        # Fuse features
        fused_emb = torch.cat([gated_multiview, gated_language], dim=-1)
        fused_emb = self.fusion_layer(fused_emb)

        # Residual connection and normalization
        fused_emb = self.final_ln(fused_emb + concatenated_emb)

        return fused_emb
