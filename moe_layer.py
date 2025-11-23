"""
Mixture-of-Experts (MoE) Layer Implementation

Based on Kimi K2 architecture:
- 384 total experts, 8 active per token
- Top-k expert routing
- Load balancing mechanisms

Key findings from paper:
- Sparsity scaling: More experts (higher sparsity) improves performance
- Kimi K2 uses sparsity ratio of 48 (384/8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """
    Single expert network: FFN with SwiGLU activation.

    Architecture:
    x -> W_up -> SwiGLU -> W_down -> output

    Args:
        hidden_dim: Input/output dimension
        expert_dim: Expert's internal dimension (default: 2048 for K2)
    """

    def __init__(self, hidden_dim: int, expert_dim: int = 2048):
        super().__init__()
        self.w_up = nn.Linear(hidden_dim, expert_dim * 2, bias=False)
        self.w_down = nn.Linear(expert_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.

        SwiGLU(x) = (W_up_1(x) * σ(W_up_2(x))) @ W_down
        where σ is swish/silu activation
        """
        gate, up = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer.

    Routes each token to top-k experts based on learned routing weights.

    Args:
        hidden_dim: Model hidden dimension
        num_experts: Total number of experts (384 in K2)
        num_active: Number of active experts per token (8 in K2)
        expert_dim: Expert internal dimension (2048 in K2)
    """

    def __init__(
        self,
        hidden_dim: int = 7168,
        num_experts: int = 384,
        num_active: int = 8,
        expert_dim: int = 2048,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.hidden_dim = hidden_dim

        # Router: learns which experts to activate
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(hidden_dim, expert_dim) for _ in range(num_experts)
        ])

        # Shared expert (always active, as mentioned in paper)
        self.shared_expert = Expert(hidden_dim, expert_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            output: Mixed expert outputs [batch_size, seq_len, hidden_dim]
            router_probs: Router probabilities for analysis
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]

        # Compute routing scores
        router_logits = self.router(x_flat)  # [B*L, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts per token
        topk_probs, topk_indices = torch.topk(
            router_probs, self.num_active, dim=-1
        )  # [B*L, num_active]

        # Normalize top-k probabilities
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == i).any(dim=-1)  # [B*L]

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]

                # Compute expert output
                expert_output = self.experts[i](expert_input)

                # Get routing weights for this expert
                expert_positions = topk_indices == i
                weights = torch.where(
                    expert_positions,
                    topk_probs,
                    torch.zeros_like(topk_probs)
                ).sum(dim=-1, keepdim=True)[expert_mask]

                # Accumulate weighted output
                output[expert_mask] += weights * expert_output

        # Add shared expert (always active)
        shared_output = self.shared_expert(x_flat)
        output += shared_output

        # Reshape back
        output = output.view(batch_size, seq_len, hidden_dim)

        return output, router_probs

    def get_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage.

        Args:
            router_probs: Router probabilities [batch*seq, num_experts]

        Returns:
            Load balance auxiliary loss
        """
        # Compute expert usage frequency
        expert_usage = router_probs.mean(dim=0)  # [num_experts]

        # Ideal usage: 1 / num_experts
        ideal_usage = 1.0 / self.num_experts

        # L2 penalty for deviation from ideal
        load_balance_loss = ((expert_usage - ideal_usage) ** 2).sum()

        return load_balance_loss


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Kimi K2 MoE Layer Example")
    print("=" * 60)

    # Create a smaller version for demonstration
    moe = MoELayer(
        hidden_dim=512,
        num_experts=32,  # Smaller than K2's 384
        num_active=4,    # Smaller than K2's 8
        expert_dim=1024  # Smaller than K2's 2048
    )

    # Input: [batch=2, seq_len=10, hidden_dim=512]
    x = torch.randn(2, 10, 512)

    print(f"\nInput shape: {x.shape}")
    print(f"Number of experts: {moe.num_experts}")
    print(f"Active experts per token: {moe.num_active}")
    print(f"Sparsity ratio: {moe.num_experts / moe.num_active}")

    # Forward pass
    output, router_probs = moe(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Router probs shape: {router_probs.shape}")

    # Compute load balance loss
    lb_loss = moe.get_load_balance_loss(router_probs)
    print(f"\nLoad balance loss: {lb_loss.item():.4f}")

    # Analyze expert usage
    expert_usage = router_probs.mean(dim=0)
    print(f"\nExpert usage statistics:")
    print(f"  Mean: {expert_usage.mean().item():.4f}")
    print(f"  Std:  {expert_usage.std().item():.4f}")
    print(f"  Min:  {expert_usage.min().item():.4f}")
    print(f"  Max:  {expert_usage.max().item():.4f}")

    print("\n" + "=" * 60)
    print("MoE layer demonstration completed!")
    print("=" * 60)
