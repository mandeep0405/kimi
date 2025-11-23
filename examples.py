"""
Complete Examples: Kimi K2 Key Components

This file demonstrates how to use the key components implemented
from the Kimi K2 paper.
"""

import torch
import torch.nn as nn
from muonclip_optimizer import MuonClipOptimizer
from moe_layer import MoELayer


def example_1_muonclip_optimizer():
    """
    Example 1: Using MuonClip Optimizer

    Demonstrates:
    - Creating a model
    - Using MuonClip optimizer
    - Simulating QK-Clip during training
    """
    print("\n" + "=" * 70)
    print("Example 1: MuonClip Optimizer with QK-Clip")
    print("=" * 70)

    # Create a simple attention-like model
    class SimpleAttention(nn.Module):
        def __init__(self, dim=512, num_heads=8):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            B, L, D = x.shape

            # Project Q, K, V
            q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

            # Compute attention scores
            scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)

            # Track max logits per head (for QK-Clip)
            max_logits = scores.abs().max(dim=-1)[0].max(dim=-1)[0]  # [B, num_heads]

            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum('bhqk,bkhd->bqhd', attn, v)
            out = out.reshape(B, L, D)
            out = self.out_proj(out)

            return out, max_logits.mean(dim=0)  # Average across batch

    # Create model and optimizer
    model = SimpleAttention(dim=512, num_heads=8)
    optimizer = MuonClipOptimizer(
        model.parameters(),
        lr=2e-4,
        qk_clip_threshold=100.0,
        weight_decay=0.1
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Optimizer: MuonClip (lr={optimizer.param_groups[0]['lr']})")
    print(f"QK-Clip threshold: {optimizer.param_groups[0]['qk_clip_threshold']}")

    # Simulate a training step
    print("\nSimulating training step...")

    x = torch.randn(4, 32, 512)  # [batch=4, seq=32, dim=512]
    output, max_logits = model(x)

    # Create dummy loss
    loss = output.mean()
    loss.backward()

    print(f"  Max attention logits per head: {max_logits.tolist()}")

    # Check if any head exceeds threshold
    threshold = optimizer.param_groups[0]['qk_clip_threshold']
    needs_clipping = (max_logits > threshold).any()

    if needs_clipping:
        print(f"  ⚠️  Some heads exceed threshold {threshold}, QK-Clip will be applied")
    else:
        print(f"  ✓  All heads below threshold {threshold}, no clipping needed")

    # Optimizer step with max logits
    max_logits_dict = {
        'q_proj.weight': max_logits,
        'k_proj.weight': max_logits
    }
    optimizer.step(max_logits=max_logits_dict)
    optimizer.zero_grad()

    print("✓ Training step completed successfully!")


def example_2_moe_layer():
    """
    Example 2: Mixture-of-Experts Layer

    Demonstrates:
    - Creating MoE layer
    - Routing tokens to experts
    - Load balancing
    """
    print("\n" + "=" * 70)
    print("Example 2: Mixture-of-Experts Layer")
    print("=" * 70)

    # Create MoE layer (scaled down from K2's 384 experts)
    moe = MoELayer(
        hidden_dim=512,
        num_experts=32,
        num_active=4,
        expert_dim=1024
    )

    print(f"MoE Configuration:")
    print(f"  Total experts: {moe.num_experts}")
    print(f"  Active per token: {moe.num_active}")
    print(f"  Sparsity ratio: {moe.num_experts / moe.num_active}x")
    print(f"  (K2 uses: 384 experts, 8 active, 48x sparsity)")

    # Process input
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, 512)

    print(f"\nProcessing input: {x.shape}")

    output, router_probs = moe(x)

    print(f"Output shape: {output.shape}")

    # Analyze routing
    print("\nExpert Routing Analysis:")
    tokens_processed = batch_size * seq_len
    expert_usage = router_probs.mean(dim=0)

    print(f"  Tokens processed: {tokens_processed}")
    print(f"  Average expert usage: {expert_usage.mean().item():.4f}")
    print(f"  Usage std dev: {expert_usage.std().item():.4f}")
    print(f"  Most used expert: {expert_usage.argmax().item()} ({expert_usage.max().item():.4f})")
    print(f"  Least used expert: {expert_usage.argmin().item()} ({expert_usage.min().item():.4f})")

    # Load balance loss
    lb_loss = moe.get_load_balance_loss(router_probs)
    print(f"\n  Load balance loss: {lb_loss.item():.6f}")
    print(f"  (Lower is better - indicates more balanced expert usage)")



if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" KIMI K2: Key Components Tutorial")
    print(" Based on: 'Kimi K2: Open Agentic Intelligence'")
    print("=" * 70)

    # Run all examples
    example_1_muonclip_optimizer()
    example_2_moe_layer()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. MuonClip: Token-efficient optimizer with QK-Clip for stability")
    print("  2. MoE: Sparse expert routing for efficient scaling")
    print("=" * 70 + "\n")
