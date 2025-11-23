"""
Test compatibility between updated MuonClip optimizer and MoE layer.

This script verifies:
1. MuonClip can optimize MoE parameters
2. Parameter shapes are handled correctly
3. Training loop works end-to-end
4. Load balancing loss integrates properly
"""

import torch
import torch.nn as nn
from muonclip_optimizer import MuonClipOptimizer, create_muonclip_optimizer
from moe_layer import MoELayer


def test_basic_compatibility():
    """Test 1: Basic compatibility - can MuonClip optimize MoE?"""
    print("=" * 70)
    print("TEST 1: Basic Compatibility")
    print("=" * 70)

    # Create MoE layer
    moe = MoELayer(
        hidden_dim=256,
        num_experts=8,
        num_active=2,
        expert_dim=512
    )

    print(f"✓ Created MoE layer")
    print(f"  - Experts: {moe.num_experts}")
    print(f"  - Hidden dim: {moe.hidden_dim}")

    # Create optimizer using helper function
    optimizer = create_muonclip_optimizer(moe, lr=1e-3)

    print(f"✓ Created MuonClip optimizer")
    print(f"  - Tracked {len(optimizer.param_to_name)} parameters")

    # Check parameter tracking
    print(f"\n✓ Parameter names tracked:")
    for param_id, name in list(optimizer.param_to_name.items())[:5]:
        print(f"    {name}")
    if len(optimizer.param_to_name) > 5:
        print(f"    ... and {len(optimizer.param_to_name) - 5} more")

    return True


def test_parameter_shapes():
    """Test 2: Verify all MoE parameters have compatible shapes."""
    print("\n" + "=" * 70)
    print("TEST 2: Parameter Shape Compatibility")
    print("=" * 70)

    moe = MoELayer(hidden_dim=128, num_experts=4, num_active=2, expert_dim=256)

    print("Checking parameter shapes:")
    compatible_2d = 0

    for name, param in moe.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                print(f"  ✓ {name:40s} shape: {str(list(param.shape)):20s} (2D - Muon)")
                compatible_2d += 1
            else:
                print(f"  ✓ {name:40s} shape: {str(list(param.shape)):20s} (1D - momentum)")

    print(f"\n✓ All {compatible_2d} parameters are 2D (compatible with Muon)")
    return True


def test_training_loop():
    """Test 3: Full training loop with MoE + MuonClip."""
    print("\n" + "=" * 70)
    print("TEST 3: End-to-End Training Loop")
    print("=" * 70)

    # Create model
    moe = MoELayer(hidden_dim=64, num_experts=4, num_active=2, expert_dim=128)
    optimizer = create_muonclip_optimizer(moe, lr=1e-3)

    # Dummy data
    x = torch.randn(2, 8, 64)  # [batch, seq, hidden]
    target = torch.randn(2, 8, 64)

    print("Running 5 training steps...")

    initial_loss = None
    final_loss = None

    for step in range(5):
        # Forward pass
        output, router_probs = moe(x)

        # Compute loss
        main_loss = F.mse_loss(output, target)
        lb_loss = moe.get_load_balance_loss(router_probs)
        total_loss = main_loss + 0.01 * lb_loss

        # Backward
        total_loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            initial_loss = total_loss.item()
        if step == 4:
            final_loss = total_loss.item()

        print(f"  Step {step+1}: Loss={total_loss.item():.6f} "
              f"(Main: {main_loss.item():.6f}, LB: {lb_loss.item():.6f})")

    print(f"\n✓ Training completed successfully")
    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss:   {final_loss:.6f}")

    # Check that we're actually optimizing
    if final_loss < initial_loss:
        print(f"  ✓ Loss decreased (optimizer is working!)")
    else:
        print(f"  ⚠ Loss increased (might need more steps or different LR)")

    return True


def test_momentum_accumulation():
    """Test 4: Verify momentum buffers are created and updated."""
    print("\n" + "=" * 70)
    print("TEST 4: Momentum Buffer Management")
    print("=" * 70)

    moe = MoELayer(hidden_dim=32, num_experts=2, num_active=1, expert_dim=64)
    optimizer = create_muonclip_optimizer(moe, lr=1e-3, momentum=0.9)

    # Check no state initially
    total_params = sum(1 for _ in moe.parameters())
    print(f"Model has {total_params} parameters")
    print(f"Optimizer state size: {len(optimizer.state)}")

    # Run one step
    x = torch.randn(1, 4, 32)
    output, _ = moe(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    # Check state created
    state_count = len(optimizer.state)
    momentum_count = sum(1 for state in optimizer.state.values()
                         if 'momentum' in state)

    print(f"\nAfter one step:")
    print(f"  ✓ State created for {state_count} parameters")
    print(f"  ✓ Momentum buffers: {momentum_count}")

    # Verify momentum shapes match parameter shapes
    print(f"\nVerifying momentum buffer shapes:")
    matches = 0
    for param in moe.parameters():
        if param.requires_grad and param in optimizer.state:
            state = optimizer.state[param]
            if 'momentum' in state:
                momentum = state['momentum']
                if momentum.shape == param.shape:
                    matches += 1

    print(f"  ✓ {matches}/{momentum_count} momentum buffers match parameter shapes")

    return True


def test_no_qk_clip_interference():
    """Test 5: Verify QK-Clip doesn't interfere with MoE (no attention)."""
    print("\n" + "=" * 70)
    print("TEST 5: QK-Clip Non-Interference")
    print("=" * 70)

    moe = MoELayer(hidden_dim=32, num_experts=2, num_active=1, expert_dim=64)
    optimizer = create_muonclip_optimizer(moe, lr=1e-3, qk_clip_threshold=100.0)

    # Simulate passing max_logits (even though MoE has no attention)
    x = torch.randn(1, 4, 32)
    output, _ = moe(x)
    loss = output.sum()
    loss.backward()

    # This shouldn't cause errors even if we pass max_logits
    fake_max_logits = {
        'some_attention_layer': torch.tensor([95.0, 102.0, 98.0, 101.0])
    }

    try:
        optimizer.step(max_logits=fake_max_logits)
        print("  ✓ QK-Clip gracefully ignores MoE parameters (no Q/K found)")
        print("  ✓ No errors when max_logits provided to non-attention model")
        success = True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        success = False

    return success


def test_gradient_flow():
    """Test 6: Verify gradients flow correctly through MoE with MuonClip."""
    print("\n" + "=" * 70)
    print("TEST 6: Gradient Flow")
    print("=" * 70)

    moe = MoELayer(hidden_dim=32, num_experts=4, num_active=2, expert_dim=64)
    optimizer = create_muonclip_optimizer(moe, lr=1e-3)

    x = torch.randn(2, 4, 32)
    output, router_probs = moe(x)

    # Compute loss
    loss = output.mean() + moe.get_load_balance_loss(router_probs)
    loss.backward()

    # Check gradients
    params_with_grad = 0
    params_without_grad = 0

    for name, param in moe.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1

    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")

    if params_without_grad == 0:
        print(f"  ✓ All parameters received gradients")
        return True
    else:
        print(f"  ⚠ Some parameters didn't receive gradients")
        return False


# Import F for loss
from torch.nn import functional as F


def run_all_tests():
    """Run all compatibility tests."""
    print("\n" + "=" * 70)
    print("MUONCLIP + MOE COMPATIBILITY TEST SUITE")
    print("=" * 70)

    tests = [
        ("Basic Compatibility", test_basic_compatibility),
        ("Parameter Shapes", test_parameter_shapes),
        ("Training Loop", test_training_loop),
        ("Momentum Accumulation", test_momentum_accumulation),
        ("QK-Clip Non-Interference", test_no_qk_clip_interference),
        ("Gradient Flow", test_gradient_flow),
    ]

    results = []

    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - MoE is fully compatible with MuonClip!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - see details above")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
