import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple


def newton_schulz_iteration(matrix: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration for matrix orthogonalization.
    
    Handles rectangular matrices by working with the larger dimension.
    
    Args:
        matrix: Input matrix [n, m] to orthogonalize
        num_iters: Number of iterations (default: 5)
    
    Returns:
        Orthogonalized matrix with same shape as input
    """
    # Handle edge cases
    if matrix.dim() != 2:
        return matrix
    
    n, m = matrix.shape
    
    # Determine which dimension to orthogonalize
    # Work with the larger dimension for numerical stability
    if n >= m:
        # Orthogonalize columns: work with M^T M
        Z = matrix.clone()
        Z = Z / (Z.norm() + 1e-7)
        
        for _ in range(num_iters):
            # Z = 1.5Z - 0.5 Z(Z^T Z)
            Z = 1.5 * Z - 0.5 * Z @ (Z.t() @ Z)
    else:
        # Orthogonalize rows: work with M M^T
        Z = matrix.clone()
        Z = Z / (Z.norm() + 1e-7)
        
        for _ in range(num_iters):
            # Z = 1.5Z - 0.5 (ZZ^T)Z
            Z = 1.5 * Z - 0.5 * (Z @ Z.t()) @ Z
    
    return Z


class MuonClipOptimizer(torch.optim.Optimizer):
    """
    MuonClip: Muon optimizer with QK-Clip for attention stability.
    
    Features:
    - Newton-Schulz orthogonalization for momentum
    - RMS matching to Adam's update magnitude
    - Per-head QK-Clip for attention logit stability
    - Proper handling of 1D and 2D parameters
    
    Args:
        params: Model parameters (can include parameter groups with 'name' key)
        lr: Learning rate (default: 2e-4)
        momentum: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.1)
        ns_iters: Newton-Schulz iterations (default: 5)
        qk_clip_threshold: Max attention logit value (default: 100.0)
    """
    
    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_iters: int = 5,
        qk_clip_threshold: float = 100.0,
    ):
        # Initialize param_to_name BEFORE calling super().__init__()
        # because parent's __init__ calls add_param_group which needs it
        self.param_to_name = {}

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_iters=ns_iters,
            qk_clip_threshold=qk_clip_threshold,
        )
        super().__init__(params, defaults)

        # Note: param_to_name is populated via add_param_group()
        # which is called by parent's __init__
    
    @torch.no_grad()
    def step(
        self, 
        closure=None, 
        max_logits: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure to re-evaluate the model
            max_logits: Dict mapping layer names to max attention logits per head
                       Format: {'transformer.0.attn': tensor([98.5, 102.3, 95.7, ...])}
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['step'] = 0
                
                state['step'] += 1
                momentum_buffer = state['momentum']
                
                # Update momentum: M = β × M + grad
                momentum_buffer.mul_(group['momentum']).add_(grad)
                
                # Apply Muon update based on parameter shape
                if p.dim() >= 2:
                    # 2D+ parameters (weight matrices): Apply Muon
                    update = self._compute_muon_update(
                        momentum_buffer, 
                        grad, 
                        group['ns_iters']
                    )
                else:
                    # 1D parameters (biases, layer norm): Use momentum directly
                    update = momentum_buffer.clone()
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Apply update
                p.data.add_(update, alpha=-group['lr'])
        
        # Apply QK-Clip if max_logits provided
        if max_logits is not None:
            self._apply_qk_clip(max_logits)
        
        return loss
    
    def _compute_muon_update(
        self, 
        momentum: torch.Tensor, 
        grad: torch.Tensor, 
        ns_iters: int
    ) -> torch.Tensor:
        """
        Compute Muon update: Orthogonalize momentum + RMS match.
        
        Steps:
        1. Newton-Schulz orthogonalization
        2. Scale to larger dimension
        3. RMS matching to gradient magnitude
        
        Args:
            momentum: Momentum buffer
            grad: Current gradient
            ns_iters: Number of Newton-Schulz iterations
        
        Returns:
            Orthogonalized and RMS-matched update
        """
        # Step 1: Orthogonalize momentum
        ortho_momentum = newton_schulz_iteration(momentum, ns_iters)
        
        # Step 2: Scale by dimension (from Muon paper)
        # Makes update scale appropriately with layer size
        n, m = momentum.shape[-2:]
        scale = (max(n, m) ** 0.5) / (2.5)  # 2.5 is from original Muon
        ortho_update = ortho_momentum * scale
        
        # Step 3: RMS matching (critical for stability!)
        # Scale orthogonalized update to match gradient's RMS
        grad_rms = grad.norm() / (grad.numel() ** 0.5)
        update_rms = ortho_update.norm() / (ortho_update.numel() ** 0.5)
        
        rms_scale = grad_rms / (update_rms + 1e-8)
        ortho_update = ortho_update * rms_scale
        
        return ortho_update
    
    @torch.no_grad()
    def _apply_qk_clip(self, max_logits: Dict[str, torch.Tensor]):
        """
        Apply per-head QK-Clip to query and key projection weights.
        
        Algorithm:
        1. For each attention layer, check max logit per head
        2. If S_max^h > τ, compute γ_h = τ / S_max^h
        3. Scale: W_q ← W_q × √γ, W_k ← W_k × √γ
        
        Args:
            max_logits: Dict of layer name to max logits per head
                       Example: {'layer.0.attn': tensor([95.0, 102.3, 98.1, ...])}
        """
        for group in self.param_groups:
            threshold = group['qk_clip_threshold']
            
            for p in group['params']:
                param_id = id(p)
                if param_id not in self.param_to_name:
                    continue
                
                param_name = self.param_to_name[param_id]
                
                # Check if this is Q or K projection
                is_query = any(x in param_name.lower() for x in ['query', 'q_proj', 'wq'])
                is_key = any(x in param_name.lower() for x in ['key', 'k_proj', 'wk'])
                
                if not (is_query or is_key):
                    continue
                
                # Find matching logits
                # Layer name might be 'transformer.0.attn.q_proj'
                # max_logits key might be 'transformer.0.attn'
                layer_name = None
                for name in max_logits.keys():
                    if name in param_name:
                        layer_name = name
                        break
                
                if layer_name is None:
                    continue
                
                logits = max_logits[layer_name]  # Shape: [num_heads]
                
                # Compute per-head scaling: γ_h = min(1, τ / S_max^h)
                gamma = torch.clamp(threshold / (logits + 1e-7), max=1.0)
                sqrt_gamma = torch.sqrt(gamma)  # Shape: [num_heads]
                
                # Apply scaling
                # Weight shape typically: [d_model, d_model] or [num_heads * d_head, d_model]
                # Need to scale each head's weights separately
                
                if p.dim() == 2:
                    d_out, d_in = p.shape
                    num_heads = sqrt_gamma.shape[0]
                    head_dim = d_out // num_heads
                    
                    # Reshape to [num_heads, head_dim, d_in]
                    w_reshaped = p.data.view(num_heads, head_dim, d_in)
                    
                    # Scale each head
                    # sqrt_gamma: [num_heads] -> [num_heads, 1, 1]
                    scale = sqrt_gamma.view(num_heads, 1, 1)
                    w_reshaped.mul_(scale)
                    
                    # Reshape back
                    p.data = w_reshaped.view(d_out, d_in)
    
    def add_param_group(self, param_group):
        """Override to track parameter names."""
        super().add_param_group(param_group)
        
        # Update name mapping
        for p in param_group['params']:
            if 'name' in param_group:
                self.param_to_name[id(p)] = param_group['name']


# Helper function to create optimizer with proper naming
def create_muonclip_optimizer(model: nn.Module, lr: float = 2e-4, **kwargs):
    """
    Create MuonClip optimizer with automatic parameter naming.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        **kwargs: Additional optimizer arguments
    
    Returns:
        MuonClipOptimizer instance with parameter names tracked
    """
    param_groups = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_groups.append({
                'params': [param],
                'name': name
            })
    
    return MuonClipOptimizer(param_groups, lr=lr, **kwargs)


