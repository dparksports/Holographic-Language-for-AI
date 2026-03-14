import torch
import triton
import triton.language as tl

@triton.jit
def fused_l2_hopfield_kernel(
    noisy_ptr, dict_ptr, out_ptr,
    D: tl.constexpr, K: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Executes entirely in ultra-fast SRAM/L1 Cache
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    
    # 1. Load the drifting noisy vector
    noisy_vec = tl.load(noisy_ptr + offsets, mask=mask)
    
    min_dist = float('inf')
    best_idx = 0
    
    # 2. Subtraction loop over the K (50) anchors residing in cache
    for k in range(K):
        anchor = tl.load(dict_ptr + k * D + offsets, mask=mask)
        # THE USER'S INSIGHT: Subtraction and squared L2 Distance
        diff = noisy_vec - anchor
        dist = tl.sum(diff * diff, axis=0)
        
        # Track the closest anchor
        if dist < min_dist:
            min_dist = dist
            best_idx = k
            
    # 3. Snap to the closest pristine anchor and write to output
    best_anchor = tl.load(dict_ptr + best_idx * D + offsets, mask=mask)
    tl.store(out_ptr + offsets, best_anchor, mask=mask)

def fast_sram_cleanup(noisy_vector, dictionary):
    D = noisy_vector.shape[-1]
    K = dictionary.shape[0]
    out = torch.empty_like(noisy_vector)
    
    # BLOCK_SIZE must be next power of 2 (4096 is already a power of 2)
    fused_l2_hopfield_kernel[(1,)](
        noisy_vector, dictionary, out, 
        D=D, K=K, BLOCK_SIZE=4096
    )
    return out