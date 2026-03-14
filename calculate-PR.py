import torch
from transformers import AutoModelForCausalLM

def calc_participation_ratio(covariance_matrix):
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)
    eigenvalues = torch.relu(eigenvalues) # Clean tiny numerical artifacts
    # PR = (Sum of evalues)^2 / Sum of (evalues^2)
    pr = (torch.sum(eigenvalues)**2) / torch.sum(eigenvalues**2)
    return pr.item()

def run_experiment_a():
    print("Loading Llama-3-8B Embeddings...")
    model_id = "meta-llama/Meta-Llama-3-8B"
    # Load on CPU/GPU to extract weights
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    embed_weights = model.model.embed_tokens.weight.detach() # Shape: [128256, 4096]

    # 1. Standard English Tokens (Sample 1000 common subwords)
    # Tokens 1000 to 2000 are standard English text
    english_tokens = embed_weights[1000:2000] 
    
    # 2. Simulate Orthogonal Anchors (Using Gram-Schmidt / QR decomposition)
    dim = embed_weights.shape[1]
    q, r = torch.linalg.qr(torch.randn(dim, 50)) 
    orthogonal_anchors = q.T # Now perfectly 90-degrees apart [50, 4096]

    # Calculate Covariance
    english_cov = torch.cov(english_tokens.T)
    anchor_cov = torch.cov(orthogonal_anchors.T)

    pr_english = calc_participation_ratio(english_cov)
    pr_anchors = calc_participation_ratio(anchor_cov)

    print(f"\n--- Experiment A Results ---")
    print(f"Standard English Tokens PR: {pr_english:.2f} (Highly diffuse/overlapping)")
    print(f"Orthogonal Anchors PR:      {pr_anchors:.2f} (Mathematically pristine)")

if __name__ == "__main__":
    run_experiment_a()