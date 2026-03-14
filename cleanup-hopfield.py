import torch
import torch.nn.functional as F

def continuous_hopfield_cleanup(noisy_vector, dictionary_matrix, beta=15.0):
    # E = -lse(beta, X^T * xi)
    # Effectively softmax attention over the pristine dictionary anchors
    similarities = F.linear(noisy_vector, dictionary_matrix)
    attention_weights = F.softmax(beta * similarities, dim=-1)
    cleaned_vector = torch.matmul(attention_weights, dictionary_matrix)
    return F.normalize(cleaned_vector, p=2, dim=-1)

def run_experiment_c():
    d = 4096 
    steps = 1000
    
    # Create 50 pristine dictionary anchors
    dictionary = F.normalize(torch.randn(50, d), p=2, dim=-1)
    target_anchor = dictionary[0].clone()
    
    drift_vector = target_anchor.clone()
    cleaned_vector = target_anchor.clone()
    
    print(f"\n--- Experiment C Results (After {steps} recursive steps) ---")
    print(f"{'Step':<10} | {'FP32 Drift (Unprotected)':<30} | {'Hopfield Stabilized':<20}")
    
    for i in range(1, steps + 1):
        # Inject standard floating point precision noise/crosstalk inherent in deep networks
        noise = torch.randn(d) * 0.05
        
        # 1. Pure Accumulation (Drift)
        drift_vector = F.normalize(drift_vector + noise, p=2, dim=-1)
        
        # 2. Accumulation + Hopfield Cleanup
        cleaned_vector = F.normalize(cleaned_vector + noise, p=2, dim=-1)
        cleaned_vector = continuous_hopfield_cleanup(cleaned_vector, dictionary)
        
        if i % 200 == 0:
            sim_drift = F.cosine_similarity(drift_vector, target_anchor, dim=0).item()
            sim_clean = F.cosine_similarity(cleaned_vector, target_anchor, dim=0).item()
            print(f"{i:<10} | {sim_drift:<30.4f} | {sim_clean:<20.4f}")

if __name__ == "__main__":
    run_experiment_c()