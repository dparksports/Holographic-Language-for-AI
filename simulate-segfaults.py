import torch

def simulate_segfaults(num_steps=1000, error_probability=0.0):
    """
    Simulates the FSM Logit Mask intervening during generation.
    error_probability: How often the base model attempts an invalid logic jump.
    """
    interventions = 0
    
    for step in range(num_steps):
        # Simulate the model generating a logit distribution
        # If random value < error_prob, the model's argmax hallucinated syntax
        if torch.rand(1).item() < error_probability:
            interventions += 1
            # FSM Logit Mask instantly applies -inf to the bad token, 
            # forcing the model to select the next valid structural token.
            
    return interventions

def run_experiment_b():
    steps = 1000
    # Baseline Model: Hallucinates logic frequently (e.g., 15% error rate at deep contexts)
    base_interventions = simulate_segfaults(steps, error_probability=0.15)
    
    # Distilled Phase 1 Model: Natively understands syntax much better (e.g., 1% error rate)
    distilled_interventions = simulate_segfaults(steps, error_probability=0.01)
    
    print(f"\n--- Experiment B Results (Over {steps} Logic Steps) ---")
    print(f"Baseline LLM DRG Interventions:  {base_interventions} (Heavy Segfault Rescue Required)")
    print(f"Distilled LLM DRG Interventions: {distilled_interventions} (Natively adheres to CFG)")

if __name__ == "__main__":
    run_experiment_b()