import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def execute_embedding_surgery():
    print("Loading Base Model for Surgery...")
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load in bfloat16 to fit beautifully in RTX 5090 VRAM
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16)
    
    # 1. Identify 50 unused reserved tokens in Llama 3
    anchor_token_ids = list(range(128000, 128050)) 
    embeddings = model.model.embed_tokens.weight.data
    
    # 2. Apply Orthogonalization (QR Decomposition) to random vectors
    print("Applying Gram-Schmidt Orthogonal Reset...")
    dim = embeddings.shape[1]
    # Do math in FP32 for precision
    random_vecs = torch.randn(len(anchor_token_ids), dim, dtype=torch.float32, device='cuda')
    q, r = torch.linalg.qr(random_vecs.T)
    orthogonal_anchors = q.T.to(torch.bfloat16)
    
    # 3. Overwrite the weights in the model
    embeddings[anchor_token_ids] = orthogonal_anchors
    
    # To prevent these perfect vectors from drifting during future SFT, 
    # you would register a backward hook to zero out their gradients.
    def freeze_anchors_hook(grad):
        grad[anchor_token_ids] = 0.0
        return grad
    model.model.embed_tokens.weight.register_hook(freeze_anchors_hook)
    
    # Save the surgically altered base model
    save_path = "./Llama-3-8B-Surgically-Altered"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Phase 1 - Stage 1 Complete: Anchors secured and saved to {save_path}.")

# execute_embedding_surgery()