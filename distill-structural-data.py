import torch
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForLanguageModeling
from transformers import TrainingArguments
from unsloth import FastLanguageModel

# Custom Data Collator for Loss Masking
class StructuralDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, anchor_token_ids):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.anchor_ids = set(anchor_token_ids)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        # Mask out semantic English tokens
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                token_id = labels[i, j].item()
                # If it is NOT an anchor token, tell PyTorch Loss to ignore it (-100)
                if token_id not in self.anchor_ids and token_id != -100:
                    labels[i, j] = -100 
                    
        batch["labels"] = labels
        return batch

def execute_structural_distillation():
    print("Initializing Unsloth for Structural Distillation on RTX 5090...")
    
    # 1. Load the Surgically Altered model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./Llama-3-8B-Surgically-Altered",
        max_seq_length = 4096,
        dtype = torch.bfloat16, 
        load_in_4bit = False, # 32GB VRAM can fit 8B natively in bf16 with LoRA
    )
    
    # 2. Add LoRA adapters strictly to the routing layers (Attention + MLPs)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )
    
    # Assume we generated synthetic data using Project-Chevron's fuzzer (Stage 2)
    dataset = load_dataset("json", data_files="synthetic_logic_data.json", split="train")
    
    # 3. Train!
    anchor_token_ids = list(range(128000, 128050))
    custom_collator = StructuralDataCollator(tokenizer, anchor_token_ids)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        data_collator = custom_collator,
        max_seq_length = 4096,
        args = TrainingArguments(
            per_device_train_batch_size = 4, # Maximize 5090 VRAM
            gradient_accumulation_steps = 4,
            max_steps = 1000,
            learning_rate = 2e-4,
            bf16 = True, # Use 5090's bf16 tensor cores
            optim = "adamw_8bit",
            output_dir = "outputs",
        ),
    )
    
    print("Executing Phase 1 - Stage 3: Supervised Structural Distillation...")
    trainer.train()
    
    model.save_pretrained_merged("Holographic-Llama-3-Phase1", tokenizer, save_method = "lora")
    print("Training Complete. Model MLPs are now structurally distilled.")

# execute_structural_distillation()