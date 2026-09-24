import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging

class PDNCDualBranchNetwork(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small", num_classes=3):
        super(PDNCDualBranchNetwork, self).__init__()
        # Branch 1 (Contextual)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.deberta = AutoModel.from_pretrained(model_name)
        # DeBERTa v3 small hidden size is 768
        
        # Branch 2 (Statistical)
        self.statistical_branch = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 32, 128),  # 800 -> 128
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids=None, attention_mask=None, statistical_features=None, inputs_embeds=None):
        # Branch 1: Contextual
        if inputs_embeds is not None:
            deberta_output = self.deberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            deberta_output = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
            
        # Mean pooling to get sequence representation (768-dim)
        token_embeddings = deberta_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Branch 2: Statistical
        stat_output = self.statistical_branch(statistical_features)
        
        # Fusion
        fused = torch.cat((pooled_output, stat_output), dim=1)
        logits = self.fusion(fused)
        
        return logits


def get_model():
    """
    Returns the initialized model, skipping quantization to enable PyTorch Autograd.
    """
    import os
    logging.info("Loading base model...")
    model = PDNCDualBranchNetwork()
    
    # Load calibrated weights if available
    if os.path.exists("model.pt"):
        logging.info("Found calibrated weights. Loading model.pt...")
        model.load_state_dict(torch.load("model.pt"))
    else:
        logging.warning("model.pt not found. Using random initialized classification head.")
        
    model.eval()
    
    # Freeze all parameters to prevent massive CPU autograd overhead during XAI backward pass
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def get_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
