import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import sys
import types
_dep_shim = types.ModuleType('dependency_versions_check')
_dep_shim.dep_version_check = lambda *args, **kwargs: None
sys.modules['transformers.dependency_versions_check'] = _dep_shim
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging

class PDNCDualBranchNetwork(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small", num_classes=3, stat_dim=8):
        super(PDNCDualBranchNetwork, self).__init__()
        # Branch 1 (Contextual)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.deberta = AutoModel.from_pretrained(model_name, local_files_only=True)
        # DeBERTa v3 small hidden size is 768
        
        # Branch 2 (Statistical)
        # PDNN Concept: BatchNorm1d normalizes the output of each layer,
        # reducing internal covariate shift and stabilizing gradient flow
        self.statistical_branch = nn.Sequential(
            nn.Linear(stat_dim, 32),
            nn.BatchNorm1d(32),   # PDNN: Batch Normalization
            nn.ReLU(),            # PDNN: ReLU Activation Function
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Fusion Layer
        # PDNN Concept: Multi-modal feature fusion — combine deep contextual
        # features (768-dim DeBERTa) with shallow statistical features (32-dim)
        self.fusion = nn.Sequential(
            nn.Linear(768 + 32, 256),  # PDNN: Hidden layer 1
            nn.BatchNorm1d(256),        # PDNN: Batch Normalization
            nn.Dropout(0.3),            # PDNN: Dropout regularization
            nn.GELU(),                  # PDNN: GELU Activation (used in Transformers)
            nn.Linear(256, 128),        # PDNN: Hidden layer 2
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, num_classes) # PDNN: Output layer
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


def get_model(stat_dim=8):
    """
    Returns the initialized model, skipping quantization to enable PyTorch Autograd.
    """
    import os
    logging.info("Loading base model...")
    model = PDNCDualBranchNetwork(stat_dim=stat_dim)
    
    # Load calibrated weights if available
    if os.path.exists("model.pt"):
        logging.info("Found calibrated weights. Loading model.pt...")
        try:
            model.load_state_dict(torch.load("model.pt", map_location="cpu"))
        except RuntimeError as e:
            logging.warning(f"Strict state_dict loading encountered mismatch: {e}. Attempting strict=False...")
            try:
                model.load_state_dict(torch.load("model.pt", map_location="cpu"), strict=False)
            except Exception as e2:
                logging.error(f"Fallback loading failed: {e2}")
    else:
        logging.warning("model.pt not found. Using random initialized classification head.")
        
    model.eval()
    
    # Freeze all parameters to prevent massive CPU autograd overhead during XAI backward pass
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def get_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", local_files_only=True)
