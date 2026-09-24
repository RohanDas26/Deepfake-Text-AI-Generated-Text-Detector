import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss optimization script as requested.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where gamma = 2.0
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # can be a tensor of weights for each class
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [batch_size, num_classes]
        # targets: [batch_size]
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = focal_loss * at
            
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def train_model():
    """
    Training loop to calibrate the classification head using Focal Loss.
    """
    from model import PDNCDualBranchNetwork, get_tokenizer
    import numpy as np
    
    # Initialize model
    print("Loading model and tokenizer...")
    model = PDNCDualBranchNetwork()
    tokenizer = get_tokenizer()
    
    # Freeze the DeBERTa base model to prevent CPU hanging during backprop
    for param in model.deberta.parameters():
        param.requires_grad = False
        
    # Initialize Focal Loss with gamma=2.0
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    # Synthetic Data Creation
    texts = [
        # Human texts
        "Honestly, I wasn't expecting the server to launch on the first try. Spent three hours debugging.",
        "Hello, my name is XYZ and I am a software engineer. I love building tools.",
        "This is crazy! Why did it break again? I swear it worked yesterday.",
        # AI texts
        "The Real-Time AI Text Detection System is a high-performance application designed for text classification.",
        "Artificial intelligence refers to the simulation of human intelligence in machines.",
        "In conclusion, the study demonstrates significant improvements in the metrics.",
        # Paraphrased texts
        "To be honest, I didn't anticipate the server starting up immediately. I wasted three hours fixing it.",
        "The AI Text Detection System is a fast application made for classifying text.",
        "Ultimately, the research shows a major boost in the results."
    ]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    # Pre-computed exact features from NLPEngine
    stats = [
        [53.803688049316406, 0.5, 0.9375], 
        [22.24472999572754, 0.4444444444444444, 0.9333333333333333], 
        [110.77217864990234, 0.1767766952966369, 0.9230769230769231], 
        [97.31729125976562, 0.0, 0.9166666666666666], 
        [63.73149108886719, 0.0, 0.9090909090909091], 
        [156.65420532226562, 0.0, 0.8], 
        [53.08696746826172, 0.3333333333333333, 0.9411764705882353], 
        [211.7998809814453, 0.0, 0.9230769230769231], 
        [50.28373718261719, 0.0, 0.9]
    ]
        
    # Convert to tensors
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    statistical_features = torch.tensor(stats, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    
    print("Starting calibration training...")
    model.train()
    
    epochs = 40
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs["input_ids"], inputs["attention_mask"], statistical_features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    print("Training complete. Saving calibrated weights to model.pt...")
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt successfully.")

if __name__ == "__main__":
    train_model()
