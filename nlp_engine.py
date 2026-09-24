import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import logging

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class NLPEngine:
    def __init__(self):
        logging.info("Initializing NLPEngine...")
        self.model_id = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.eval()

    def get_perplexity(self, text: str) -> float:
        # Optimization: Truncate to 512 tokens max for fast latency
        encodings = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = encodings.input_ids
        
        # Optimization: use inference_mode for faster execution
        with torch.inference_mode():
            # Pass input_ids as both inputs and labels
            outputs = self.model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss

        ppl = torch.exp(neg_log_likelihood).item()
        
        if math.isinf(ppl) or math.isnan(ppl):
            return 10000.0
        return ppl

    def get_burstiness(self, text: str) -> float:
        # Burstiness (B) = \sigma_{length} / \mu_{length}
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        lengths = [len(word_tokenize(s)) for s in sentences]
        
        if len(lengths) <= 1:
            # Neutralize burstiness bias for single sentences rather than passing hard 0.0
            return 0.50
        
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        if mean_len == 0:
            return 0.0
        
        return float(std_len / mean_len)

    def get_ttr(self, text: str) -> float:
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum()]
        if not words:
            return 0.0
            
        unique_words = set(words)
        return len(unique_words) / len(words)

    def extract_features(self, raw_text: str) -> np.ndarray:
        if not raw_text or len(raw_text.strip()) == 0:
            return np.array([0.0, 0.0, 0.0])
            
        # Code/Syntax filter
        code_keywords = ['def ', 'import ', 'return ', ';', ' = ', '=>', 'class ', 'const ', 'let ']
        code_score = sum(1 for k in code_keywords if k in raw_text)
        
        if code_score >= 2:
            ppl = 600.0  # Bypass normal scoring and force extreme PPL to trigger Human heuristic
        else:
            ppl = self.get_perplexity(raw_text)
            
        burstiness = self.get_burstiness(raw_text)
        ttr = self.get_ttr(raw_text)
        
        return np.array([ppl, burstiness, ttr])


def extract_attributions(model, tokenizer, text, statistical_features_tensor):
    """
    Extract token-level attributions using Input x Gradient on DeBERTa.
    Returns list of dicts: [{'word': str, 'score': float}]
    Positive scores drive the prediction toward AI, negative towards Human.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Ensure model is in eval mode
    model.eval()
    
    # Extract embeddings and enable gradients
    embeddings = model.deberta.embeddings.word_embeddings(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)
    embeddings.retain_grad()
    
    # Zero out any existing gradients before forward pass
    model.zero_grad()
    
    # Forward pass using inputs_embeds
    logits = model(
        inputs_embeds=embeddings, 
        attention_mask=attention_mask, 
        statistical_features=statistical_features_tensor
    )
    
    # Backpropagate the logit for the AI class (Class 1)
    logit_ai = logits[0, 1]
    logit_ai.backward()
    
    # Input x Gradient saliency with detachment
    saliency = (embeddings.grad * embeddings).sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    # Clear gradients immediately after extraction to free CPU graph memory
    model.zero_grad()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    attributions = []
    for i, token in enumerate(tokens):
        # Skip special tokens
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, '[CLS]', '[SEP]', '[PAD]']:
            continue
            
        score = float(saliency[i])
        # Clean up DeBERTa's space character
        display_token = token.replace(' ', ' ').strip()
        if display_token:
            attributions.append({"word": display_token, "score": score})
            
    # Garbage Collection & Tensor Detachment
    del logits
    del logit_ai
    del embeddings
    import gc
    gc.collect()
            
    return attributions
