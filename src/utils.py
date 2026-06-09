import torch
import editdistance
from typing import List
import src.config as config

class LabelEncoder:
    def __init__(self):
        """
        Map alphanumeric characters to integers and vice versa
        CRNN model using  (CTC) 
        special 'blank' token (mapped to index 0)
        """
        self.blank_token = "-"
        self.chars = [self.blank_token] + list(config.VOCAB)
        
        # Bi-directional mapping 
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        self.num_classes = len(self.chars) #37 

    def encode(self, text: str) -> torch.Tensor:
        """raw string sequence ->long integer tensor"""
        encoded = [self.char_to_idx[char] for char in text.upper()]
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of token integers directly back into text
        for ground-truth verification and non-CTC decoding steps
        """
        return "".join([self.idx_to_char[idx] for idx in token_ids if idx in self.idx_to_char])

def ctc_greedy_decode(log_probs: torch.Tensor, encoder: LabelEncoder) -> List[str]:
    """
    Decodes raw model output matrix down to clean text sequences.
    
    Args:
        log_probs: Shape (Time_steps, Batch_size, Num_classes) output from CRNN.
        encoder: Initialized LabelEncoder instance.
        
    Steps:
        1. Take argmax to find the most probable token at each time step
        2. Collapse adjacent duplicate characters
        3. Drop the CTC blank tokens (index 0)
    """
    # Finding max indices across the class dimension (dim 2), then transpose to (Batch, Time)
    argmax_indices = log_probs.argmax(dim=2).permute(1, 0)
    
    decoded_outputs = []
    for batch_element in argmax_indices:
        collapsed_sequence = []
        previous_token = -1
        
        for current_token in batch_element:
            token_val = current_token.item()
            
            # Step 2: Collapse consecutive identical tokens
            if token_val != previous_token:
                # Step 3: Strip away special CTC blank token
                if token_val != 0: 
                    collapsed_sequence.append(token_val)
                previous_token = token_val
                
        #remaining valid indices->strings
        decoded_str = "".join([encoder.idx_to_char[idx] for idx in collapsed_sequence])
        decoded_outputs.append(decoded_str)
        
    return decoded_outputs

def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute CER via Levenshtein distance
    Formula: (Insertions + Deletions + Substitutions) / Total Target Length
    """
    total_distance = 0
    total_character_count = 0
    
    for pred, target in zip(predictions, targets):
        
        dist = editdistance.eval(pred, target)
        total_distance += dist
        total_character_count += len(target)
        
    if total_character_count == 0:
        return 0.0
        
    return total_distance / total_character_count