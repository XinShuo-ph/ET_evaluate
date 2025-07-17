import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_code(code: str) -> List[str]:
    """
    Preprocess code for better BLEU comparison
    
    Args:
        code: Raw code string
        
    Returns:
        List of normalized tokens
    """
    if not code or not isinstance(code, str):
        return []
    
    # Remove extra whitespace and normalize
    code = re.sub(r'\s+', ' ', code.strip())
    
    # Normalize common C/C++ patterns
    code = re.sub(r'\s*{\s*', ' { ', code)  # Normalize braces
    code = re.sub(r'\s*}\s*', ' } ', code)
    code = re.sub(r'\s*;\s*', ' ; ', code)  # Normalize semicolons
    code = re.sub(r'\s*,\s*', ' , ', code)  # Normalize commas
    code = re.sub(r'\s*\(\s*', ' ( ', code)  # Normalize parentheses
    code = re.sub(r'\s*\)\s*', ' ) ', code)
    
    # Split into tokens (simple whitespace tokenization for code)
    tokens = code.split()
    
    # Remove empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

def compute_bleu_score(generated_code: str, reference_code: str) -> float:
    """
    Compute BLEU score between generated and reference code
    
    Args:
        generated_code: Generated code string
        reference_code: Ground truth reference code
        
    Returns:
        BLEU score (0-1)
    """
    # Preprocess both codes
    generated_tokens = preprocess_code(generated_code)
    reference_tokens = preprocess_code(reference_code)
    
    if not generated_tokens or not reference_tokens:
        return 0.0
    
    # Compute BLEU score with smoothing
    try:
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            [reference_tokens],  # Reference is a list of token lists
            generated_tokens,    # Candidate tokens
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing_function
        )
        return bleu_score
    except:
        # Fallback for edge cases
        return 0.0

if __name__ == "__main__":
    # Test the BLEU function
    reference_code = """
    #include "cctk.h"
    #include "cctk_Arguments.h"
    #include "cctk_Parameters.h"
    
    void MyThorn_InitialData(CCTK_ARGUMENTS) {
        DECLARE_CCTK_ARGUMENTS;
        DECLARE_CCTK_PARAMETERS;
        
        for (int k = 0; k < cctk_lsh[2]; k++) {
            for (int j = 0; j < cctk_lsh[1]; j++) {
                for (int i = 0; i < cctk_lsh[0]; i++) {
                    int idx = CCTK_GFINDEX3D(cctkGH, i, j, k);
                    phi[idx] = 1.0;
                }
            }
        }
    }
    """
    
    generated_code = """
    #include "cctk.h"
    #include "cctk_Arguments.h"
    #include "cctk_Parameters.h"
    
    void MyThorn_InitialData(CCTK_ARGUMENTS) {
        DECLARE_CCTK_ARGUMENTS;
        DECLARE_CCTK_PARAMETERS;
        
        for (int k = 0; k < cctk_lsh[2]; k++) {
            for (int j = 0; j < cctk_lsh[1]; j++) {
                for (int i = 0; i < cctk_lsh[0]; i++) {
                    int index = CCTK_GFINDEX3D(cctkGH, i, j, k);
                    phi[index] = 1.0;
                }
            }
        }
    }
    """
    
    # Test identical code
    identical_bleu = compute_bleu_score(reference_code, reference_code)
    print(f"Identical code BLEU: {identical_bleu:.4f}")
    
    # Test similar code
    similar_bleu = compute_bleu_score(generated_code, reference_code)
    print(f"Similar code BLEU: {similar_bleu:.4f}") 