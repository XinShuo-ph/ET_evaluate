import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from typing import Dict, Tuple

class LLMCodeJudge:
    """LLM-based judge for evaluating code quality using Qwen 1.5B model"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", device="cuda"):
        """Initialize the LLM judge with the specified model"""
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"LLM Judge loaded: {model_name} on {self.device}")
    
    def create_judge_prompt(self, generated_code: str, reference_code: str) -> str:
        """Create a prompt for the LLM judge to evaluate code quality"""
        prompt = f"""<|system|>
You are an expert code reviewer specializing in EinsteinToolkit code. Evaluate the generated code against the reference implementation.

<|user|>
Please evaluate the GENERATED CODE compared to the REFERENCE CODE on a scale of 0-10 for overall quality.

Consider:
- Compilation: Does it compile?
- Correctness: Does it implement the same functionality?
- Numerical Accuracy: Does it match the reference code if provided the same input?


REFERENCE CODE:
```
{reference_code.strip()}
```

GENERATED CODE:
```
{generated_code.strip()}
```

Respond with just a single number from 0-10 representing the overall quality score.

<|assistant|>
Score: """
        return prompt
    
    def extract_score(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        
        if numbers:
            score = float(numbers[0])
            # Normalize to 0-1 range
            return min(max(score / 10.0, 0.0), 1.0)
            
        # print("Fallback to manually get score patterns")
        # Fallback: look for common score patterns
        if "10" in response or "excellent" in response.lower():
            return 1.0
        elif "9" in response or "very good" in response.lower():
            return 0.9
        elif "8" in response or "good" in response.lower():
            return 0.8
        elif "7" in response:
            return 0.7
        elif "6" in response:
            return 0.6
        elif "5" in response or "average" in response.lower():
            return 0.5
        elif "poor" in response.lower() or "bad" in response.lower():
            return 0.2
        else:
            return 0.5  # Default neutral score
    
    def judge_code(self, generated_code: str, reference_code: str) -> Dict[str, float]:
        """
        Judge the generated code against reference code
        
        Args:
            generated_code: Generated code string
            reference_code: Reference code string
            
        Returns:
            Dictionary with judge scores and metadata
        """
        if not generated_code or not reference_code:
            return {'llm_judge_score': 0.0, 'raw_response': '', 'score_confidence': 0.0}
        
        # Create judge prompt
        prompt = self.create_judge_prompt(generated_code, reference_code)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Short response for score
                temperature=0.1,    # Low temperature for consistent scoring
                # do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        judge_response = response[len(prompt):].strip()
        
        # Extract score
        score = self.extract_score(judge_response)
        
        # Estimate confidence based on response clarity
        confidence = 1.0 if any(char.isdigit() for char in judge_response) else 0.5
        
        return {
            'llm_judge_score': score,
            'raw_response': judge_response,
            'score_confidence': confidence
        }

def test_llm_judge():
    """Test the LLM judge with sample code"""
    
    # Initialize judge
    judge = LLMCodeJudge()
    
    # Sample EinsteinToolkit-style code
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
    
    # Test cases
    test_cases = [
        # Good code (similar to reference)
        {
            "name": "Good code",
            "code": """
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
        },
        # Poor code (missing important parts)
        {
            "name": "Poor code",
            "code": """
            void MyThorn_InitialData() {
                for (int i = 0; i < 10; i++) {
                    phi[i] = 1.0;
                }
            }
            """
        }
    ]
    
    print("Testing LLM Judge:")
    print("="*50)
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        result = judge.judge_code(test_case['code'], reference_code)
        print(f"  Score: {result['llm_judge_score']:.3f}")
        print(f"  Confidence: {result['score_confidence']:.3f}")
        print(f"  Raw Response: {result['raw_response']}")
    
    return judge

if __name__ == "__main__":
    test_llm_judge() 