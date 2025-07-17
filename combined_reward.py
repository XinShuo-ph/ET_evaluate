from bleu_reward import compute_bleu_score
from llm_judge import LLMCodeJudge
from typing import Dict

class CombinedReward:
    """Combines BLEU score and LLM judge for comprehensive code evaluation"""
    
    def __init__(self, bleu_weight=0.6, llm_weight=0.4, use_llm_judge=True):
        """
        Initialize combined reward system
        
        Args:
            bleu_weight: Weight for BLEU score (0-1)
            llm_weight: Weight for LLM judge score (0-1)
            use_llm_judge: Whether to use LLM judge (can disable for faster training)
        """
        self.bleu_weight = bleu_weight
        self.llm_weight = llm_weight
        self.use_llm_judge = use_llm_judge
        
        # Normalize weights
        total_weight = bleu_weight + (llm_weight if use_llm_judge else 0)
        if use_llm_judge:
            self.bleu_weight = bleu_weight / total_weight
            self.llm_weight = llm_weight / total_weight
        else:
            self.bleu_weight = 1.0
            self.llm_weight = 0.0
        
        # Initialize LLM judge if enabled
        self.llm_judge = LLMCodeJudge() if use_llm_judge else None
        
        print(f"CombinedReward: BLEU weight={self.bleu_weight:.3f}, LLM Judge weight={self.llm_weight:.3f}, LLM_enabled={use_llm_judge}")
    
    def get_reward(self, generated_code: str, reference_code: str) -> Dict[str, float]:
        """
        Compute combined reward for generated code
        
        Args:
            generated_code: Generated code string
            reference_code: Reference code string
            
        Returns:
            Dictionary with all reward components and final score
        """
        # Get BLEU score
        bleu_score = compute_bleu_score(generated_code, reference_code)
        
        # Get LLM judge score
        if self.use_llm_judge and self.llm_judge:
            llm_result = self.llm_judge.judge_code(generated_code, reference_code)
            llm_score = llm_result['llm_judge_score']
            llm_confidence = llm_result['score_confidence']
            llm_response = llm_result['raw_response']
        else:
            llm_score = 0.0
            llm_confidence = 0.0
            llm_response = "LLM judge disabled"
        
        # Combine scores
        combined_score = (
            self.bleu_weight * bleu_score + 
            self.llm_weight * llm_score
        )
        
        return {
            'bleu_score': bleu_score,
            'llm_judge_score': llm_score,
            'llm_confidence': llm_confidence,
            'llm_response': llm_response,
            'combined_reward': combined_score
        }

def test_combined_reward():
    """Test the combined reward system"""
    
    # Test with different configurations
    configs = [
        {"name": "Balanced", "bleu_weight": 0.6, "llm_weight": 0.4, "use_llm": True},
        {"name": "BLEU-heavy", "bleu_weight": 0.8, "llm_weight": 0.2, "use_llm": True},
        {"name": "BLEU-only", "bleu_weight": 1.0, "llm_weight": 0.0, "use_llm": False}
    ]
    
    # Sample codes
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
    
    good_code = """
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
    
    print("Testing Combined Reward System:")
    print("="*60)
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print("-" * 30)
        
            
        reward_system = CombinedReward(
            bleu_weight=config['bleu_weight'],
            llm_weight=config['llm_weight'],
            use_llm_judge=config['use_llm']
        )
        
        result = reward_system.get_reward(good_code, reference_code)
        
        print(f"BLEU Score: {result['bleu_score']:.4f}")
        if config['use_llm']:
            print(f"LLM Score: {result['llm_judge_score']:.4f}")
            print(f"LLM Response: {result['llm_response'][:50]}...")
        print(f"Combined Reward: {result['combined_reward']:.4f}")

if __name__ == "__main__":
    test_combined_reward() 