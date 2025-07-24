"""
SIMPLE OLLAMA INTEGRATION FOR STEEL PLANT CASH FLOW
Direct integration without complex Python client issues
"""

import subprocess
import json
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOllamaIntegration:
    """Simple Ollama integration using direct command execution"""
    
    def __init__(self):
        import os
        self.ollama_path = os.path.expandvars(r"C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe")
        self.model_name = "mistral:7b"
        self.is_available = self._test_availability()
    
    def _test_availability(self) -> bool:
        """Test if Ollama is available"""
        try:
            result = subprocess.run(
                [self.ollama_path, "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and self.model_name in result.stdout
        except Exception as e:
            logger.error(f"❌ Ollama test failed: {e}")
            return False
    
    def categorize_transaction(self, description: str, amount: float) -> Dict[str, Any]:
        """
        Categorize transaction using Ollama
        """
        if not self.is_available:
            return {
                'category': 'Operating Activities',
                'confidence': 0.5,
                'reasoning': 'Ollama not available',
                'method': 'fallback'
            }
        
        try:
            # Create prompt
            prompt = f"""
You are a financial analyst for a steel manufacturing company. 
Categorize this transaction into one category only:

OPERATING ACTIVITIES: Steel production, manufacturing, sales, raw materials, utilities, maintenance, employee salaries, administrative costs, marketing, legal fees

INVESTING ACTIVITIES: Machinery, equipment purchases, property, building acquisitions, technology upgrades, infrastructure, business investments

FINANCING ACTIVITIES: Loans, credit lines, debt, interest payments, dividends, share capital, equity investments

Transaction: "{description}" - Amount: ₹{amount:,.2f}

Respond with only the category name: Operating Activities, Investing Activities, or Financing Activities
"""
            
            # Run Ollama
            result = subprocess.run(
                [self.ollama_path, "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                category = self._parse_category(response)
                return {
                    'category': category,
                    'confidence': 0.8,
                    'reasoning': f'Ollama AI: {response[:100]}...',
                    'method': 'ollama_ai'
                }
            else:
                logger.error(f"❌ Ollama command failed: {result.stderr}")
                return self._fallback_categorization(description, amount)
                
        except Exception as e:
            logger.error(f"❌ Ollama categorization failed: {e}")
            return self._fallback_categorization(description, amount)
    
    def _parse_category(self, response: str) -> str:
        """Parse category from Ollama response"""
        response_lower = response.lower()
        
        if 'investing' in response_lower:
            return 'Investing Activities'
        elif 'financing' in response_lower:
            return 'Financing Activities'
        else:
            return 'Operating Activities'
    
    def _fallback_categorization(self, description: str, amount: float) -> Dict[str, Any]:
        """Fallback categorization using keywords"""
        desc_lower = description.lower()
        
        # Investing keywords
        if any(word in desc_lower for word in ['machinery', 'equipment', 'plant', 'investment', 'capital', 'asset', 'property', 'building', 'construction', 'expansion', 'acquisition', 'upgrade', 'technology', 'infrastructure']):
            category = 'Investing Activities'
        # Financing keywords
        elif any(word in desc_lower for word in ['loan', 'interest', 'financing', 'debt', 'credit', 'bank', 'mortgage', 'dividend', 'share', 'stock', 'equity', 'bond', 'refinancing', 'funding']):
            category = 'Financing Activities'
        else:
            category = 'Operating Activities'
        
        return {
            'category': category,
            'confidence': 0.6,
            'reasoning': 'Fallback keyword matching',
            'method': 'fallback'
        }
    
    def enhance_transactions(self, df: pd.DataFrame, max_transactions: int = 50) -> pd.DataFrame:
        """
        Enhance transaction categorization using Ollama
        """
        if not self.is_available:
            logger.warning("⚠️ Ollama not available - using existing categorization")
            return df
        
        logger.info(f"🤖 Enhancing {min(len(df), max_transactions)} transactions with Ollama AI...")
        
        enhanced_df = df.copy()
        enhanced_count = 0
        
        # Process limited number of transactions for performance
        for idx, row in df.head(max_transactions).iterrows():
            description = str(row.get('Description', ''))
            amount = float(row.get('Amount', 0))
            
            result = self.categorize_transaction(description, amount)
            
            # Update category if confidence is high
            if result['confidence'] > 0.7:
                enhanced_df.at[idx, 'Category'] = result['category']
                enhanced_df.at[idx, 'AI_Confidence'] = result['confidence']
                enhanced_df.at[idx, 'AI_Reasoning'] = result['reasoning']
                enhanced_count += 1
            
            # Progress update
            if (idx + 1) % 10 == 0:
                logger.info(f"   📊 Processed {idx + 1}/{min(len(df), max_transactions)} transactions...")
        
        logger.info(f"✅ Enhanced {enhanced_count} transactions with Ollama AI")
        return enhanced_df

# Global instance
simple_ollama = SimpleOllamaIntegration()

def test_simple_ollama():
    """Test simple Ollama integration"""
    print("🧪 Testing Simple Ollama Integration...")
    
    if not simple_ollama.is_available:
        print("❌ Ollama not available")
        return False
    
    # Test categorization
    test_cases = [
        ("Steel coil production - blast furnace maintenance", 1500000.0),
        ("New machinery purchase for rolling mill", 5000000.0),
        ("Bank loan repayment", 2000000.0),
        ("Employee salary payment", 500000.0)
    ]
    
    for description, amount in test_cases:
        result = simple_ollama.categorize_transaction(description, amount)
        print(f"✅ {description}: {result['category']} (confidence: {result['confidence']})")
    
    return True

if __name__ == "__main__":
    test_simple_ollama() 