import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaSimpleIntegration:
    """
    Simple Ollama Integration for AI Enhancement
    Provides basic Ollama API integration for text processing and analysis
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama integration"""
        self.base_url = base_url
        self.available_models = []
        self.is_available = False
        self._check_availability()
        
    def _check_availability(self):
        """Check if Ollama is available and running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_available = True
                models_data = response.json()
                self.available_models = [model['name'] for model in models_data.get('models', [])]
                logger.info(f"✅ Ollama available with models: {self.available_models}")
            else:
                logger.warning(f"⚠️ Ollama not responding properly: {response.status_code}")
                self.is_available = False
        except Exception as e:
            logger.warning(f"⚠️ Ollama not available: {e}")
            self.is_available = False
    
    def simple_ollama(self, prompt: str, model: str = "llama2", max_tokens: int = 100) -> Optional[str]:
        """
        Simple Ollama API call for text processing
        
        Args:
            prompt: Input prompt for the model
            model: Model name to use
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        if not self.is_available:
            logger.warning("Ollama not available, skipping request")
            return None
            
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    def enhance_descriptions(self, descriptions: List[str], model: str = "llama2") -> List[str]:
        """
        Enhance transaction descriptions using Ollama
        
        Args:
            descriptions: List of transaction descriptions
            model: Model to use for enhancement
            
        Returns:
            List of enhanced descriptions
        """
        if not self.is_available:
            logger.warning("Ollama not available, returning original descriptions")
            return descriptions
            
        enhanced_descriptions = []
        
        for desc in descriptions:
            try:
                # Create enhancement prompt
                prompt = f"""
                Enhance this transaction description to be more descriptive and clear:
                Original: {desc}
                
                Enhanced description:"""
                
                enhanced = self.simple_ollama(prompt, model, max_tokens=50)
                if enhanced:
                    enhanced_descriptions.append(enhanced)
                else:
                    enhanced_descriptions.append(desc)
                    
            except Exception as e:
                logger.error(f"Error enhancing description '{desc}': {e}")
                enhanced_descriptions.append(desc)
        
        return enhanced_descriptions
    
    def categorize_transactions(self, descriptions: List[str], model: str = "llama2") -> List[str]:
        """
        Categorize transactions using Ollama
        
        Args:
            descriptions: List of transaction descriptions
            model: Model to use for categorization
            
        Returns:
            List of categories
        """
        if not self.is_available:
            logger.warning("Ollama not available, returning default categories")
            return ["Operating Activities"] * len(descriptions)
            
        categories = []
        
        for desc in descriptions:
            try:
                prompt = f"""
                Categorize this transaction into one of these cash flow categories:
                - Operating Activities (revenue, expenses, regular business operations)
                - Investing Activities (capital expenditure, asset purchases, investments)
                - Financing Activities (loans, interest, dividends, equity)
                
                Transaction: {desc}
                Category:"""
                
                category = self.simple_ollama(prompt, model, max_tokens=20)
                if category:
                    # Clean up the response
                    category = category.strip().split('\n')[0].strip()
                    if category not in ["Operating Activities", "Investing Activities", "Financing Activities"]:
                        category = "Operating Activities"
                else:
                    category = "Operating Activities"
                    
                categories.append(category)
                
            except Exception as e:
                logger.error(f"Error categorizing '{desc}': {e}")
                categories.append("Operating Activities")
        
        return categories
    
    def analyze_patterns(self, data: List[Dict[str, Any]], model: str = "llama2") -> Dict[str, Any]:
        """
        Analyze patterns in transaction data using Ollama
        
        Args:
            data: List of transaction dictionaries
            model: Model to use for analysis
            
        Returns:
            Dictionary containing pattern analysis
        """
        if not self.is_available:
            logger.warning("Ollama not available, returning basic analysis")
            return {"patterns": "Basic analysis only", "confidence": 0.5}
            
        try:
            # Prepare data summary for analysis
            total_transactions = len(data)
            total_amount = sum(float(item.get('amount', 0)) for item in data)
            avg_amount = total_amount / total_transactions if total_transactions > 0 else 0
            
            prompt = f"""
            Analyze these transaction patterns:
            - Total transactions: {total_transactions}
            - Total amount: ${total_amount:,.2f}
            - Average amount: ${avg_amount:,.2f}
            
            Provide insights about:
            1. Revenue patterns
            2. Seasonal trends
            3. Risk factors
            4. Recommendations
            
            Analysis:"""
            
            analysis = self.simple_ollama(prompt, model, max_tokens=200)
            
            return {
                "patterns": analysis if analysis else "No patterns detected",
                "confidence": 0.8 if analysis else 0.3,
                "total_transactions": total_transactions,
                "total_amount": total_amount,
                "avg_amount": avg_amount
            }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {"patterns": "Analysis failed", "confidence": 0.1, "error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of Ollama integration"""
        return {
            "available": self.is_available,
            "base_url": self.base_url,
            "available_models": self.available_models,
            "status": "healthy" if self.is_available else "unavailable"
        }

# Global instance for easy access
ollama_integration = OllamaSimpleIntegration()

def simple_ollama(prompt: str, model: str = "llama2", max_tokens: int = 100) -> Optional[str]:
    """
    Simple function to call Ollama
    
    Args:
        prompt: Input prompt
        model: Model name
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text or None
    """
    return ollama_integration.simple_ollama(prompt, model, max_tokens)

def enhance_descriptions_with_ollama(descriptions: List[str]) -> List[str]:
    """
    Enhance descriptions using Ollama
    
    Args:
        descriptions: List of descriptions to enhance
        
    Returns:
        List of enhanced descriptions
    """
    return ollama_integration.enhance_descriptions(descriptions)

def categorize_with_ollama(descriptions: List[str]) -> List[str]:
    """
    Categorize transactions using Ollama
    
    Args:
        descriptions: List of descriptions to categorize
        
    Returns:
        List of categories
    """
    return ollama_integration.categorize_transactions(descriptions)

def analyze_patterns_with_ollama(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns using Ollama
    
    Args:
        data: Transaction data to analyze
        
    Returns:
        Pattern analysis results
    """
    return ollama_integration.analyze_patterns(data)

def check_ollama_availability():
    """Check if Ollama is available and working"""
    try:
        import httpx
        response = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False

# Test function
def test_ollama_integration():
    """Test the Ollama integration"""
    print("Testing Ollama integration...")
    
    # Test availability
    status = ollama_integration.get_health_status()
    print(f"Ollama status: {status}")
    
    # Test simple call
    if ollama_integration.is_available:
        result = simple_ollama("Hello, how are you?", max_tokens=20)
        print(f"Test response: {result}")
    else:
        print("Ollama not available for testing")

if __name__ == "__main__":
    test_ollama_integration() 