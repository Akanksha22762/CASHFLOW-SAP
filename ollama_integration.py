"""
OLLAMA INTEGRATION FOR STEEL PLANT CASH FLOW SYSTEM
Advanced AI-powered transaction analysis and categorization
"""

import ollama
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration"""
    model_name: str = "mistral:7b"
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.1  # Low temperature for consistent results

class OllamaTransactionAnalyzer:
    """Advanced AI-powered transaction analyzer using Ollama"""
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.client = None
        self.is_available = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client"""
        try:
            # Test connection to Ollama
            response = ollama.list()
            logger.info("✅ Ollama connection successful!")
            self.is_available = True
            
            # Check if our model is available
            models = []
            if 'models' in response:
                models = [model.get('name', '') for model in response['models'] if model.get('name')]
            
            logger.info(f"📋 Available models: {models}")
            
            if self.config.model_name not in models:
                logger.warning(f"⚠️ Model {self.config.model_name} not found. Available models: {models}")
                # Try to use any available model
                if models:
                    self.config.model_name = models[0]
                    logger.info(f"🔄 Using available model: {self.config.model_name}")
                else:
                    logger.error("❌ No models available in Ollama")
                    self.is_available = False
            else:
                logger.info(f"✅ Model {self.config.model_name} is available")
                
        except Exception as e:
            logger.error(f"❌ Ollama connection failed: {e}")
            self.is_available = False
    
    def categorize_transaction(self, description: str, amount: float) -> Dict[str, Any]:
        """
        Categorize transaction using Ollama AI
        """
        if not self.is_available:
            return {
                'category': 'Operating Activities',
                'confidence': 0.5,
                'reasoning': 'Ollama not available - using fallback',
                'method': 'fallback'
            }
        
        try:
            # Create steel plant specific prompt
            prompt = self._create_categorization_prompt(description, amount)
            
            # Get AI response
            response = ollama.chat(
                model=self.config.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': self.config.temperature,
                    'num_predict': 200
                }
            )
            
            # Parse response
            result = self._parse_categorization_response(response['message']['content'])
            result['method'] = 'ollama_ai'
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ollama categorization failed: {e}")
            return {
                'category': 'Operating Activities',
                'confidence': 0.3,
                'reasoning': f'Error: {str(e)}',
                'method': 'error_fallback'
            }
    
    def _create_categorization_prompt(self, description: str, amount: float) -> str:
        """Create steel plant specific categorization prompt"""
        return f"""
You are an expert financial analyst specializing in steel manufacturing transactions. 
Analyze this transaction and categorize it into one of these categories:

OPERATING ACTIVITIES:
- Steel production, manufacturing, sales
- Raw materials, utilities, maintenance
- Employee salaries, administrative costs
- Marketing, legal, consulting fees

INVESTING ACTIVITIES:
- Machinery, equipment purchases
- Property, building acquisitions
- Technology upgrades, infrastructure
- Business investments, acquisitions

FINANCING ACTIVITIES:
- Loans, credit lines, debt
- Interest payments, dividends
- Share capital, equity investments
- Financial instruments

Transaction Details:
- Description: "{description}"
- Amount: ₹{amount:,.2f}

Provide your response in this exact JSON format:
{{
    "category": "Operating Activities|Investing Activities|Financing Activities",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your categorization decision"
}}

Focus on steel manufacturing context and be precise.
"""
    
    def _parse_categorization_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                result = json.loads(json_str)
                
                # Validate category
                valid_categories = [
                    'Operating Activities', 'Investing Activities', 'Financing Activities'
                ]
                
                if result.get('category') not in valid_categories:
                    result['category'] = 'Operating Activities'
                
                # Ensure confidence is between 0 and 1
                confidence = result.get('confidence', 0.5)
                result['confidence'] = max(0.0, min(1.0, float(confidence)))
                
                return result
            else:
                # Fallback parsing
                return self._fallback_parsing(response)
                
        except Exception as e:
            logger.error(f"❌ Failed to parse Ollama response: {e}")
            return self._fallback_parsing(response)
    
    def _fallback_parsing(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON responses"""
        response_lower = response.lower()
        
        # Simple keyword matching
        if any(word in response_lower for word in ['investing', 'investment', 'equipment', 'machinery']):
            category = 'Investing Activities'
        elif any(word in response_lower for word in ['financing', 'loan', 'debt', 'dividend']):
            category = 'Financing Activities'
        else:
            category = 'Operating Activities'
        
        return {
            'category': category,
            'confidence': 0.6,
            'reasoning': f'Fallback parsing: {response[:100]}...'
        }
    
    def analyze_transaction_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze patterns in transaction data using Ollama
        """
        if not self.is_available:
            return {'error': 'Ollama not available'}
        
        try:
            # Create analysis prompt
            prompt = self._create_pattern_analysis_prompt(transactions)
            
            response = ollama.chat(
                model=self.config.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.2,
                    'num_predict': 500
                }
            )
            
            return self._parse_pattern_analysis(response['message']['content'])
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def _create_pattern_analysis_prompt(self, transactions: List[Dict]) -> str:
        """Create prompt for pattern analysis"""
        # Sample transactions for analysis
        sample_data = transactions[:10]  # Use first 10 for analysis
        
        return f"""
You are a financial analyst for a steel manufacturing company. 
Analyze these recent transactions and identify patterns:

Transactions:
{json.dumps(sample_data, indent=2)}

Provide analysis in JSON format:
{{
    "cash_flow_trend": "increasing|decreasing|stable",
    "main_categories": ["category1", "category2"],
    "seasonal_patterns": "description",
    "risk_factors": ["risk1", "risk2"],
    "recommendations": ["rec1", "rec2"]
}}
"""
    
    def _parse_pattern_analysis(self, response: str) -> Dict[str, Any]:
        """Parse pattern analysis response"""
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {'analysis': response}
        except Exception as e:
            return {'error': f'Failed to parse analysis: {e}'}

class OllamaCashFlowEnhancer:
    """Enhanced cash flow analysis using Ollama AI"""
    
    def __init__(self):
        self.analyzer = OllamaTransactionAnalyzer()
        self.cache = {}
    
    def enhance_categorization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance transaction categorization using Ollama AI
        """
        if not self.analyzer.is_available:
            logger.warning("⚠️ Ollama not available - using existing categorization")
            return df
        
        logger.info("🤖 Enhancing categorization with Ollama AI...")
        
        enhanced_df = df.copy()
        enhanced_count = 0
        
        for idx, row in df.iterrows():
            description = str(row.get('Description', ''))
            amount = float(row.get('Amount', 0))
            
            # Create cache key
            cache_key = f"{description}_{amount}"
            
            if cache_key in self.cache:
                result = self.cache[cache_key]
            else:
                result = self.analyzer.categorize_transaction(description, amount)
                self.cache[cache_key] = result
            
            # Update category if confidence is high enough
            if result['confidence'] > 0.7:
                enhanced_df.at[idx, 'Category'] = result['category']
                enhanced_df.at[idx, 'AI_Confidence'] = result['confidence']
                enhanced_df.at[idx, 'AI_Reasoning'] = result['reasoning']
                enhanced_count += 1
            
            # Progress update
            if (idx + 1) % 50 == 0:
                logger.info(f"   📊 Processed {idx + 1}/{len(df)} transactions...")
        
        logger.info(f"✅ Enhanced {enhanced_count} transactions with Ollama AI")
        return enhanced_df
    
    def generate_ai_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate AI-powered insights from transaction data
        """
        if not self.analyzer.is_available:
            return {'error': 'Ollama not available'}
        
        try:
            # Convert DataFrame to list of dicts
            transactions = df.to_dict('records')
            
            # Get pattern analysis
            patterns = self.analyzer.analyze_transaction_patterns(transactions)
            
            # Generate additional insights
            insights = {
                'pattern_analysis': patterns,
                'total_transactions': len(df),
                'ai_enhanced': True,
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate AI insights: {e}")
            return {'error': str(e)}

# Global instances
ollama_analyzer = OllamaTransactionAnalyzer()
ollama_enhancer = OllamaCashFlowEnhancer()

def test_ollama_connection():
    """Test Ollama connection and functionality"""
    print("🧪 Testing Ollama Integration...")
    
    if not ollama_analyzer.is_available:
        print("❌ Ollama not available")
        return False
    
    # Test categorization
    test_description = "Steel coil production - blast furnace maintenance"
    test_amount = 1500000.0
    
    result = ollama_analyzer.categorize_transaction(test_description, test_amount)
    
    print(f"✅ Test categorization result:")
    print(f"   Category: {result['category']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Method: {result['method']}")
    
    return True

if __name__ == "__main__":
    test_ollama_connection() 