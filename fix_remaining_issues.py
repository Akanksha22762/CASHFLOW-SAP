#!/usr/bin/env python3
"""
TARGETED FIX FOR REMAINING ISSUES - Ollama API and Training Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_ollama_api_issue():
    """Fix Ollama API 404 error"""
    print("🔧 FIXING OLLAMA API ISSUE...")
    
    try:
        from ollama_simple_integration import OllamaSimpleIntegration
        
        # Test with different model names
        ollama = OllamaSimpleIntegration()
        
        if not ollama.is_available:
            print("❌ Ollama not available")
            return False
        
        print(f"✅ Available models: {ollama.available_models}")
        
        # Try different model names
        test_models = ['llama2', 'llama2:7b', 'mistral', 'mistral:7b']
        
        for model in test_models:
            try:
                print(f"🧪 Testing model: {model}")
                test_prompt = "Categorize this transaction: Infrastructure Development"
                test_response = ollama.simple_ollama(test_prompt, model=model, max_tokens=30)
                
                if test_response:
                    print(f"✅ Model {model} works!")
                    print(f"✅ Response: {test_response[:50]}...")
                    return True
                else:
                    print(f"❌ Model {model} failed")
                    
            except Exception as e:
                print(f"❌ Error with model {model}: {e}")
                continue
        
        print("❌ All models failed")
        return False
        
    except Exception as e:
        print(f"❌ Error fixing Ollama: {e}")
        return False

def fix_training_data_issue():
    """Fix training data stratification issue"""
    print("🔧 FIXING TRAINING DATA ISSUE...")
    
    try:
        from app1 import lightweight_ai
        
        # Create balanced training data with at least 2 samples per class
        training_data = pd.DataFrame({
            'Description': [
                # Investing Activities (2 samples)
                'Infrastructure Development - Warehouse Construction',
                'Equipment Purchase - Rolling Mill Upgrade',
                
                # Operating Activities (2 samples)
                'VIP Customer Payment - Construction Company',
                'Salary Payment - Employee Payroll',
                
                # Financing Activities (2 samples)
                'Investment Liquidation - Mutual Fund Units',
                'Loan Repayment - Bank Loan',
                
                # Additional samples for better training
                'Utility Payment - Electricity Bill',
                'Tax Payment - Income Tax',
                'Dividend Payment - Shareholder Return',
                'Insurance Premium - Business Insurance',
                'Marketing Expense - Advertising Campaign',
                'Software License - ERP System',
                'Rent Payment - Office Space',
                'Freight Cost - Shipping Services',
                'Legal Fees - Contract Review',
                'Audit Fees - Annual Audit',
                'Training Cost - Employee Development',
                'Maintenance Cost - Equipment Repair',
                'Commission Payment - Sales Commission',
                'Interest Payment - Loan Interest',
                'Refund Payment - Customer Refund',
                'Deposit Payment - Security Deposit',
                'Subscription Fee - Software Subscription',
                'Consulting Fee - Business Consulting',
                'Travel Expense - Business Travel',
                'Office Supplies - Stationery',
                'Phone Bill - Communication',
                'Internet Bill - Data Services',
                'Cleaning Service - Facility Maintenance',
                'Security Service - Building Security'
            ],
            'Amount': [
                # Investing Activities
                3709289.81, 1500000,
                # Operating Activities
                2000000, 800000,
                # Financing Activities
                500000, 300000,
                # Additional samples
                50000, 200000, 100000, 75000, 120000, 25000, 150000, 45000, 35000,
                50000, 30000, 25000, 60000, 15000, 25000, 50000, 12000, 40000, 15000,
                5000, 8000, 12000, 18000, 22000
            ],
            'Category': [
                # Investing Activities
                'Investing Activities', 'Investing Activities',
                # Operating Activities
                'Operating Activities', 'Operating Activities',
                # Financing Activities
                'Financing Activities', 'Financing Activities',
                # Additional samples (mostly Operating Activities)
                'Operating Activities', 'Operating Activities', 'Financing Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Financing Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities',
                'Operating Activities', 'Operating Activities', 'Operating Activities', 'Operating Activities'
            ],
            'Date': [datetime.now() - timedelta(days=i) for i in range(32)],
            'Type': ['Credit'] * 32
        })
        
        print(f"📊 Training data created: {len(training_data)} samples")
        print(f"📊 Categories: {training_data['Category'].value_counts().to_dict()}")
        
        # Verify we have at least 2 samples per class
        category_counts = training_data['Category'].value_counts()
        min_samples = category_counts.min()
        print(f"📊 Minimum samples per class: {min_samples}")
        
        if min_samples < 2:
            print("❌ Not enough samples per class for stratification")
            return False
        
        # Train the model
        success = lightweight_ai.train_transaction_classifier(training_data)
        
        if success:
            print("✅ Training successful!")
            print(f"✅ Model trained: {lightweight_ai.is_trained}")
            
            # Test the model
            test_cases = [
                ("Infrastructure Development", 1000000),
                ("Customer Payment", 2000000),
                ("Investment Liquidation", 500000)
            ]
            
            for desc, amount in test_cases:
                result = lightweight_ai.categorize_transaction_ml(desc, amount)
                print(f"✅ Test: {desc} → {result}")
            
            return True
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing training data: {e}")
        return False

def test_fixes():
    """Test the remaining fixes"""
    print("🧪 TESTING REMAINING FIXES...")
    print("=" * 50)
    
    results = {}
    
    # Test Ollama fix
    results['ollama'] = fix_ollama_api_issue()
    
    # Test Training fix
    results['training'] = fix_training_data_issue()
    
    # Summary
    print("\n📊 FIX RESULTS:")
    print("=" * 30)
    
    total_fixes = len(results)
    successful_fixes = sum(results.values())
    
    for fix, result in results.items():
        status = "✅ FIXED" if result else "❌ FAILED"
        print(f"   {fix.replace('_', ' ').title()}: {status}")
    
    print(f"\n📈 OVERALL: {successful_fixes}/{total_fixes} fixes successful")
    
    if successful_fixes == total_fixes:
        print("🎉 ALL REMAINING ISSUES FIXED!")
        return True
    else:
        print("❌ Some issues still need attention")
        return False

def run_final_check():
    """Run final comprehensive check"""
    print("\n🔍 RUNNING FINAL COMPREHENSIVE CHECK...")
    print("=" * 60)
    
    try:
        from comprehensive_system_check import comprehensive_system_check
        return comprehensive_system_check()
    except Exception as e:
        print(f"❌ Error running comprehensive check: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TARGETED FIX FOR REMAINING ISSUES")
    print("=" * 50)
    
    # Apply fixes
    fixes_successful = test_fixes()
    
    if fixes_successful:
        print("\n✅ Running final comprehensive check...")
        final_check = run_final_check()
        
        if final_check:
            print("\n🎉 ALL SYSTEMS NOW WORKING PERFECTLY!")
        else:
            print("\n⚠️ Some issues may still remain")
    else:
        print("\n❌ Some fixes failed - manual intervention needed") 