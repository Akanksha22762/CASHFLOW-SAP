import requests
import json

def test_transaction_analysis():
    """Test the transaction analysis endpoint to see the response structure"""
    
    # Test data
    test_data = {
        'transaction_type': 'Investing Activities (XGBoost)',
        'analysis_type': 'cash_flow',
        'ai_model': 'hybrid'
    }
    
    try:
        print("🔍 Testing transaction analysis endpoint...")
        response = requests.post('http://127.0.0.1:5000/transaction-analysis', 
                               json=test_data, 
                               timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received successfully!")
            print(f"📊 Response keys: {list(result.keys())}")
            print(f"📊 Success: {result.get('success')}")
            print(f"📊 AI Model: {result.get('ai_model')}")
            print(f"📊 Transactions Analyzed: {result.get('transactions_analyzed')}")
            print(f"📊 Analysis Type: {result.get('analysis_type')}")
            
            if 'data' in result:
                data = result['data']
                print(f"📊 Data keys: {list(data.keys())}")
                print(f"📊 Transaction Count: {data.get('transaction_count')}")
                print(f"📊 Total Amount: {data.get('total_amount')}")
                print(f"📊 Avg Amount: {data.get('avg_amount')}")
                print(f"📊 Insights: {data.get('insights', '')[:100]}...")
            else:
                print("❌ No 'data' field in response")
                
        else:
            print(f"❌ Error response: {response.status_code}")
            print(f"❌ Error text: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_transaction_analysis() 