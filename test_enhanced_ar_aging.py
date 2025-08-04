import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem
import json

def test_enhanced_ar_aging():
    """
    Test the enhanced AR aging functionality with sample data
    """
    print("🧪 Testing Enhanced AR Aging Functionality...")
    
    # Create sample data
    np.random.seed(42)  # For reproducibility
    
    # Create dates spanning multiple months
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-06-30')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate 100 random transactions
    n_transactions = 100
    random_dates = np.random.choice(date_range, size=n_transactions)
    
    # Create random amounts with mostly positive values (receivables)
    amounts = np.random.normal(loc=10000, scale=5000, size=n_transactions)
    
    # Create descriptions with customer information
    customers = ['Customer_A', 'Customer_B', 'Customer_C', 'Customer_D', 'Customer_E']
    descriptions = [f"Invoice {np.random.randint(1000, 9999)} - {np.random.choice(customers)}" for _ in range(n_transactions)]
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'Date': random_dates,
        'Description': descriptions,
        'Amount': amounts,
        'Type': ['INWARD' if amt > 0 else 'OUTWARD' for amt in amounts]
    })
    
    # Initialize the AI system
    ai_system = AdvancedRevenueAISystem()
    
    # Run the enhanced AR aging analysis
    print("📊 Running enhanced AR aging analysis...")
    results = ai_system.enhanced_analyze_ar_aging(df)
    
    # Check if the analysis was successful
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    # Check if advanced AI features are present
    if 'advanced_ai_features' not in results:
        print("❌ Advanced AI features not found in results")
        return False
    
    # Print the advanced features
    print("\n✅ Advanced AI features found:")
    for feature, data in results['advanced_ai_features'].items():
        print(f"  - {feature}: {type(data).__name__}")
    
    # Print some key metrics
    print("\n📊 Key AR Aging Metrics:")
    print(f"  - DSO Days: {results.get('dso_days', 'N/A')}")
    print(f"  - Collection Probability: {results.get('weighted_collection_probability', 'N/A')}")
    print(f"  - Total Receivables: {results.get('total_receivables', 'N/A')}")
    
    # Print advanced metrics
    print("\n🤖 Advanced AI Metrics:")
    
    # Collection Optimization
    if 'collection_optimization' in results['advanced_ai_features']:
        co = results['advanced_ai_features']['collection_optimization']
        print(f"  - Potential Savings: ₹{co.get('potential_savings', 0):,.2f}")
        print(f"  - Optimal Allocation: {co.get('optimal_allocation', {})}")
    
    # Customer Segmentation
    if 'customer_segmentation' in results['advanced_ai_features']:
        cs = results['advanced_ai_features']['customer_segmentation']
        print(f"  - Cluster Count: {cs.get('cluster_count', 0)}")
        if 'clusters' in cs:
            print(f"  - Cluster Types: {[c.get('type') for c in cs['clusters'].values()]}")
    
    # Payment Prediction
    if 'payment_prediction' in results['advanced_ai_features']:
        pp = results['advanced_ai_features']['payment_prediction']
        print(f"  - Next 3 Months Forecast: {pp.get('next_3_months', [])}")
        print(f"  - Model Accuracy: {pp.get('model_accuracy', 0) * 100:.1f}%")
    
    # Risk Assessment
    if 'risk_assessment' in results['advanced_ai_features']:
        ra = results['advanced_ai_features']['risk_assessment']
        print(f"  - Overall Risk Level: {ra.get('overall_risk_level', 'Unknown')}")
        print(f"  - Risk Score: {ra.get('overall_risk_score', 0):.1f}/100")
    
    print("\n✅ Enhanced AR Aging test completed successfully!")
    return True

if __name__ == "__main__":
    test_enhanced_ar_aging()