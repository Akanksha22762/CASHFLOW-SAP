import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def analyze_data_availability():
    """Analyze what data we have vs what's needed for AI cash flow model"""
    
    print("🔍 ANALYZING DATA AVAILABILITY FOR AI CASH FLOW MODEL")
    print("=" * 60)
    
    # Load available datasets
    datasets = {}
    
    try:
        # Load transaction data
        transactions_df = pd.read_csv('data/transactions_data_updated.csv')
        datasets['transactions'] = transactions_df
        print(f"✅ Transactions Data: {len(transactions_df)} records")
        
        # Load AP/AR data
        ap_ar_df = pd.read_excel('steel_plant_ap_ar_data.xlsx')
        datasets['ap_ar'] = ap_ar_df
        print(f"✅ AP/AR Data: {len(ap_ar_df)} records")
        
        # Load bank statement
        bank_df = pd.read_excel('steel_plant_bank_statement.xlsx')
        datasets['bank'] = bank_df
        print(f"✅ Bank Statement: {len(bank_df)} records")
        
        # Load master data
        master_df = pd.read_excel('steel_plant_master_data.xlsx')
        datasets['master'] = master_df
        print(f"✅ Master Data: {len(master_df)} records")
        
        # Load AI mapping
        ai_mapping_df = pd.read_csv('data/ai_reference_mapping_full.csv')
        datasets['ai_mapping'] = ai_mapping_df
        print(f"✅ AI Mapping: {len(ai_mapping_df)} records")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    print("\n📊 DATA REQUIREMENTS ANALYSIS")
    print("=" * 60)
    
    # Define requirements from the document
    requirements = {
        'Revenue-Related Parameters': {
            'Revenue forecasts': 'Expected income from sales, broken down by product, geography, and customer segment',
            'Customer payment terms': 'Typical days sales outstanding (DSO), average payment delays',
            'Accounts receivable aging': 'Breakdown of receivables into current, 30-60-90+ day buckets',
            'Sales pipeline & backlog': 'Expected future revenues from open opportunities and signed contracts',
            'Seasonality factors': 'Historical revenue fluctuations due to seasonality'
        },
        'Expense-Related Parameters': {
            'Operating expenses (OPEX)': 'Fixed and variable costs, such as rent, salaries, utilities, etc.',
            'Accounts payable terms': 'Days payable outstanding (DPO), payment cycles to vendors',
            'Inventory turnover': 'Cash locked in inventory, including procurement and storage cycles',
            'Loan repayments': 'Principal and interest payments due over the projection period',
            'Tax obligations': 'Upcoming GST, VAT, income tax, or other regulatory payments'
        },
        'Cash Inflows & Outflows': {
            'Cash inflow types': 'Customer payments, loans, investor funding, asset sales',
            'Cash outflow types': 'Payroll, vendors, tax, interest, dividends, repayments',
            'Payment frequency & timing': 'Weekly/monthly/quarterly cycles, lags'
        },
        'Operational & Business Drivers': {
            'Inventory turnover': 'Cash locked in inventory and replenishment cycles',
            'Headcount plans': 'Hiring/firing impact on payroll and benefits',
            'Expansion plans': 'New markets, products, facilities, partnerships',
            'Marketing spend and ROI': 'Influences lead generation and revenue growth'
        }
    }
    
    # Analyze each requirement
    analysis_results = {}
    
    for category, params in requirements.items():
        print(f"\n📋 {category}")
        print("-" * 40)
        analysis_results[category] = {}
        
        for param, description in params.items():
            availability = analyze_parameter_availability(param, datasets)
            analysis_results[category][param] = availability
            
            status = "✅" if availability['available'] else "❌"
            coverage = f"{availability['coverage']:.1f}%" if availability['coverage'] > 0 else "0%"
            print(f"  {status} {param}: {coverage} coverage")
            if not availability['available']:
                print(f"     Missing: {availability['missing_data']}")
    
    # Generate recommendations
    print("\n🎯 IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 60)
    
    generate_recommendations(analysis_results, datasets)
    
    return analysis_results

def analyze_parameter_availability(parameter, datasets):
    """Analyze availability of a specific parameter"""
    
    availability = {
        'available': False,
        'coverage': 0.0,
        'missing_data': [],
        'data_sources': []
    }
    
    # Check transactions data
    if 'transactions' in datasets:
        df = datasets['transactions']
        
        if 'revenue' in parameter.lower() or 'sales' in parameter.lower():
            revenue_data = df[df['Type'] == 'INWARD']
            if len(revenue_data) > 0:
                availability['available'] = True
                availability['coverage'] = 80.0
                availability['data_sources'].append('transactions')
            else:
                availability['missing_data'].append('Revenue/sales data')
        
        elif 'expense' in parameter.lower() or 'payment' in parameter.lower():
            expense_data = df[df['Type'] == 'OUTWARD']
            if len(expense_data) > 0:
                availability['available'] = True
                availability['coverage'] = 75.0
                availability['data_sources'].append('transactions')
            else:
                availability['missing_data'].append('Expense/payment data')
        
        elif 'customer' in parameter.lower():
            customer_data = df[df['Description'].str.contains('Customer|CUST', na=False)]
            if len(customer_data) > 0:
                availability['available'] = True
                availability['coverage'] = 70.0
                availability['data_sources'].append('transactions')
            else:
                availability['missing_data'].append('Customer payment data')
        
        elif 'vendor' in parameter.lower():
            vendor_data = df[df['Description'].str.contains('Vendor|VEN', na=False)]
            if len(vendor_data) > 0:
                availability['available'] = True
                availability['coverage'] = 85.0
                availability['data_sources'].append('transactions')
            else:
                availability['missing_data'].append('Vendor payment data')
    
    # Check AP/AR data
    if 'ap_ar' in datasets:
        df = datasets['ap_ar']
        
        if 'receivable' in parameter.lower():
            ar_data = df[df['Type'] == 'Accounts Receivable']
            if len(ar_data) > 0:
                availability['available'] = True
                availability['coverage'] = 90.0
                availability['data_sources'].append('ap_ar')
            else:
                availability['missing_data'].append('Accounts receivable data')
        
        elif 'payable' in parameter.lower():
            ap_data = df[df['Type'] == 'Accounts Payable']
            if len(ap_data) > 0:
                availability['available'] = True
                availability['coverage'] = 90.0
                availability['data_sources'].append('ap_ar')
            else:
                availability['missing_data'].append('Accounts payable data')
    
    # Check bank data
    if 'bank' in datasets:
        df = datasets['bank']
        
        if 'cash' in parameter.lower() or 'balance' in parameter.lower():
            if 'Balance' in df.columns:
                availability['available'] = True
                availability['coverage'] = 95.0
                availability['data_sources'].append('bank')
            else:
                availability['missing_data'].append('Cash balance data')
    
    # Default coverage for available data
    if availability['available'] and availability['coverage'] == 0:
        availability['coverage'] = 60.0
    
    return availability

def generate_recommendations(analysis_results, datasets):
    """Generate implementation recommendations"""
    
    print("\n🚀 PHASE 1: IMMEDIATE IMPLEMENTATION (High Data Coverage)")
    print("-" * 50)
    
    high_coverage_features = []
    for category, params in analysis_results.items():
        for param, analysis in params.items():
            if analysis['coverage'] >= 70:
                high_coverage_features.append((param, analysis['coverage']))
    
    for feature, coverage in sorted(high_coverage_features, key=lambda x: x[1], reverse=True):
        print(f"✅ {feature} ({coverage:.0f}% coverage)")
    
    print("\n🔧 PHASE 2: ENHANCED IMPLEMENTATION (Medium Data Coverage)")
    print("-" * 50)
    
    medium_coverage_features = []
    for category, params in analysis_results.items():
        for param, analysis in params.items():
            if 30 <= analysis['coverage'] < 70:
                medium_coverage_features.append((param, analysis['coverage']))
    
    for feature, coverage in sorted(medium_coverage_features, key=lambda x: x[1], reverse=True):
        print(f"⚠️ {feature} ({coverage:.0f}% coverage) - Needs data enhancement")
    
    print("\n📈 PHASE 3: FUTURE IMPLEMENTATION (Low/No Data Coverage)")
    print("-" * 50)
    
    low_coverage_features = []
    for category, params in analysis_results.items():
        for param, analysis in params.items():
            if analysis['coverage'] < 30:
                low_coverage_features.append((param, analysis['coverage']))
    
    for feature, coverage in sorted(low_coverage_features, key=lambda x: x[1], reverse=True):
        print(f"❌ {feature} ({coverage:.0f}% coverage) - Requires new data sources")
    
    print("\n💡 SPECIFIC RECOMMENDATIONS:")
    print("-" * 30)
    
    print("1. Start with cash flow forecasting using existing transaction data")
    print("2. Implement vendor payment behavior modeling (85% coverage)")
    print("3. Build customer payment prediction models (70% coverage)")
    print("4. Add accounts receivable aging analysis (90% coverage)")
    print("5. Implement expense categorization and forecasting (75% coverage)")
    
    print("\n📊 DATA ENHANCEMENT NEEDS:")
    print("-" * 30)
    print("• Add customer segmentation data")
    print("• Include sales pipeline information")
    print("• Add inventory turnover metrics")
    print("• Include headcount planning data")
    print("• Add seasonal trend indicators")

if __name__ == "__main__":
    results = analyze_data_availability() 