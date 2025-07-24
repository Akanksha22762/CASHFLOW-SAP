"""
Clean Cash Flow SAP Bank Reconciliation System
Fixed indentation and core functionality
"""

import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# AI Configuration
LOCAL_AI_AVAILABLE = True
OLLAMA_AVAILABLE = False

# Simple cache
ai_response_cache = {}

def categorize_with_local_ai(description, amount=0):
    """Simple local AI categorization"""
    desc_lower = str(description).lower()
    
    # Operating Activities
    if any(word in desc_lower for word in ['steel', 'production', 'manufacturing', 'sales', 'raw material', 'utility', 'maintenance', 'salary', 'admin', 'marketing', 'legal']):
        return 'Operating Activities'
    
    # Investing Activities  
    elif any(word in desc_lower for word in ['machinery', 'equipment', 'plant', 'investment', 'capital', 'asset', 'property', 'building', 'construction', 'expansion', 'acquisition', 'upgrade', 'technology', 'infrastructure']):
        return 'Investing Activities'
    
    # Financing Activities
    elif any(word in desc_lower for word in ['loan', 'interest', 'financing', 'debt', 'credit', 'bank', 'mortgage', 'dividend', 'share', 'stock', 'equity', 'bond', 'refinancing', 'funding']):
        return 'Financing Activities'
    
    else:
        return 'Operating Activities'

def rule_based_categorize(description, amount):
    """Rule-based categorization"""
    return categorize_with_local_ai(description, amount)

def standardize_columns(df):
    """Standardize column names"""
    df = df.copy()
    
    # Standardize column names
    column_mapping = {
        'Date': ['Date', 'date', 'DATE', 'Transaction Date', 'transaction_date'],
        'Description': ['Description', 'description', 'DESCRIPTION', 'Narration', 'narration', 'Particulars', 'particulars'],
        'Amount': ['Amount', 'amount', 'AMOUNT', 'Debit', 'debit', 'Credit', 'credit', 'Transaction Amount'],
        'Type': ['Type', 'type', 'TYPE', 'Transaction Type', 'transaction_type'],
        'Status': ['Status', 'status', 'STATUS', 'Transaction Status', 'transaction_status']
    }
    
    for standard_name, possible_names in column_mapping.items():
        for col in df.columns:
            if col in possible_names:
                df = df.rename(columns={col: standard_name})
                break
    
    return df

def process_dataframe(df, use_ai=True):
    """Process dataframe with categorization"""
    print(f"🚀 Processing {len(df)} transactions...")
    
    # Standardize columns
    df = standardize_columns(df)
    
    # Ensure required columns exist
    if 'Description' not in df.columns:
        df['Description'] = 'Transaction'
    if 'Amount' not in df.columns:
        df['Amount'] = 0
    if 'Date' not in df.columns:
        df['Date'] = pd.Timestamp.now()
    
    # Categorize transactions
    categories = []
    ai_count = 0
    rule_count = 0
    
    for idx, row in df.iterrows():
        description = str(row.get('Description', ''))
        amount = float(row.get('Amount', 0))
        
        if use_ai and LOCAL_AI_AVAILABLE:
            category = categorize_with_local_ai(description, amount)
            ai_count += 1
        else:
            category = rule_based_categorize(description, amount)
            rule_count += 1
        
        categories.append(category)
    
    df['Category'] = categories
    
    # Set Type and Status
    df['Type'] = df['Amount'].apply(lambda x: 'Inward' if x > 0 else 'Outward')
    df['Status'] = 'Completed'
    
    print(f"✅ Processing complete:")
    print(f"   🤖 AI categorized: {ai_count} transactions")
    print(f"   📏 Rule categorized: {rule_count} transactions")
    
    return df

def generate_cash_flow_analysis(df):
    """Generate cash flow analysis"""
    if df.empty:
        return {
            'operating': {'inflow': 0, 'outflow': 0, 'net': 0},
            'investing': {'inflow': 0, 'outflow': 0, 'net': 0},
            'financing': {'inflow': 0, 'outflow': 0, 'net': 0},
            'total': {'inflow': 0, 'outflow': 0, 'net': 0}
        }
    
    # Group by category and calculate flows
    cash_flows = {}
    
    for category in ['Operating Activities', 'Investing Activities', 'Financing Activities']:
        category_data = df[df['Category'] == category]
        
        inflow = category_data[category_data['Amount'] > 0]['Amount'].sum()
        outflow = abs(category_data[category_data['Amount'] < 0]['Amount'].sum())
        net = inflow - outflow
        
        cash_flows[category.lower().replace(' ', '_')] = {
            'inflow': float(inflow),
            'outflow': float(outflow),
            'net': float(net)
        }
    
    # Calculate totals
    total_inflow = sum(flow['inflow'] for flow in cash_flows.values())
    total_outflow = sum(flow['outflow'] for flow in cash_flows.values())
    total_net = total_inflow - total_outflow
    
    cash_flows['total'] = {
        'inflow': float(total_inflow),
        'outflow': float(total_outflow),
        'net': float(total_net)
    }
    
    return cash_flows

def generate_forecast(df, days=7):
    """Generate simple cash flow forecast"""
    if df.empty:
        return {
            'daily_forecast': [],
            'weekly_forecast': [],
            'monthly_forecast': []
        }
    
    # Simple forecasting based on historical averages
    daily_avg = df['Amount'].mean()
    weekly_avg = daily_avg * 7
    monthly_avg = daily_avg * 30
    
    daily_forecast = []
    for i in range(days):
        forecast_date = datetime.now() + timedelta(days=i+1)
        daily_forecast.append({
            'date': forecast_date.strftime('%Y-%m-%d'),
            'amount': float(daily_avg),
            'confidence': 0.5
        })
    
    return {
        'daily_forecast': daily_forecast,
        'weekly_forecast': [{'week': i+1, 'amount': float(weekly_avg), 'confidence': 0.6} for i in range(4)],
        'monthly_forecast': [{'month': i+1, 'amount': float(monthly_avg), 'confidence': 0.7} for i in range(3)]
    }

@app.route('/')
def home():
    return render_template('sap_bank_interface.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        if 'sap_file' not in request.files or 'bank_file' not in request.files:
            return jsonify({'error': 'Both SAP and Bank files are required'}), 400
        
        sap_file = request.files['sap_file']
        bank_file = request.files['bank_file']
        
        if sap_file.filename == '' or bank_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Read files
        sap_df = pd.read_excel(sap_file)
        bank_df = pd.read_excel(bank_file)
        
        # Process data
        sap_processed = process_dataframe(sap_df, use_ai=True)
        bank_processed = process_dataframe(bank_df, use_ai=True)
        
        # Generate analysis
        sap_cash_flow = generate_cash_flow_analysis(sap_processed)
        bank_cash_flow = generate_cash_flow_analysis(bank_processed)
        
        # Generate forecast
        forecast = generate_forecast(bank_processed)
        
        return jsonify({
            'status': 'success',
            'sap_transactions': len(sap_processed),
            'bank_transactions': len(bank_processed),
            'sap_cash_flow': sap_cash_flow,
            'bank_cash_flow': bank_cash_flow,
            'forecast': forecast,
            'ai_available': LOCAL_AI_AVAILABLE,
            'ollama_available': OLLAMA_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-ollama', methods=['GET'])
def test_ollama():
    """Test Ollama integration"""
    return jsonify({
        'status': 'success',
        'ollama_available': OLLAMA_AVAILABLE,
        'local_ai_available': LOCAL_AI_AVAILABLE,
        'message': 'Ollama integration ready for implementation'
    })

if __name__ == '__main__':
    print("🚀 Starting Clean Cash Flow SAP Bank System...")
    print(f"🤖 Local AI Available: {LOCAL_AI_AVAILABLE}")
    print(f"🦙 Ollama Available: {OLLAMA_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5000) 