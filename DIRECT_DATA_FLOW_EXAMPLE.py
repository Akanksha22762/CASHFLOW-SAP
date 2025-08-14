
# ===== BACKEND ROUTE EXAMPLE =====
@app.route('/analyze-cash-flow', methods=['POST'])
def analyze_cash_flow():
    """Analyze cash flow and return data directly"""
    try:
        # Load and process data
        bank_df = pd.read_excel('data/bank_data_processed.xlsx')
        
        # Calculate actual values
        operating_count = len(bank_df[bank_df['Category'].str.contains('Operating', case=False)])
        investing_count = len(bank_df[bank_df['Category'].str.contains('Investing', case=False)])
        financing_count = len(bank_df[bank_df['Category'].str.contains('Financing', case=False)])
        total_count = len(bank_df)
        
        # Return data in the exact format frontend expects
        response_data = {
            'operating_activities': {
                'count': operating_count,
                'total': float(bank_df[bank_df['Category'].str.contains('Operating', case=False)]['Amount'].sum()),
                'transactions': []
            },
            'investing_activities': {
                'count': investing_count,
                'total': float(bank_df[bank_df['Category'].str.contains('Investing', case=False)]['Amount'].sum()),
                'transactions': []
            },
            'financing_activities': {
                'count': financing_count,
                'total': float(bank_df[bank_df['Category'].str.contains('Financing', case=False)]['Amount'].sum()),
                'transactions': []
            },
            'total_transactions': total_count,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ===== FRONTEND CALL EXAMPLE =====
function analyzeCashFlow() {
    fetch('/analyze-cash-flow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        // Display backend data directly - no synchronization needed!
        displayBackendDataDirectly(data, 'cash_flow');
    })
    .catch(error => {
        console.error('Analysis failed:', error);
    });
}
