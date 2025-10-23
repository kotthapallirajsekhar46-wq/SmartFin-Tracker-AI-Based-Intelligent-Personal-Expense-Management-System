"""
SmartFin Tracker: Main Execution Script
Run the complete system and generate results for the report
"""

import os
import json
from smartfin_tracker import SmartFinTracker
from visualizations import ExpenseVisualizer
from data_generator import TransactionGenerator
import pandas as pd


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("SmartFin Tracker: AI-Based Intelligent Personal Expense Management")
    print("=" * 70)
    print()
    
    # Initialize system
    print("1. Initializing SmartFin Tracker System...")
    tracker = SmartFinTracker('smartfin.db')
    print("   ✓ Database initialized")
    print()
    
    # Train AI model
    print("2. Training AI Categorization Model...")
    accuracy, report = tracker.train_categorization_model()
    print(f"   ✓ Model trained successfully")
    print(f"   ✓ Model accuracy: {accuracy:.2%}")
    print()
    
    # Save model performance
    with open('results/model_performance.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'classification_report': report
        }, f, indent=2)
    
    # Import sample data
    print("3. Importing Transaction Data...")
    if os.path.exists('sample_transactions.csv'):
        tracker.import_data('sample_transactions.csv', format='csv')
        print("   ✓ Transactions imported successfully")
    else:
        print("   ! Sample data not found, generating new data...")
        generator = TransactionGenerator()
        transactions = generator.generate_transactions(150, 90)
        transactions += generator.generate_recurring_transactions(3)
        generator.save_to_csv(transactions, 'sample_transactions.csv')
        tracker.import_data('sample_transactions.csv', format='csv')
        print("   ✓ Generated and imported transactions")
    print()
    
    # Get transaction statistics
    print("4. Analyzing Transaction Data...")
    df = tracker.get_transactions()
    print(f"   ✓ Total transactions: {len(df)}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   ✓ Categories: {df['category'].nunique()}")
    print()
    
    # Get spending summary
    print("5. Generating Spending Summary...")
    summary = tracker.get_spending_summary()
    print(f"   ✓ Total spending: ${summary['total_spending']:.2f}")
    print(f"   ✓ Average transaction: ${summary['average_transaction']:.2f}")
    print(f"   ✓ Daily average: ${summary['daily_average']:.2f}")
    print()
    
    print("   Category Breakdown:")
    for category, amount in sorted(summary['category_breakdown'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"     - {category}: ${amount:.2f}")
    print()
    
    # Set budgets
    print("6. Setting Budget Limits...")
    budgets = {
        'Food & Dining': 500,
        'Transportation': 300,
        'Shopping': 400,
        'Entertainment': 200,
        'Bills & Utilities': 1500,
        'Groceries': 400,
    }
    
    for category, limit in budgets.items():
        tracker.set_budget(category, limit)
    print(f"   ✓ Set budgets for {len(budgets)} categories")
    print()
    
    # Check budget status
    print("7. Checking Budget Status...")
    budget_status = tracker.check_budget_status()
    for status in budget_status:
        symbol = "✓" if status['status'] == 'OK' else "⚠" if status['status'] == 'Warning' else "✗"
        print(f"   {symbol} {status['category']}: ${status['spent']:.2f} / ${status['limit']:.2f} ({status['percentage']:.1f}%)")
    print()
    
    # Predict future spending
    print("8. Predicting Future Spending...")
    predictions = tracker.predict_future_spending(30)
    print(f"   ✓ Predicted spending (30 days): ${predictions['predicted_total']:.2f}")
    print(f"   ✓ Daily average: ${predictions['daily_average']:.2f}")
    print(f"   ✓ Confidence level: {predictions['confidence']}")
    print()
    
    # Detect anomalies
    print("9. Detecting Spending Anomalies...")
    anomalies = tracker.detect_anomalies()
    if anomalies:
        print(f"   ⚠ Found {len(anomalies)} unusual transactions:")
        for i, anomaly in enumerate(anomalies[:5], 1):
            print(f"     {i}. {anomaly['description']}: ${anomaly['amount']:.2f} on {anomaly['date']}")
    else:
        print("   ✓ No anomalies detected")
    print()
    
    # Test AI categorization
    print("10. Testing AI Categorization...")
    test_descriptions = [
        "Starbucks coffee morning",
        "Uber ride to airport",
        "Amazon online shopping",
        "Netflix monthly subscription",
        "Electricity bill payment",
        "Doctor consultation fee",
        "Python course on Udemy",
        "Flight tickets to Paris",
        "Walmart grocery shopping",
        "Haircut at salon"
    ]
    
    print("   Testing automatic categorization:")
    categorization_results = []
    for desc in test_descriptions:
        predicted = tracker.predict_category(desc)
        categorization_results.append({
            'description': desc,
            'predicted_category': predicted
        })
        print(f"     - '{desc}' → {predicted}")
    print()
    
    # Save categorization results
    with open('results/categorization_test.json', 'w') as f:
        json.dump(categorization_results, f, indent=2)
    
    # Export data
    print("11. Exporting Data...")
    tracker.export_data('results/all_transactions.csv', format='csv')
    tracker.export_data('results/all_transactions.json', format='json')
    print("   ✓ Data exported to CSV and JSON formats")
    print()
    
    # Save summary statistics
    print("12. Saving Summary Statistics...")
    stats = {
        'summary': summary,
        'budget_status': budget_status,
        'predictions': predictions,
        'anomalies': anomalies,
        'model_accuracy': accuracy
    }
    
    with open('results/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print("   ✓ Statistics saved")
    print()
    
    # Close database
    tracker.close()
    
    print("=" * 70)
    print("System execution completed successfully!")
    print("=" * 70)
    print()
    
    return tracker


if __name__ == '__main__':
    main()

