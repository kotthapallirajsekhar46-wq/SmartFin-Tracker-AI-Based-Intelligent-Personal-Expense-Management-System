"""
SmartFin Tracker: Visualization Module
Generate charts and graphs for expense analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ExpenseVisualizer:
    """Generate visualizations for expense data"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_category_distribution(self, df: pd.DataFrame, filename='category_distribution.png'):
        """Plot pie chart of spending by category"""
        
        category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette('Set3', len(category_totals))
        
        plt.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        plt.title('Expense Distribution by Category', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_spending_trend(self, df: pd.DataFrame, filename='spending_trend.png'):
        """Plot line chart of spending over time"""
        
        df['date'] = pd.to_datetime(df['date'])
        daily_spending = df.groupby('date')['amount'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_spending['date'], daily_spending['amount'], 
                marker='o', linewidth=2, markersize=6, color='#2E86AB')
        
        plt.title('Daily Spending Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Amount ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add trend line
        z = np.polyfit(range(len(daily_spending)), daily_spending['amount'], 1)
        p = np.poly1d(z)
        plt.plot(daily_spending['date'], p(range(len(daily_spending))), 
                "r--", alpha=0.8, linewidth=2, label='Trend')
        plt.legend()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_category_comparison(self, df: pd.DataFrame, filename='category_comparison.png'):
        """Plot bar chart comparing categories"""
        
        category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=True)
        
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette('viridis', len(category_totals))
        
        plt.barh(category_totals.index, category_totals.values, color=colors)
        plt.title('Spending by Category', fontsize=16, fontweight='bold')
        plt.xlabel('Total Amount ($)', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(category_totals.values):
            plt.text(v + 5, i, f'${v:.2f}', va='center')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_payment_method_distribution(self, df: pd.DataFrame, 
                                        filename='payment_methods.png'):
        """Plot distribution of payment methods"""
        
        payment_totals = df.groupby('payment_method')['amount'].sum()
        
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('pastel', len(payment_totals))
        
        plt.bar(payment_totals.index, payment_totals.values, color=colors, edgecolor='black')
        plt.title('Spending by Payment Method', fontsize=16, fontweight='bold')
        plt.xlabel('Payment Method', fontsize=12)
        plt.ylabel('Total Amount ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(payment_totals.values):
            plt.text(i, v + 10, f'${v:.2f}', ha='center', fontweight='bold')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_monthly_comparison(self, df: pd.DataFrame, filename='monthly_comparison.png'):
        """Plot monthly spending comparison"""
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        
        monthly_spending = df.groupby('month')['amount'].sum().reset_index()
        monthly_spending['month'] = monthly_spending['month'].astype(str)
        
        plt.figure(figsize=(12, 6))
        plt.bar(monthly_spending['month'], monthly_spending['amount'], 
               color='#A23B72', edgecolor='black', alpha=0.7)
        
        plt.title('Monthly Spending Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Amount ($)', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(monthly_spending['amount']):
            plt.text(i, v + 20, f'${v:.2f}', ha='center', fontweight='bold')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_budget_status(self, budget_status: list, filename='budget_status.png'):
        """Plot budget vs actual spending"""
        
        if not budget_status:
            return None
        
        df = pd.DataFrame(budget_status)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['limit'], width, label='Budget Limit', 
                      color='#4CAF50', alpha=0.8)
        bars2 = ax.bar(x + width/2, df['spent'], width, label='Actual Spent', 
                      color='#F44336', alpha=0.8)
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Amount ($)', fontsize=12)
        ax.set_title('Budget vs Actual Spending', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['category'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_heatmap(self, df: pd.DataFrame, filename='spending_heatmap.png'):
        """Plot heatmap of spending patterns"""
        
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
        
        # Create pivot table
        pivot = df.pivot_table(values='amount', index='day_of_week', 
                              columns='week', aggfunc='sum', fill_value=0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Amount ($)'})
        
        plt.title('Spending Heatmap by Day and Week', fontsize=16, fontweight='bold')
        plt.xlabel('Week Number', fontsize=12)
        plt.ylabel('Day of Week', fontsize=12)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_top_expenses(self, df: pd.DataFrame, top_n=10, 
                         filename='top_expenses.png'):
        """Plot top N expenses"""
        
        top_expenses = df.nlargest(top_n, 'amount')
        
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette('Reds_r', len(top_expenses))
        
        plt.barh(range(len(top_expenses)), top_expenses['amount'], color=colors)
        plt.yticks(range(len(top_expenses)), 
                  [f"{row['description'][:30]}..." if len(row['description']) > 30 
                   else row['description'] 
                   for _, row in top_expenses.iterrows()])
        
        plt.title(f'Top {top_n} Expenses', fontsize=16, fontweight='bold')
        plt.xlabel('Amount ($)', fontsize=12)
        plt.ylabel('Description', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(top_expenses['amount']):
            plt.text(v + 5, i, f'${v:.2f}', va='center')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_prediction_comparison(self, actual: float, predicted: float, 
                                   filename='prediction_comparison.png'):
        """Plot actual vs predicted spending"""
        
        categories = ['Actual', 'Predicted']
        values = [actual, predicted]
        colors = ['#2E86AB', '#F18F01']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
        
        plt.title('Actual vs Predicted Spending', fontsize=16, fontweight='bold')
        plt.ylabel('Amount ($)', fontsize=12)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'${val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add accuracy percentage
        accuracy = (1 - abs(actual - predicted) / actual) * 100 if actual > 0 else 0
        plt.text(0.5, max(values) * 0.5, f'Accuracy: {accuracy:.1f}%', 
                ha='center', fontsize=14, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def create_dashboard_summary(self, summary_data: dict, 
                                filename='dashboard_summary.png'):
        """Create a summary dashboard with key metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SmartFin Tracker - Dashboard Summary', 
                    fontsize=18, fontweight='bold')
        
        # Total Spending
        axes[0, 0].text(0.5, 0.5, f"${summary_data.get('total_spending', 0):.2f}", 
                       ha='center', va='center', fontsize=36, fontweight='bold',
                       color='#2E86AB')
        axes[0, 0].text(0.5, 0.2, 'Total Spending', ha='center', va='center', 
                       fontsize=14)
        axes[0, 0].axis('off')
        
        # Transaction Count
        axes[0, 1].text(0.5, 0.5, str(summary_data.get('transaction_count', 0)), 
                       ha='center', va='center', fontsize=36, fontweight='bold',
                       color='#A23B72')
        axes[0, 1].text(0.5, 0.2, 'Total Transactions', ha='center', va='center', 
                       fontsize=14)
        axes[0, 1].axis('off')
        
        # Average Transaction
        axes[1, 0].text(0.5, 0.5, f"${summary_data.get('average_transaction', 0):.2f}", 
                       ha='center', va='center', fontsize=36, fontweight='bold',
                       color='#F18F01')
        axes[1, 0].text(0.5, 0.2, 'Average Transaction', ha='center', va='center', 
                       fontsize=14)
        axes[1, 0].axis('off')
        
        # Daily Average
        axes[1, 1].text(0.5, 0.5, f"${summary_data.get('daily_average', 0):.2f}", 
                       ha='center', va='center', fontsize=36, fontweight='bold',
                       color='#4CAF50')
        axes[1, 1].text(0.5, 0.2, 'Daily Average', ha='center', va='center', 
                       fontsize=14)
        axes[1, 1].axis('off')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_all_visualizations(self, tracker):
        """Generate all visualizations for the report"""
        
        print("Generating visualizations...")
        
        # Get transaction data
        df = tracker.get_transactions()
        
        if df.empty:
            print("No transaction data available")
            return []
        
        generated_files = []
        
        # Generate all plots
        generated_files.append(self.plot_category_distribution(df))
        generated_files.append(self.plot_spending_trend(df))
        generated_files.append(self.plot_category_comparison(df))
        generated_files.append(self.plot_payment_method_distribution(df))
        generated_files.append(self.plot_monthly_comparison(df))
        generated_files.append(self.plot_heatmap(df))
        generated_files.append(self.plot_top_expenses(df))
        
        # Budget status
        budget_status = tracker.check_budget_status()
        if budget_status:
            generated_files.append(self.plot_budget_status(budget_status))
        
        # Summary dashboard
        summary = tracker.get_spending_summary()
        generated_files.append(self.create_dashboard_summary(summary))
        
        # Prediction comparison
        predictions = tracker.predict_future_spending(30)
        if predictions:
            actual = summary.get('total_spending', 0)
            predicted = predictions.get('predicted_total', 0)
            generated_files.append(self.plot_prediction_comparison(actual, predicted))
        
        print(f"Generated {len(generated_files)} visualizations")
        return [f for f in generated_files if f is not None]

