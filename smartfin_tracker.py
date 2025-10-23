"""
SmartFin Tracker: AI-Based Intelligent Personal Expense Management System
Main system implementation with AI categorization and analytics
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from typing import List, Dict, Tuple
import pickle
import os

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class SmartFinTracker:
    """Main SmartFin Tracker System"""
    
    def __init__(self, db_path='smartfin.db'):
        self.db_path = db_path
        self.conn = None
        self.vectorizer = None
        self.classifier = None
        self.categories = [
            'Food & Dining', 'Transportation', 'Shopping', 'Entertainment',
            'Bills & Utilities', 'Healthcare', 'Education', 'Travel',
            'Groceries', 'Personal Care', 'Investment', 'Other'
        ]
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                payment_method TEXT,
                is_recurring BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Budget table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                monthly_limit REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        
        self.conn.commit()
        
    def add_transaction(self, date: str, description: str, amount: float, 
                       category: str = None, payment_method: str = 'Cash',
                       is_recurring: bool = False, notes: str = ''):
        """Add a new transaction with optional AI categorization"""
        
        # Auto-categorize if category not provided
        if category is None:
            category = self.predict_category(description)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (date, description, amount, category, 
                                     payment_method, is_recurring, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (date, description, amount, category, payment_method, 
              is_recurring, notes))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def train_categorization_model(self, training_data: pd.DataFrame = None):
        """Train AI model for expense categorization"""
        
        if training_data is None:
            # Use sample training data
            training_data = self._get_sample_training_data()
        
        # Prepare features and labels
        X = training_data['description']
        y = training_data['category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        self._save_model()
        
        return accuracy, classification_report(y_test, y_pred, output_dict=True)
    
    def predict_category(self, description: str) -> str:
        """Predict category for a transaction description"""
        
        if self.classifier is None:
            self._load_model()
        
        if self.classifier is None:
            # Train model if not available
            self.train_categorization_model()
        
        # Vectorize and predict
        desc_vec = self.vectorizer.transform([description])
        predicted_category = self.classifier.predict(desc_vec)[0]
        
        return predicted_category
    
    def get_transactions(self, start_date: str = None, end_date: str = None,
                        category: str = None) -> pd.DataFrame:
        """Retrieve transactions with optional filters"""
        
        query = "SELECT * FROM transactions WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY date DESC"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_spending_summary(self, start_date: str = None, 
                            end_date: str = None) -> Dict:
        """Get spending summary statistics"""
        
        df = self.get_transactions(start_date, end_date)
        
        if df.empty:
            return {}
        
        summary = {
            'total_spending': df['amount'].sum(),
            'transaction_count': len(df),
            'average_transaction': df['amount'].mean(),
            'category_breakdown': df.groupby('category')['amount'].sum().to_dict(),
            'payment_method_breakdown': df.groupby('payment_method')['amount'].sum().to_dict(),
            'daily_average': df['amount'].sum() / max(df['date'].nunique(), 1)
        }
        
        return summary
    
    def predict_future_spending(self, days: int = 30) -> Dict:
        """Predict future spending based on historical patterns"""
        
        # Get last 90 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        df = self.get_transactions(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            return {}
        
        # Calculate daily average
        daily_avg = df['amount'].sum() / 90
        
        # Category-wise prediction
        category_daily_avg = df.groupby('category')['amount'].sum() / 90
        
        predictions = {
            'predicted_total': daily_avg * days,
            'daily_average': daily_avg,
            'category_predictions': (category_daily_avg * days).to_dict(),
            'confidence': 'Medium' if len(df) > 30 else 'Low'
        }
        
        return predictions
    
    def set_budget(self, category: str, monthly_limit: float):
        """Set monthly budget for a category"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO budgets (category, monthly_limit)
            VALUES (?, ?)
        ''', (category, monthly_limit))
        
        self.conn.commit()
    
    def check_budget_status(self) -> Dict:
        """Check current spending against budgets"""
        
        # Get current month's spending
        now = datetime.now()
        start_date = now.replace(day=1).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        
        df = self.get_transactions(start_date, end_date)
        current_spending = df.groupby('category')['amount'].sum().to_dict()
        
        # Get budgets
        budgets_df = pd.read_sql_query("SELECT * FROM budgets", self.conn)
        
        budget_status = []
        for _, row in budgets_df.iterrows():
            category = row['category']
            limit = row['monthly_limit']
            spent = current_spending.get(category, 0)
            remaining = limit - spent
            percentage = (spent / limit * 100) if limit > 0 else 0
            
            status = 'OK' if percentage < 80 else 'Warning' if percentage < 100 else 'Exceeded'
            
            budget_status.append({
                'category': category,
                'limit': limit,
                'spent': spent,
                'remaining': remaining,
                'percentage': percentage,
                'status': status
            })
        
        return budget_status
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect unusual spending patterns"""
        
        df = self.get_transactions()
        
        if len(df) < 10:
            return []
        
        anomalies = []
        
        # Check for unusually large transactions
        mean_amount = df['amount'].mean()
        std_amount = df['amount'].std()
        threshold = mean_amount + (2 * std_amount)
        
        large_transactions = df[df['amount'] > threshold]
        
        for _, row in large_transactions.iterrows():
            anomalies.append({
                'type': 'Large Transaction',
                'date': row['date'],
                'description': row['description'],
                'amount': row['amount'],
                'expected_range': f"${mean_amount:.2f} ± ${std_amount:.2f}"
            })
        
        return anomalies
    
    def export_data(self, filepath: str, format: str = 'csv'):
        """Export transaction data"""
        
        df = self.get_transactions()
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
    
    def import_data(self, filepath: str, format: str = 'csv'):
        """Import transaction data"""
        
        if format == 'csv':
            df = pd.read_csv(filepath)
        elif format == 'json':
            df = pd.read_json(filepath)
        elif format == 'excel':
            df = pd.read_excel(filepath)
        
        # Add transactions
        for _, row in df.iterrows():
            self.add_transaction(
                date=row.get('date', datetime.now().strftime('%Y-%m-%d')),
                description=row['description'],
                amount=row['amount'],
                category=row.get('category'),
                payment_method=row.get('payment_method', 'Cash'),
                notes=row.get('notes', '')
            )
    
    def _get_sample_training_data(self) -> pd.DataFrame:
        """Generate sample training data for ML model"""
        
        training_samples = [
            # Food & Dining
            ('Restaurant bill', 'Food & Dining'),
            ('Pizza delivery', 'Food & Dining'),
            ('Coffee shop', 'Food & Dining'),
            ('Lunch at cafe', 'Food & Dining'),
            ('Dinner with friends', 'Food & Dining'),
            
            # Transportation
            ('Uber ride', 'Transportation'),
            ('Gas station', 'Transportation'),
            ('Metro card recharge', 'Transportation'),
            ('Taxi fare', 'Transportation'),
            ('Car maintenance', 'Transportation'),
            
            # Shopping
            ('Amazon purchase', 'Shopping'),
            ('Clothing store', 'Shopping'),
            ('Electronics shop', 'Shopping'),
            ('Online shopping', 'Shopping'),
            ('Department store', 'Shopping'),
            
            # Entertainment
            ('Movie tickets', 'Entertainment'),
            ('Concert tickets', 'Entertainment'),
            ('Streaming subscription', 'Entertainment'),
            ('Gaming purchase', 'Entertainment'),
            ('Theme park', 'Entertainment'),
            
            # Bills & Utilities
            ('Electricity bill', 'Bills & Utilities'),
            ('Water bill', 'Bills & Utilities'),
            ('Internet bill', 'Bills & Utilities'),
            ('Phone bill', 'Bills & Utilities'),
            ('Rent payment', 'Bills & Utilities'),
            
            # Healthcare
            ('Doctor visit', 'Healthcare'),
            ('Pharmacy', 'Healthcare'),
            ('Medical test', 'Healthcare'),
            ('Health insurance', 'Healthcare'),
            ('Dental checkup', 'Healthcare'),
            
            # Education
            ('Course fee', 'Education'),
            ('Books purchase', 'Education'),
            ('Online course', 'Education'),
            ('Tuition fee', 'Education'),
            ('Educational supplies', 'Education'),
            
            # Travel
            ('Flight ticket', 'Travel'),
            ('Hotel booking', 'Travel'),
            ('Travel insurance', 'Travel'),
            ('Vacation package', 'Travel'),
            ('Airport parking', 'Travel'),
            
            # Groceries
            ('Supermarket', 'Groceries'),
            ('Grocery store', 'Groceries'),
            ('Fresh vegetables', 'Groceries'),
            ('Weekly groceries', 'Groceries'),
            ('Food market', 'Groceries'),
            
            # Personal Care
            ('Haircut', 'Personal Care'),
            ('Salon visit', 'Personal Care'),
            ('Cosmetics', 'Personal Care'),
            ('Gym membership', 'Personal Care'),
            ('Spa treatment', 'Personal Care'),
            
            # Investment
            ('Stock purchase', 'Investment'),
            ('Mutual fund', 'Investment'),
            ('Savings deposit', 'Investment'),
            ('Investment plan', 'Investment'),
            ('Portfolio addition', 'Investment'),
        ]
        
        return pd.DataFrame(training_samples, columns=['description', 'category'])
    
    def _save_model(self):
        """Save trained model to disk"""
        
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f'{model_dir}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(f'{model_dir}/classifier.pkl', 'wb') as f:
            pickle.dump(self.classifier, f)
    
    def _load_model(self):
        """Load trained model from disk"""
        
        try:
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('models/classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
        except FileNotFoundError:
            pass
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Main function for testing"""
    
    tracker = SmartFinTracker()
    
    # Train model
    print("Training AI categorization model...")
    accuracy, report = tracker.train_categorization_model()
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Add sample transactions
    print("\nAdding sample transactions...")
    sample_transactions = [
        ('2025-01-15', 'Starbucks coffee', 5.50, 'Cash'),
        ('2025-01-16', 'Uber to office', 12.30, 'Credit Card'),
        ('2025-01-17', 'Grocery shopping at Walmart', 85.20, 'Debit Card'),
        ('2025-01-18', 'Netflix subscription', 15.99, 'Credit Card'),
        ('2025-01-19', 'Electricity bill payment', 120.00, 'Online Banking'),
    ]
    
    for date, desc, amount, method in sample_transactions:
        tracker.add_transaction(date, desc, amount, payment_method=method)
    
    # Get summary
    print("\nSpending Summary:")
    summary = tracker.get_spending_summary()
    print(json.dumps(summary, indent=2))
    
    tracker.close()
    print("\nSmartFin Tracker initialized successfully!")


if __name__ == '__main__':
    main()

