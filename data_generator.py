"""
SmartFin Tracker: Test Data Generator
Generate realistic transaction data for testing and demonstration
"""

import random
from datetime import datetime, timedelta
import pandas as pd


class TransactionGenerator:
    """Generate realistic transaction data"""
    
    def __init__(self):
        self.categories = {
            'Food & Dining': [
                ('Restaurant bill', 15, 80),
                ('Pizza delivery', 12, 35),
                ('Coffee shop', 3, 8),
                ('Fast food', 5, 15),
                ('Lunch at cafe', 10, 25),
                ('Dinner with friends', 30, 100),
                ('Breakfast', 5, 15),
            ],
            'Transportation': [
                ('Uber ride', 8, 30),
                ('Gas station', 30, 70),
                ('Metro card recharge', 20, 50),
                ('Taxi fare', 10, 40),
                ('Car maintenance', 50, 300),
                ('Parking fee', 5, 20),
                ('Toll charges', 2, 10),
            ],
            'Shopping': [
                ('Amazon purchase', 20, 150),
                ('Clothing store', 30, 200),
                ('Electronics shop', 50, 500),
                ('Online shopping', 15, 100),
                ('Department store', 25, 150),
                ('Shoes purchase', 40, 150),
                ('Accessories', 10, 50),
            ],
            'Entertainment': [
                ('Movie tickets', 10, 30),
                ('Concert tickets', 50, 200),
                ('Streaming subscription', 10, 20),
                ('Gaming purchase', 20, 60),
                ('Theme park', 40, 100),
                ('Sports event', 30, 150),
                ('Books and magazines', 10, 40),
            ],
            'Bills & Utilities': [
                ('Electricity bill', 80, 150),
                ('Water bill', 30, 60),
                ('Internet bill', 40, 80),
                ('Phone bill', 30, 70),
                ('Rent payment', 800, 1500),
                ('Gas bill', 40, 100),
                ('Insurance premium', 100, 300),
            ],
            'Healthcare': [
                ('Doctor visit', 50, 200),
                ('Pharmacy', 15, 80),
                ('Medical test', 80, 300),
                ('Health insurance', 150, 400),
                ('Dental checkup', 60, 150),
                ('Medicine purchase', 20, 100),
                ('Health supplements', 25, 80),
            ],
            'Education': [
                ('Course fee', 100, 500),
                ('Books purchase', 30, 150),
                ('Online course', 50, 200),
                ('Tuition fee', 500, 2000),
                ('Educational supplies', 20, 80),
                ('Workshop registration', 50, 200),
                ('Certification exam', 100, 300),
            ],
            'Travel': [
                ('Flight ticket', 200, 800),
                ('Hotel booking', 100, 400),
                ('Travel insurance', 30, 100),
                ('Vacation package', 500, 2000),
                ('Airport parking', 20, 60),
                ('Car rental', 50, 200),
                ('Travel accessories', 20, 100),
            ],
            'Groceries': [
                ('Supermarket', 50, 150),
                ('Grocery store', 40, 120),
                ('Fresh vegetables', 15, 40),
                ('Weekly groceries', 60, 180),
                ('Food market', 30, 90),
                ('Organic store', 40, 120),
                ('Bulk shopping', 100, 300),
            ],
            'Personal Care': [
                ('Haircut', 15, 50),
                ('Salon visit', 30, 100),
                ('Cosmetics', 20, 80),
                ('Gym membership', 40, 100),
                ('Spa treatment', 50, 200),
                ('Beauty products', 25, 80),
                ('Fitness class', 20, 60),
            ],
            'Investment': [
                ('Stock purchase', 100, 1000),
                ('Mutual fund', 200, 1500),
                ('Savings deposit', 500, 3000),
                ('Investment plan', 300, 2000),
                ('Portfolio addition', 400, 2500),
                ('Retirement fund', 500, 3000),
                ('Bond purchase', 300, 1500),
            ],
        }
        
        self.payment_methods = [
            'Cash', 'Credit Card', 'Debit Card', 'Online Banking', 
            'UPI', 'Mobile Wallet', 'PayPal'
        ]
    
    def generate_transactions(self, num_transactions=200, days_back=90):
        """Generate random transactions"""
        
        transactions = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for _ in range(num_transactions):
            # Random date
            random_days = random.randint(0, days_back)
            transaction_date = end_date - timedelta(days=random_days)
            
            # Random category
            category = random.choice(list(self.categories.keys()))
            
            # Random transaction from category
            transaction_types = self.categories[category]
            desc_template, min_amount, max_amount = random.choice(transaction_types)
            
            # Random amount
            amount = round(random.uniform(min_amount, max_amount), 2)
            
            # Random payment method
            payment_method = random.choice(self.payment_methods)
            
            # Create transaction
            transactions.append({
                'date': transaction_date.strftime('%Y-%m-%d'),
                'description': desc_template,
                'amount': amount,
                'category': category,
                'payment_method': payment_method,
                'notes': ''
            })
        
        # Sort by date
        transactions.sort(key=lambda x: x['date'])
        
        return transactions
    
    def generate_recurring_transactions(self, months=3):
        """Generate recurring monthly transactions"""
        
        recurring = [
            ('Rent payment', 1200, 'Bills & Utilities', 'Online Banking', 1),
            ('Internet bill', 60, 'Bills & Utilities', 'Credit Card', 5),
            ('Phone bill', 45, 'Bills & Utilities', 'Credit Card', 8),
            ('Gym membership', 50, 'Personal Care', 'Credit Card', 1),
            ('Streaming subscription', 15, 'Entertainment', 'Credit Card', 1),
            ('Health insurance', 250, 'Healthcare', 'Online Banking', 1),
        ]
        
        transactions = []
        end_date = datetime.now()
        
        for month_offset in range(months):
            for desc, amount, category, method, day in recurring:
                transaction_date = end_date - timedelta(days=30 * month_offset)
                transaction_date = transaction_date.replace(day=day)
                
                transactions.append({
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'description': desc,
                    'amount': amount,
                    'category': category,
                    'payment_method': method,
                    'notes': 'Recurring transaction'
                })
        
        return transactions
    
    def save_to_csv(self, transactions, filename='sample_transactions.csv'):
        """Save transactions to CSV file"""
        
        df = pd.DataFrame(transactions)
        df.to_csv(filename, index=False)
        print(f"Saved {len(transactions)} transactions to {filename}")
        return filename


def main():
    """Generate sample data"""
    
    generator = TransactionGenerator()
    
    # Generate random transactions
    random_transactions = generator.generate_transactions(num_transactions=150, days_back=90)
    
    # Generate recurring transactions
    recurring_transactions = generator.generate_recurring_transactions(months=3)
    
    # Combine all transactions
    all_transactions = random_transactions + recurring_transactions
    
    # Save to CSV
    generator.save_to_csv(all_transactions, 'sample_transactions.csv')
    
    print(f"\nGenerated {len(all_transactions)} total transactions")
    print(f"- Random transactions: {len(random_transactions)}")
    print(f"- Recurring transactions: {len(recurring_transactions)}")


if __name__ == '__main__':
    main()

