import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import re

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def safe_filename(name):
    name = re.sub(r'[^\w\s-]', '', name)  # Remove special characters
    name = re.sub(r'\s+', '_', name)      # Replace spaces with underscores
    return name[:50]                      # Limit length to 50 characters

def load_data(path):
    df = pd.read_csv(path)
    df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
    df['Festival_flag'] = df['Festival'].notna()
    df['Price'] = np.where(df['Festival_flag'], df['Festival_Sale_Price'], df['Discount_Price'])
    df['day_of_year'] = df['Sale_Date'].dt.dayofyear
    df['day_of_week'] = df['Sale_Date'].dt.dayofweek
    df['month'] = df['Sale_Date'].dt.month
    return df

def train_and_save(df):
    features = ['day_of_year', 'day_of_week', 'month']

    for product in df['Product_name'].unique():
        df_prod = df[df['Product_name'] == product]
        fest = df_prod[df_prod['Festival_flag']]
        nonfest = df_prod[~df_prod['Festival_flag']]

        safe_name = safe_filename(product)

        if not nonfest.empty:
            model_nonfest = RandomForestRegressor(n_estimators=100, random_state=42)
            model_nonfest.fit(nonfest[features], nonfest['Discount_Price'])
            with open(f'models/{safe_name}_nonfest.pkl', 'wb') as f:
                pickle.dump(model_nonfest, f)

        if not fest.empty:
            model_fest = RandomForestRegressor(n_estimators=100, random_state=42)
            model_fest.fit(fest[features], fest['Festival_Sale_Price'])
            with open(f'models/{safe_name}_fest.pkl', 'wb') as f:
                pickle.dump(model_fest, f)

    print("✅ All product models trained and saved.")

if __name__ == '__main__':
    df = load_data('data/final-ecommerce.csv')
    train_and_save(df)
