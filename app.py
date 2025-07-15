from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
import os
import re

app = Flask(__name__)

# Load raw data
raw = pd.read_csv('data/final-ecommerce.csv')
raw['Sale_Date'] = pd.to_datetime(raw['Sale_Date'])

# Build festival map
festival_map = (
    raw.loc[raw['Festival'].notna(), ['Sale_Date','Festival']]
       .assign(md=lambda x: x['Sale_Date'].dt.strftime('%m-%d'))
       .drop_duplicates('md')
       .set_index('md')['Festival']
       .to_dict()
)
festival_dates = set(festival_map.keys())

# Load product list
products = sorted(raw['Product_name'].unique())

def safe_filename(name):
    name = re.sub(r'[^\w\s-]', '', name)  # Remove special characters
    name = re.sub(r'\s+', '_', name)      # Replace spaces with underscores
    return name[:50]                      # Limit length to 50 characters

@app.template_global()
def today_date():
    return date.today().isoformat()

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    selected_product = None

    if request.method == 'POST':
        date_str = request.form['date']
        selected_product = request.form['product']
        dt = datetime.strptime(date_str, '%Y-%m-%d').date()
        feat = np.array([[dt.timetuple().tm_yday, dt.weekday(), dt.month]])
        md = dt.strftime('%m-%d')

        # Use safe_filename to match the training script's file naming
        safe_name = safe_filename(selected_product)
        model_fest_path = f'models/{safe_name}_fest.pkl'
        model_nonfest_path = f'models/{safe_name}_nonfest.pkl'

        if md in festival_dates and os.path.exists(model_fest_path):
            with open(model_fest_path, 'rb') as f:
                model_fest = pickle.load(f)
            name = festival_map[md]
            price = model_fest.predict(feat)[0]
            prediction = f"{selected_product} - {name} Sale Price on {dt}: $ {price:.2f}"
        elif os.path.exists(model_nonfest_path):
            with open(model_nonfest_path, 'rb') as f:
                model_nonfest = pickle.load(f)
            price = model_nonfest.predict(feat)[0]
            prediction = f"{selected_product} - Discount Price on {dt}: ₹ {price:.2f}"
        else:
            prediction = f"No model found for {selected_product}."

    return render_template('index.html', products=products, prediction=prediction, selected_product=selected_product)

if __name__ == '__main__':
    app.run(debug=True)