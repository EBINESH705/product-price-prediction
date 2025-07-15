# 🎯 Product Price Prediction with Festival-Aware Discounts

Welcome to the **Price Predictor Web App** — a smart machine learning-powered application that predicts future prices of e-commerce products. It intelligently considers **festival dates** and shows **special discounted prices** when applicable!

---

## 🌟 Features

- 📅 **Date-based price prediction** using ML models (Random Forest)
- 🛍️ Select **any product** from the dropdown to get accurate results
- 🎉 Automatically applies **festival discounts** (e.g., Diwali, Christmas, Black Friday)
- 🧠 Trains **separate models** for festival and non-festival days
- 🖥️ Clean and simple **Flask-based web UI**

---

## 🧪 Tech Stack

- Python 3.x  
- Flask (Web Framework)  
- Scikit-learn (Machine Learning)  
- Pandas, NumPy (Data handling)  
- HTML + Jinja2 (Templates)

---

## 🔍 How It Works

1. Upload a product sales dataset (`final-ecommerce.csv`) containing:
   - Product name
   - Sale date
   - Discount price
   - Festival name (if any)
   - Festival sale price

2. The app trains separate models for:
   - **Festival days** – using `Festival_Sale_Price`
   - **Regular days** – using `Discount_Price`

3. During prediction:
   - The selected date is checked for a festival.
   - The correct model (festival/non-festival) is used.
   - The predicted price is shown on the web page.

---

## 📊 Sample Dataset Structure

| Product_name | Sale_Date  | Discount_Price | Festival | Festival_Sale_Price |
|--------------|------------|----------------|----------|----------------------|
| iPhone 15    | 2024-12-25 | 75000          | Christmas| 70000                |
| Samsung TV   | 2024-07-01 | 45000          |          |                      |

---

## ⚙️ Setup Instructions

1. 📥 Clone the repo:
   ```bash
   [git clone https://github.com/your-username/price-predictor-app.git
   cd price-predictor-app](https://github.com/EBINESH705/product-price-prediction.git)
