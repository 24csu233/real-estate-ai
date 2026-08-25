# Real Estate AI

## AI-Powered Property Management and Price Prediction System

Real Estate AI is a web-based application developed using Python and Streamlit to help manage real estate properties, predict property prices, recommend suitable properties, manage customer leads, and analyze real estate data.

## Features

- 🔐 User Login System
- 🏠 Property Management
- 🔍 Property Search and Filtering
- 🤖 AI-Based Property Price Prediction
- ⭐ AI Property Recommendations
- 👥 Customer Lead Management
- 📊 Real Estate Market Analytics
- 🗄️ SQLite Database Integration

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Random Forest Regression
- SQLite
- Plotly

## Machine Learning

The project uses a **Random Forest Regression** model to estimate property prices.

The model uses:

- Location
- BHK
- Area
- Property Type

as input features to predict the estimated property price.

## Project Structure

```text
real-estate-ai/
│
├── app.py
├── database.py
├── properties.csv
├── requirements.txt
├── data/
│   └── leads.csv
└── README.md
