# 🛒 SmartCart – Customer Purchase Prediction App

SmartCart is a data-driven machine learning application that predicts whether a customer is likely to complete a purchase based on their browsing behavior.  
The project helps e-commerce businesses optimize marketing strategies, reduce churn, and improve conversion rates.

---

## 🚀 Features

- Predicts purchase likelihood using customer session data
- Interactive web interface built with Streamlit
- Real-time predictions from trained ML model
- Visual insights into customer behavior
- Easy deployment on Streamlit Cloud

---

## 🧠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib / Seaborn**

---

## 📊 Dataset

- Customer session data collected over one year
- Each row represents a unique user session
- Includes browsing duration, page views, interactions, and more

File used:

---

## ⚙️ Project Structure

smartcart/
│
├── app.py # Main Streamlit app
├── smartcart_customers.csv # Dataset
├── model.pkl # Trained ML model
├── requirements.txt # Dependencies
└── README.md


---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/smartcart.git
cd smartcart
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run app.py
📈 Model Workflow
Data preprocessing and feature engineering

Train-test split

Model training using Scikit-learn

Model serialization using Pickle

Real-time predictions via Streamlit UI

🌐 Deployment
This app is deployed using Streamlit Cloud.

➡️ See deployment steps below.

🎯 Use Cases
E-commerce conversion optimization

Customer behavior analysis

Targeted marketing campaigns

Sales funnel prediction