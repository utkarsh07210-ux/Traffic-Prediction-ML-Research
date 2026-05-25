# 🚦 Intelligent Transportation Systems: IoT & ML for Urban Traffic Management

**Authors:** Utkarsh Mishra, Shivam Chaudhary  
**Institution:** Keshav Mahavidyalaya, University of Delhi  
**Subject:** Research Methodology | B.Sc (Hons.) Computer Science | 2025–26  

---

## 📌 Abstract

Traffic congestion is a growing problem in urban areas like New Delhi, causing delays, increased fuel consumption, and environmental pollution. This research proposes a **traffic flow prediction system** using Machine Learning techniques — **Linear Regression** and **Long Short-Term Memory (LSTM)** neural networks — applied to time-series traffic data of New Delhi to predict future traffic patterns and support intelligent decision-making in Intelligent Transportation Systems (ITS).

---

## 🎯 Problem Statement

Existing traffic management systems are largely reactive and fail to adapt to dynamic traffic conditions. This research addresses the need for an **automated, ML-based traffic prediction system** capable of accurately forecasting traffic conditions in real time using historical data.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Linear Regression) |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Visualization | Matplotlib |
| Dataset | Kaggle — New Delhi Traffic Probe & Analytics 2024 |

---

## 📁 Repository Structure

```
Traffic-Prediction-ML-Research/
├── traffic_prediction.py          ← Main ML code (Linear Regression + LSTM)
├── traffic_analysis_results.png   ← Output charts
├── Traffic_Prediction_Research_Paper.pdf  ← Full research paper
└── README.md
```

---

## ⚙️ How to Run

**Step 1 — Install dependencies:**
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

**Step 2 — (Optional) Add real dataset:**
- Download from Kaggle: [New Delhi Traffic Probe & Analytics 2024](https://www.kaggle.com)
- Save as `traffic_data.csv` in the same folder
- If not provided, the code generates synthetic sample data automatically

**Step 3 — Run:**
```bash
python traffic_prediction.py
```

---

## 📊 Methodology

### Data Preprocessing
- Removed missing and duplicate values
- Extracted temporal features (hour, day of week)
- Applied **MinMax Normalization:**

```
X_norm = (X - X_min) / (X_max - X_min)
```

### Models Used

**1. Linear Regression (Baseline)**
```
y = β₀ + β₁x₁ + β₂x₂
```
- Simple and computationally efficient
- Fails to capture non-linear temporal patterns

**2. LSTM Neural Network**
```
h_t = o_t · tanh(C_t)
ŷ_t = W_y · h_t + b_y
```
- Learns sequential dependencies and temporal patterns
- Better suited for time-series traffic prediction

### Train / Test Split
- Training: **80%** (~200 records)
- Testing: **20%** (~50 records)

### Evaluation Metric — RMSE
```
RMSE = √(1/n · Σ(y - ŷ)²)
```
Lower RMSE = better model performance.

---

## 📈 Results

| Model | RMSE | Performance |
|---|---|---|
| Linear Regression | 0.2831 | Baseline — smooth but misses fluctuations |
| LSTM | 0.3398 | Better temporal learning on larger datasets |

### Traffic Classification
| Class | Label | Speed Range |
|---|---|---|
| 0 | Low Traffic | ≥ 60 km/h |
| 1 | Medium Traffic | 35–60 km/h |
| 2 | High Traffic (Congested) | < 35 km/h |

---

## 📉 Output Charts

The script generates 4 charts saved as `traffic_analysis_results.png`:

1. **Traffic Prediction Comparison** — Actual vs Linear Regression vs LSTM
2. **RMSE Comparison** — Bar chart comparing model errors
3. **Delhi Geospatial Visualization** — Traffic classes plotted on lat/lon map
4. **Hourly Traffic Pattern** — Average speed by hour with peak hour highlights

---

## 🔭 Future Scope

- Integrate **real-time IoT sensor data** and GPS feeds
- Incorporate **weather conditions**, road incidents, and public events
- Explore advanced models: **SVM, Random Forest, Transformer-based architectures**
- Deploy as a **web dashboard** for live traffic monitoring

---

## 📝 Conclusion

Linear Regression provides a simple, computationally efficient baseline but is limited in capturing complex temporal traffic patterns. The LSTM model demonstrates superior learning of time-dependent behaviour such as peak-hour congestion and daily traffic trends, making it more suitable for real-world Intelligent Transportation Systems.

---

## 📚 References

1. K. Shehzad Khattak and Z. Hussain Khan — *"Evaluation and Challenges of IoT Simulators for ITS Applications"* — Science, Engineering and Technology, 2024
2. H. Yan and Y. Li — *"Generative AI for Intelligent Transportation Systems"* — ACM Comput. Surv., 2025
3. R. Liu and S. Y. Shin — *"A Review of Traffic Flow Prediction Methods in ITS"* — MDPI, 2025
4. E. Omol et al. — *"Anomaly Detection in IoT Sensor Data Using ML for Smart Grids"*
5. M. Hassan et al. — *"Application of ML in Intelligent Transport Systems"* — Discover Civil Engineering, 2025
6. Kaggle — New Delhi Traffic Probe & Analytics 2024 Dataset

---

## 👨‍💻 Authors

| Name | Institution |
|---|---|
| **Utkarsh Mishra** | Keshav Mahavidyalaya, University of Delhi |
| **Shivam Chaudhary** | Keshav Mahavidyalaya, University of Delhi |

📧 utkarsh07021@gmail.com  
🔗 [linkedin.com/in/utkarshmishra06](https://linkedin.com/in/utkarshmishra06)  
💻 [github.com/utkarsh07210-ux](https://github.com/utkarsh07210-ux)
