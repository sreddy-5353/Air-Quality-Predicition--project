# 🌫️ Predicting Air Quality with Neural Networks

A full Streamlit web application that uses a deep neural network to predict the Air Quality Index (AQI) from environmental sensor readings.

## 🗂️ Project Structure

```
air_quality_app/
├── app.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── pages_src/
│   ├── home.py               # Landing page
│   ├── data_explorer.py      # EDA & visualizations
│   ├── model_training.py     # Configure & train the NN
│   ├── predictor.py          # Live AQI prediction
│   └── evaluation.py         # Model evaluation & benchmarks
├── utils/
│   ├── data_loader.py        # Dataset loading (with synthetic fallback)
│   └── model.py              # Build, train, save the model
└── data/
    └── air_quality.csv       # (optional) Your real dataset here
```

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add your own dataset
# Place it at: data/air_quality.csv
# Required columns: PM2.5, PM10, CO, NO2, SO2, O3, Temperature, Humidity, Wind_Speed, AQI

# 3. Run the app
streamlit run app.py
```

## 📊 App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, AQI scale, tech stack |
| 📊 Data Explorer | Raw data, distributions, correlations, trends |
| 🧠 Model Training | Configure layers, neurons, optimizer — train live |
| 🔮 Live Predictor | Input sensor values → instant AQI prediction |
| 📈 Evaluation | Loss curves, pred vs actual, error analysis, benchmarks |

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow 2.x / Keras** — Neural network
- **Streamlit** — Web interface
- **Scikit-learn** — Preprocessing & metrics
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data handling

## 📡 Dataset

If no dataset is provided, the app auto-generates a synthetic dataset of 2,000 records with realistic pollutant distributions. You can replace it with real data from sources like:
- [OpenAQ](https://openaq.org)
- [Kaggle Air Quality Datasets](https://www.kaggle.com/datasets?search=air+quality)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
