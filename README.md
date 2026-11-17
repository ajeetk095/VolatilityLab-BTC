# VolatilityLab-BTC

A best-practices handbook for modeling Bitcoin return risk (MScFE 610, Group 8665). It covers skewness, kurtosis and heteroscedasticity, outlier sensitivity, and non-stationarity, each comparing BTC-USD with a simulated normal benchmark.

## Run it online (free)
- Notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajeetk095/VolatilityLab-BTC/blob/main/VolatilityLab_BTC.ipynb)
- Notebook in Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ajeetk095/VolatilityLab-BTC/main?labpath=VolatilityLab_BTC.ipynb)
- Interactive app (live data, projections, guidance): https://YOUR-APP-NAME.streamlit.app

## Run locally
```bash
pip install -r requirements.txt
jupyter notebook VolatilityLab_BTC.ipynb     # the notebook
streamlit run app.py                          # the app
```

## Files
- `VolatilityLab_BTC.ipynb`: the analysis (run all cells)
- `app.py`: interactive Streamlit app
- `requirements.txt`: Python dependencies
- `VolatilityLab-BTC_Report.pdf`: the written report (no code)
