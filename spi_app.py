import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm

# Configure page
st.set_page_config(
    page_title="SPI Calculator",
    page_icon="🌧️",
    layout="centered"
)

# Title and description
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>🌧️ Standardized Precipitation Index (SPI) Calculator</h1>
    <p style='text-align: center; font-size: 16px;'>Upload monthly precipitation data, choose a scale (1–48), and generate the SPI time series with download option.</p>
    """,
    unsafe_allow_html=True
)

# File uploader and scale selector
uploaded_file = st.file_uploader("Upload CSV with 'Date' and 'Precip' columns", type='csv')
scale = st.selectbox("Select SPI Scale", [1, 3, 6, 12, 24, 48])

# SPI functions
def calculate_spi(precip_values, scale=3):
    spi_values = []
    precip_values = np.array(precip_values)
    for i in range(len(precip_values)):
        if i < scale - 1:
            spi_values.append(np.nan)
            continue
        window = precip_values[i - scale + 1: i + 1]
        if np.any(np.isnan(window)) or np.std(window) == 0:
            spi_values.append(np.nan)
            continue
        try:
            fit_alpha, fit_loc, fit_beta = gamma.fit(window, floc=0)
            if fit_alpha <= 0 or fit_beta <= 0:
                spi_values.append(np.nan)
                continue
            cdf = gamma.cdf(window[-1], fit_alpha, loc=fit_loc, scale=fit_beta)
            spi = norm.ppf(cdf)
            spi_values.append(spi)
        except:
            spi_values.append(np.nan)
    return np.array(spi_values)

def calculate_spi1_with_climatology(df):
    spi_1 = []
    for idx in range(len(df)):
        current_month = df.loc[idx, 'Month']
        current_value = df.loc[idx, 'Precip']
        historical = df.loc[(df.index < idx) & (df['Month'] == current_month), 'Precip'].dropna().values
        if len(historical) < 10 or current_value <= 0:
            spi_1.append(np.nan)
            continue
        try:
            fit_alpha, fit_loc, fit_beta = gamma.fit(historical, floc=0)
            if fit_alpha <= 0 or fit_beta <= 0:
                spi_1.append(np.nan)
                continue
            cdf = gamma.cdf(current_value, fit_alpha, loc=fit_loc, scale=fit_beta)
            spi = norm.ppf(cdf)
            spi_1.append(spi)
        except:
            spi_1.append(np.nan)
    return np.array(spi_1)

# If a file is uploaded, process it
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Precip'] = pd.to_numeric(df['Precip'], errors='coerce')
    df['Month'] = df['Date'].dt.month

    if scale == 1:
        df['SPI_1'] = calculate_spi1_with_climatology(df)
        spi_col = 'SPI_1'
    else:
        df[f'SPI_{scale}'] = calculate_spi(df['Precip'], scale)
        spi_col = f'SPI_{scale}'

    # Show chart
    st.subheader("📈 SPI Time Series")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df[spi_col], color='royalblue', marker='o', linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f"SPI-{scale} Time Series", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("SPI Value")
    ax.grid(True, linestyle=':')
    st.pyplot(fig)

    # Show table
    st.subheader("🧾 Preview of Data")
    st.dataframe(df[['Date', 'Precip', spi_col]].head(20))

    # Download button
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download Result CSV",
        data=csv,
        file_name=f"spi_result_{scale}.csv",
        mime="text/csv"
    )

# Sidebar credit
st.sidebar.markdown("**Developed by: S M Samiul Islam / UIHILAB**")
