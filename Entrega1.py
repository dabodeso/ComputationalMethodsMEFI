import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Cargar datos
# --------------------------------------------------
file_path = "GovernmentBondPrices_UnitedStates.xlsx"
df = pd.read_excel(file_path)

# --------------------------------------------------
# 2. Normalizar nombres de columnas Prueba
# --------------------------------------------------
df = df.rename(columns={
    "Cpn": "Coupon_rate",
    "Mat Date": "Maturity_Date",
    "TYPE": "Instrument_Type",
    "Date": "Valuation_Date"
})

# --------------------------------------------------
# 3. Conversión de precios US (32avos)
# --------------------------------------------------
UNICODE_FRACTIONS = {
    '¼': 0.25,
    '½': 0.5,
    '¾': 0.75,
    '⅛': 0.125,
    '⅜': 0.375,
    '⅝': 0.625,
    '⅞': 0.875
}

def us_price_to_decimal(price):
    """
    Convierte precios tipo '99*19⅝' a decimal.
    Si el precio ya viene en decimal (Bills / STRIPS), lo devuelve tal cual.
    """
    if isinstance(price, (int, float)):
        return float(price)

    try:
        main, frac = price.split('*')
        integer = float(main)

        match = re.match(r"(\d+)(.*)", frac)
        thirty_seconds = int(match.group(1))
        remainder = match.group(2)

        extra = UNICODE_FRACTIONS.get(remainder, 0.0)

        return integer + (thirty_seconds + extra) / 32

    except Exception:
        return np.nan


df["Bid_clean"] = df["Bid"].apply(us_price_to_decimal)
df["Ask_clean"] = df["Ask"].apply(us_price_to_decimal)

def normalize_coupon(coupon):
    """
    Normaliza el cupón para que SIEMPRE esté en porcentaje.
    Ej:
    3.5     -> 3.5
    3.5%    -> 3.5
    0.035   -> 3.5
    """
    if pd.isna(coupon):
        return np.nan

    # Si viene como string, limpiar %
    if isinstance(coupon, str):
        coupon = coupon.strip().replace('%', '')

    try:
        coupon = float(coupon)
    except ValueError:
        return np.nan

    # Si está en formato decimal, convertir a %
    if coupon <= 1:
        coupon *= 100

    return coupon


df["Coupon_rate"] = df["Coupon_rate"].apply(normalize_coupon)

# --------------------------------------------------
# 4. Detección correcta: precio vs yield
# --------------------------------------------------

def detect_quote_type(x):
    """
    Detecta automáticamente si la cotización es precio o yield.
    """
    if isinstance(x, str) and '*' in x:
        return "Price"        # formato 32avos
    try:
        val = float(x)
        if val < 30:          # yields suelen ser 0–10 aprox.
            return "Yield"
        else:
            return "Price"
    except:
        return np.nan


df["Bid_type"] = df["Bid"].apply(detect_quote_type)
df["Ask_type"] = df["Ask"].apply(detect_quote_type)

# Si cualquiera de las dos columnas indica yield → tratamos como yield
df["Quote_type"] = np.where(
    (df["Bid_type"] == "Yield") | (df["Ask_type"] == "Yield"),
    "Yield",
    "Price"
)

# --------------------------------------------------
# 4.1 Calcular el valor limpio según tipo
# --------------------------------------------------

# Para precios → usamos la función de conversión correcta
df.loc[df["Quote_type"] == "Price", "Quoted_value"] = (
    df.loc[df["Quote_type"] == "Price", ["Bid", "Ask"]]
      .applymap(us_price_to_decimal)
      .mean(axis=1)
)

# Para yields → promedio directo como yield %
df.loc[df["Quote_type"] == "Yield", "Quoted_value"] = (
    df.loc[df["Quote_type"] == "Yield", ["Bid", "Ask"]]
      .astype(float)
      .mean(axis=1)
)

# --------------------------------------------------
# 5. Fechas y tiempo a vencimiento
# --------------------------------------------------
df["Maturity_Date"] = pd.to_datetime(df["Maturity_Date"])
df["Valuation_Date"] = pd.to_datetime(df["Valuation_Date"])

df["T"] = (df["Maturity_Date"] - df["Valuation_Date"]).dt.days / 365.25
df = df[df["T"] > 0]

# --------------------------------------------------
# 6. Clasificación de instrumentos
# --------------------------------------------------
df["Instrument_class"] = np.where(
    df["Coupon_rate"] == 0,
    "Zero",
    "Coupon"
)

clean_df = df[[
    "RIC",
    "Name",
    "Instrument_Type",
    "Coupon_rate",
    "Quoted_value",
    "Quote_type",
    "T",
    "Instrument_class"
]].dropna()


clean_df.reset_index(drop=True, inplace=True)

# --------------------------------------------------
# 8. Resultado
# --------------------------------------------------
#pd.set_option("display.max_columns", None)
#pd.set_option("display.width", None)

print(clean_df)

# --------------------------------------------------
# 9. Task 1 – OLS sobre SPOT RATES (según enunciado)
# --------------------------------------------------
# Reglas:
# - ZEROS → spot exacto (continuo)
# - CUPONES → YTM continuo como proxy del spot
# - Ajustamos Nelson–Siegel con OLS sobre los spots

import numpy as np
from scipy.optimize import brentq


# ------------------------ 9.1 Nelson–Siegel ------------------------

def ns_terms(T, lam):
    f1 = (1 - np.exp(-lam * T)) / (lam * T)
    f2 = f1 - np.exp(-lam * T)
    return f1, f2

def nelson_siegel_rate(T, beta0, beta1, beta2, lam):
    f1, f2 = ns_terms(T, lam)
    return beta0 + beta1 * f1 + beta2 * f2


# ------------------------ 9.2 Spot de ZERO-coupon ------------------------

def spot_from_zero(price, T):
    # DF = price / 100  →  r = - ln(DF) / T
    DF = price / 100.0
    return -np.log(DF) / T


# ------------------------ 9.3 YTM continuo (proxy spot) ------------------------

def ytm_continuous(price, coupon_rate, T):
    face = 100
    freq = 2
    c = face * coupon_rate / 100 / freq

    # función de valoración con descuento continuo
    def f(r):
        pv = 0.0
        for k in range(1, int(np.round(T * freq)) + 1):
            t = k / freq
            pv += c * np.exp(-r * t)
        pv += face * np.exp(-r * T)
        return pv - price

    # buscamos r entre -50% y 50%
    return brentq(f, -0.5, 0.5)


# ------------------------ 9.4 Construimos la curva de SPOT ------------------------

spots = []

for _, row in clean_df.iterrows():
    T = row["T"]
    price = row["Quoted_value"]
    coup = row["Coupon_rate"]

    if row["Instrument_class"] == "Zero":
        r = spot_from_zero(price, T)

    else:   # bono con cupón
        r = ytm_continuous(price, coup, T)

    spots.append(r)

clean_df["Spot"] = spots


# ------------------------ 9.5 OLS + grid search sobre λ ------------------------

lambda_grid = np.linspace(0.1, 5.0, 80)

best_sse = np.inf
best_params = None

for lam in lambda_grid:

    X = []
    y = []

    for _, row in clean_df.iterrows():
        T = row["T"]
        spot = row["Spot"]

        f1, f2 = ns_terms(T, lam)

        X.append([1.0, f1, f2])
        y.append(spot)

    X = np.array(X)
    y = np.array(y)

    betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1, b2 = betas

    spot_fit = nelson_siegel_rate(clean_df["T"], b0, b1, b2, lam)
    sse = np.sum((y - spot_fit) ** 2)

    if sse < best_sse:
        best_sse = sse
        best_params = (b0, b1, b2, lam)

beta0_ols, beta1_ols, beta2_ols, best_lam = best_params
sse_ols = best_sse


# ------------------------ 9.6 Construimos la CURVA DE YIELDS ------------------------

T_grid = np.linspace(0.1, clean_df["T"].max(), 200)

spot_curve_ols = nelson_siegel_rate(
    T_grid,
    beta0_ols, beta1_ols, beta2_ols, best_lam
)

ols_curve = pd.DataFrame({
    "T": T_grid,
    "Spot_OLS": spot_curve_ols
})


# ------------------------ 9.7 Resultados ------------------------

print("\n" + "="*50)
print("        Nelson–Siegel Calibration Results")
print("="*50)

print("\n[ OLS — Spot-based (consistent with assignment) ]")
print(f"  • beta₀ (level)     : {beta0_ols:>10.6f}")
print(f"  • beta₁ (slope)     : {beta1_ols:>10.6f}")
print(f"  • beta₂ (curvature) : {beta2_ols:>10.6f}")
print(f"  • λ (decay factor)  : {best_lam:>10.6f}")
print(f"  • SSE (spots)       : {sse_ols:>10.6e}")

print("="*50 + "\n")


# ------------------------ 9.8 Plot ------------------------

plt.figure(figsize=(8,5))
plt.plot(ols_curve["T"], 100 * ols_curve["Spot_OLS"])
plt.xlabel("Maturity (years)")
plt.ylabel("Spot rate (%)")
plt.title("Nelson–Siegel — OLS on Spot Rates")
plt.grid(True)
plt.show()
