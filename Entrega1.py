import pandas as pd
import numpy as np
import re

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
# 4. Valor cotizado (precio o yield)
# --------------------------------------------------
df["Quoted_value"] = (df["Bid_clean"] + df["Ask_clean"]) / 2

df["Quote_type"] = np.where(
    df["Instrument_Type"] == "STR",
    "Yield",   # Bills / STR cotizan en yield
    "Price"    # Notes / Bonds cotizan en precio
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
# 9. Task 1 – Método 1: OLS (Ordinary Least Squares)
# --------------------------------------------------
# Calibración del modelo Nelson–Siegel mediante OLS.
# Estrategia correcta:
# 1) OLS sobre yields SOLO con bonos zero-coupon
# 2) Búsqueda en rejilla sobre λ
# 3) Selección del λ que minimiza el error EN PRECIOS

import numpy as np

# --------------------------------------------------
# 9.1 Modelo Nelson–Siegel (tipos spot)
# --------------------------------------------------

def nelson_siegel_rate(T, beta0, beta1, beta2, lambd):
    term1 = (1 - np.exp(-lambd * T)) / (lambd * T)
    term2 = term1 - np.exp(-lambd * T)
    return beta0 + beta1 * term1 + beta2 * term2


# --------------------------------------------------
# 9.2 Factores de descuento
# --------------------------------------------------

def discount_factor_ns(T, beta0, beta1, beta2, lambd):
    r = nelson_siegel_rate(T, beta0, beta1, beta2, lambd)
    return np.exp(-r * T)


# --------------------------------------------------
# 9.3 Precio de un bono (pagos semestrales)
# --------------------------------------------------

def bond_price_ns(T, coupon_rate, beta0, beta1, beta2, lambd):
    face_value = 100
    freq = 2

    n_periods = int(np.round(T * freq))
    coupon = face_value * coupon_rate / 100 / freq

    price = 0.0
    for i in range(1, n_periods + 1):
        t_i = i / freq
        df = discount_factor_ns(t_i, beta0, beta1, beta2, lambd)
        price += coupon * df

    df_T = discount_factor_ns(T, beta0, beta1, beta2, lambd)
    price += face_value * df_T

    return price


# --------------------------------------------------
# 9.4 OLS con búsqueda en rejilla para λ
# --------------------------------------------------

lambda_grid = np.linspace(0.1, 10.0, 150)

best_sse = np.inf
best_params = None

# SOLO zero-coupon para la regresión
zeros = clean_df[clean_df["Instrument_class"] == "Zero"]
zeros = zeros[(zeros["T"] > 0.1) & (zeros["T"] < 25)]

for lam in lambda_grid:

    X = []
    y_yield = []

    # --- OLS sobre ZEROS (yields correctos) ---
    for _, row in zeros.iterrows():
        T = row["T"]

        f1 = 1.0
        f2 = (1 - np.exp(-lam * T)) / (lam * T)
        f3 = f2 - np.exp(-lam * T)

        X.append([f1, f2, f3])

        DF = row["Quoted_value"] / 100
        y = -np.log(DF) / T
        y_yield.append(y)

    X = np.array(X)
    y_yield = np.array(y_yield)

    beta, _, _, _ = np.linalg.lstsq(X, y_yield, rcond=None)
    b0, b1, b2 = beta

    # --- SSE sobre TODOS los bonos (precio) ---
    sse = 0.0
    for _, row in clean_df.iterrows():
        price_model = bond_price_ns(
            row["T"], row["Coupon_rate"],
            b0, b1, b2, lam
        )
        sse += (price_model - row["Quoted_value"])**2

    if sse < best_sse:
        best_sse = sse
        best_params = (b0, b1, b2, lam)

beta0_ols, beta1_ols, beta2_ols, best_lam = best_params
sse_ols = best_sse


# --------------------------------------------------
# 9.5 Curva OLS estimada
# --------------------------------------------------

T_grid = np.linspace(0.25, clean_df["T"].max(), 200)

spot_ols = nelson_siegel_rate(
    T_grid,
    beta0_ols, beta1_ols, beta2_ols, best_lam
)

ols_curve = pd.DataFrame({
    "T": T_grid,
    "Spot_Rate_OLS": spot_ols
})


# --------------------------------------------------
# 9.6 Presentación de resultados (formato informe)
# --------------------------------------------------

print("\n" + "="*50)
print("        Nelson–Siegel Calibration Results")
print("="*50)

print("\n[ OLS with Lambda Grid Search ]")
print(f"  • beta₀ (level)     : {beta0_ols:>10.6f}")
print(f"  • beta₁ (slope)     : {beta1_ols:>10.6f}")
print(f"  • beta₂ (curvature) : {beta2_ols:>10.6f}")
print(f"  • λ (decay factor)  : {best_lam:>10.6f}")
print(f"  • SSE               : {sse_ols:>10.6e}")

print("="*50 + "\n")

