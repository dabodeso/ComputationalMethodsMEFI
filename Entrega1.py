import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# ==================================================
# 1. LOAD DATA
# ==================================================
file_path = "GovernmentBondPrices_UnitedStates.xlsx"
df = pd.read_excel(file_path)

# ==================================================
# 2. COLUMN NORMALIZATION
# ==================================================
df = df.rename(columns={
    "Cpn": "Coupon_rate",
    "Mat Date": "Maturity_Date",
    "Date": "Valuation_Date"
})

# ==================================================
# 3. US PRICE CONVERSION (32nds)
# ==================================================
UNICODE_FRACTIONS = {
    '¼': 0.25, '½': 0.5, '¾': 0.75,
    '⅛': 0.125, '⅜': 0.375,
    '⅝': 0.625, '⅞': 0.875
}

def us_price_to_decimal(price):
    if isinstance(price, (int, float)):
        return float(price)
    try:
        main, frac = price.split('*')
        integer = float(main)
        match = re.match(r"(\d+)(.*)", frac)
        thirty = int(match.group(1))
        remainder = match.group(2)
        extra = UNICODE_FRACTIONS.get(remainder, 0.0)
        return integer + (thirty + extra) / 32
    except:
        return np.nan

# ==================================================
# 4. COUPON NORMALIZATION
# ==================================================
def normalize_coupon(c):
    if pd.isna(c):
        return 0.0
    if isinstance(c, str):
        c = c.replace('%', '').strip()
    c = float(c)
    return c * 100 if c <= 1 else c

df["Coupon_rate"] = df["Coupon_rate"].apply(normalize_coupon)

# ==================================================
# 5. QUOTE TYPE DETECTION
# ==================================================
def detect_quote_type(x):
    if isinstance(x, str) and '*' in x:
        return "Price"
    try:
        val = float(x)
        return "Yield" if val < 30 else "Price"
    except:
        return np.nan

df["Bid_type"] = df["Bid"].apply(detect_quote_type)
df["Ask_type"] = df["Ask"].apply(detect_quote_type)

df["Quote_type"] = np.where(
    (df["Bid_type"] == "Yield") | (df["Ask_type"] == "Yield"),
    "Yield",
    "Price"
)

df.loc[df["Quote_type"] == "Price", "Quoted_value"] = (
    df.loc[df["Quote_type"] == "Price", ["Bid", "Ask"]]
      .map(us_price_to_decimal)
      .mean(axis=1)
)

df.loc[df["Quote_type"] == "Yield", "Quoted_value"] = (
    df.loc[df["Quote_type"] == "Yield", ["Bid", "Ask"]]
      .astype(float)
      .mean(axis=1)
)

# ==================================================
# 6. TIME TO MATURITY  (ESTO FALTABA)
# ==================================================
df["Maturity_Date"] = pd.to_datetime(df["Maturity_Date"])
df["Valuation_Date"] = pd.to_datetime(df["Valuation_Date"])
df["T"] = (df["Maturity_Date"] - df["Valuation_Date"]).dt.days / 365.25
df = df[df["T"] > 0]

# ==================================================
# 7. INSTRUMENT CLASS
# ==================================================
df["Instrument_class"] = np.where(df["Coupon_rate"] == 0, "Zero", "Coupon")

clean_df = df[[
    "RIC", "Coupon_rate", "Quoted_value",
    "Quote_type", "T", "Instrument_class"
]].dropna().sort_values("T").reset_index(drop=True)

# ==================================================
# 7.5. DIAGNÓSTICO: Estadísticas de los datos
# ==================================================
print("\n" + "="*60)
print("DIAGNÓSTICO DE DATOS")
print("="*60)
print(f"\nTotal de instrumentos después de limpieza: {len(clean_df)}")
print(f"\nPor tipo de instrumento:")
print(clean_df["Instrument_class"].value_counts())
print(f"\nPor tipo de cotización:")
print(clean_df["Quote_type"].value_counts())
print(f"\nBonos con cupón por tipo de cotización:")
if len(clean_df[clean_df["Instrument_class"] == "Coupon"]) > 0:
    print(clean_df[clean_df["Instrument_class"] == "Coupon"]["Quote_type"].value_counts())
print(f"\nRango de vencimientos: {clean_df['T'].min():.2f} - {clean_df['T'].max():.2f} años")
print("="*60 + "\n")

# ==================================================
# 8. BOOTSTRAPPING
# ==================================================
# Guardar todos los spots (no solo uno por T) para usar todos en OLS
spot_curve_list = []  # Lista de (T, spot, instrument_type) para guardar todos los spots
spot_curve_dict = {}  # Diccionario para interpolación (promedio por T)

def interp_spot(t):
    """Interpola el spot rate para el tiempo t usando los spots conocidos."""
    if not spot_curve_dict:
        return 0.0  # Si no hay spots, retornar 0
    Ts = np.array(sorted(spot_curve_dict.keys()))
    rs = np.array([spot_curve_dict[x] for x in Ts])
    if t <= Ts[0]:
        return rs[0]  # Extrapolación hacia atrás: usar el primer spot
    if t >= Ts[-1]:
        return rs[-1]  # Extrapolación hacia adelante: usar el último spot
    return np.interp(t, Ts, rs)

for _, row in clean_df.iterrows():
    T = row["T"]
    P = row["Quoted_value"]
    cpn = row["Coupon_rate"]

    # -------- ZERO --------
    if row["Instrument_class"] == "Zero":
        if row["Quote_type"] == "Yield":
            r = P / 100
        else:
            r = -np.log(P / 100) / T
        spot_curve_list.append((T, r, "Zero"))
        # Para interpolación, guardar promedio si hay múltiples bonos con mismo T
        if T in spot_curve_dict:
            # Promediar si ya existe (para interpolación)
            spot_curve_dict[T] = (spot_curve_dict[T] + r) / 2
        else:
            spot_curve_dict[T] = r
        continue

    # -------- COUPON --------
    face = 100
    freq = 2
    c = face * cpn / 100 / freq
    n = int(np.round(T * freq))
    
    # Si está cotizado en yield, usar YTM como proxy del spot rate
    # (según el enunciado: "For coupon-paying bonds, the YTM will be used as a proxy for the spot rates")
    if row["Quote_type"] == "Yield":
        r_T = P / 100  # YTM como proxy del spot rate (convertir de porcentaje a decimal)
        spot_curve_list.append((T, r_T, "Coupon"))
        # Para interpolación, guardar promedio si hay múltiples bonos con mismo T
        if T in spot_curve_dict:
            spot_curve_dict[T] = (spot_curve_dict[T] + r_T) / 2
        else:
            spot_curve_dict[T] = r_T
        continue
    
    # Si está cotizado en precio, hacer bootstrapping
    # Calcular PV de cupones conocidos usando spots interpolados
    pv_known = 0.0
    for k in range(1, n):
        t = k / freq
        if t < T:  # Solo cupones antes del vencimiento
            r_t = interp_spot(t)
            pv_known += c * np.exp(-r_t * t)
    
    # Calcular el spot rate para el último flujo (último cupón + principal)
    # P = PV(cupones conocidos) + PV(último cupón + principal)
    # P - pv_known = (face + c) * exp(-r_T * T)
    pv_final = P - pv_known
    
    # Validar que el precio final sea positivo y razonable
    if pv_final > 0 and pv_final < (face + c) * 2:  # Validación razonable
        r_T = -np.log(pv_final / (face + c)) / T
        spot_curve_list.append((T, r_T, "Coupon"))
        # Para interpolación, guardar promedio si hay múltiples bonos con mismo T
        if T in spot_curve_dict:
            spot_curve_dict[T] = (spot_curve_dict[T] + r_T) / 2
        else:
            spot_curve_dict[T] = r_T

# ==================================================
# 9. DATAFRAME DE SPOTS (TODOS los spots, no solo uno por T)
# ==================================================
boot_df = pd.DataFrame(spot_curve_list, columns=["T", "Spot", "Instrument_type"]).sort_values("T")

# Diagnóstico: verificar cuántos spots se generaron
print("\n" + "="*60)
print("RESULTADOS DEL BOOTSTRAPPING")
print("="*60)
print(f"\nTotal de spots generados: {len(boot_df)}")
print(f"\nSpots únicos por T: {boot_df['T'].nunique()}")
print(f"\nRango de vencimientos con spots: {boot_df['T'].min():.2f} - {boot_df['T'].max():.2f} años")
print(f"\nRango de spot rates: {boot_df['Spot'].min()*100:.4f}% - {boot_df['Spot'].max()*100:.4f}%")
print("="*60 + "\n")

# ==================================================
# 10. NELSON–SIEGEL OLS (siguiendo exactamente el notebook de clase)
# ==================================================
def ns_functions(tau, lam):
    """
    Compute the Nelson–Siegel loading functions f1(t) and f2(t).
    Siguiendo exactamente la notación del notebook de clase.
    """
    # Manejar división por cero cuando tau es muy pequeño
    ratio = tau / lam
    # Usar aproximación de Taylor cuando ratio -> 0: (1 - exp(-x))/x -> 1 cuando x -> 0
    f1 = np.where(ratio < 1e-10, 1.0, (1 - np.exp(-ratio)) / ratio)
    f2 = f1 - np.exp(-ratio)
    return f1, f2

def nelson_siegel_curve(tau, beta0, beta1, beta2, lam):
    """
    Compute the Nelson–Siegel yield curve for given parameters.
    Siguiendo exactamente la notación del notebook de clase.
    """
    f1, f2 = ns_functions(tau, lam)
    return beta0 + beta1 * f1 + beta2 * f2

# Extraer todos los spots observados (como en el notebook)
tau_obs = boot_df["T"].values
y_obs = boot_df["Spot"].values

# Grid search sobre lambda + OLS en betas (exactamente como en el notebook)
lambda_grid = np.linspace(0.1, 5.0, 100)
best_sse = np.inf
best_params = None
best_lam = None

for lam in lambda_grid:
    # Calculate the base functions
    f1, f2 = ns_functions(tau_obs, lam)
    
    # Generate the X matrix (exactamente como en el notebook)
    X = np.column_stack([np.ones_like(tau_obs), f1, f2])
    
    # Solve the equation by OLS (exactamente como en el notebook)
    betas, residuals, rank, s = np.linalg.lstsq(X, y_obs, rcond=None)
    
    # Estimate the Spot values from the betas
    y_fit = X @ betas
    
    # Estimate the sum of the square of the errors
    sse = np.sum((y_obs - y_fit)**2)
    
    # We keep the smallest value of the sum of squared residuals and its associated parameters
    if sse < best_sse:
        best_sse = sse
        best_params = betas
        best_lam = lam

# Identify the betas
beta0_ols, beta1_ols, beta2_ols = best_params

# Estimate the spot rates
y_fit_ols = nelson_siegel_curve(tau_obs, beta0_ols, beta1_ols, beta2_ols, best_lam)

# Estimate the errors
sse_ols = np.sum((y_obs - y_fit_ols)**2)

# ==================================================
# 11. PLOT (siguiendo el estilo del notebook)
# ==================================================
T_grid = np.linspace(0.05, boot_df["T"].max(), 300)
curve = nelson_siegel_curve(T_grid, beta0_ols, beta1_ols, beta2_ols, best_lam)

plt.figure(figsize=(8,5))

# Separar bonos Zero y Coupon para colorearlos diferente
zero_bonds = boot_df[boot_df["Instrument_type"] == "Zero"]
coupon_bonds = boot_df[boot_df["Instrument_type"] == "Coupon"]

# Bonos Zero en azul
if len(zero_bonds) > 0:
    plt.scatter(zero_bonds["T"], 100 * zero_bonds["Spot"], 
                color='blue', label="Zero-coupon bonds", alpha=0.7, s=50)

# Bonos con cupón en verde claro
if len(coupon_bonds) > 0:
    plt.scatter(coupon_bonds["T"], 100 * coupon_bonds["Spot"], 
                color='lightgreen', label="Coupon bonds", alpha=0.7, s=50)

# Curva Nelson-Siegel ajustada
plt.plot(T_grid, 100 * curve, label="Nelson–Siegel (fitted)", linewidth=2, color='red')

plt.xlabel("Maturity (years)")
plt.ylabel("Spot rate (%)")
plt.title("Bootstrapped Yield Curve + Nelson–Siegel OLS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==================================================
# 12. RESULTADOS (siguiendo el formato del notebook)
# ==================================================
print("\n" + "="*50)
print("        Nelson–Siegel Calibration Results")
print("="*50)
print("\n[ OLS with Lambda Grid Search ]")
print(f"  • beta₀ (level)     : {beta0_ols:>12.6f}")
print(f"  • beta₁ (slope)     : {beta1_ols:>12.6f}")
print(f"  • beta₂ (curvature) : {beta2_ols:>12.6f}")
print(f"  • λ (decay factor)  : {best_lam:>12.6f}")
print(f"  • SSE               : {sse_ols:>12.6e}")
print("="*50 + "\n")
