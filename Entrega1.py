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
# 8. BOOTSTRAPPING
# ==================================================
spot_curve = {}

def interp_spot(t):
    Ts = np.array(sorted(spot_curve.keys()))
    rs = np.array([spot_curve[x] for x in Ts])
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
        spot_curve[T] = r
        continue

    # -------- COUPON --------
    face = 100
    freq = 2
    c = face * cpn / 100 / freq
    n = int(np.round(T * freq))

    pv_known = 0.0
    for k in range(1, n):
        t = k / freq
        r_t = interp_spot(t)
        pv_known += c * np.exp(-r_t * t)

    r_T = -np.log((P - pv_known) / (face + c)) / T
    spot_curve[T] = r_T

# ==================================================
# 9. DATAFRAME DE SPOTS
# ==================================================
boot_df = pd.DataFrame({
    "T": list(spot_curve.keys()),
    "Spot": list(spot_curve.values())
}).sort_values("T")

# ==================================================
# 10. NELSON–SIEGEL OLS
# ==================================================
def ns_terms(T, lam):
    x = lam * T
    f1 = (1 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    return f1, f2

def nelson_siegel(T, b0, b1, b2, lam):
    f1, f2 = ns_terms(T, lam)
    return b0 + b1 * f1 + b2 * f2

lambda_grid = np.linspace(0.1, 5.0, 80)
best_sse = np.inf

for lam in lambda_grid:
    X, y = [], []
    for _, row in boot_df.iterrows():
        f1, f2 = ns_terms(row["T"], lam)
        X.append([1, f1, f2])
        y.append(row["Spot"])

    X = np.array(X)
    y = np.array(y)

    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = nelson_siegel(boot_df["T"], *betas, lam)
    sse = np.sum((y - fit) ** 2)

    if sse < best_sse:
        best_sse = sse
        best_params = (*betas, lam)

b0, b1, b2, lam = best_params

# ==================================================
# 11. PLOT
# ==================================================
T_grid = np.linspace(0.05, boot_df["T"].max(), 300)
curve = nelson_siegel(T_grid, b0, b1, b2, lam)

plt.figure(figsize=(8,5))
plt.scatter(boot_df["T"], 100 * boot_df["Spot"], label="Bootstrap spots")
plt.plot(T_grid, 100 * curve, label="Nelson–Siegel", linewidth=2)
plt.xlabel("Maturity (years)")
plt.ylabel("Spot rate (%)")
plt.title("Bootstrapped Yield Curve + Nelson–Siegel")
plt.legend()
plt.grid(True)
plt.show()

print("\nNelson–Siegel parameters (Bootstrap-based)")
print(f"beta0 = {b0:.6f}")
print(f"beta1 = {b1:.6f}")
print(f"beta2 = {b2:.6f}")
print(f"lambda = {lam:.4f}")
print(f"SSE = {best_sse:.3e}")
