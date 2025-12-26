import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import brentq

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
    "TYPE": "Instrument_Type",
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
        thirty_seconds = int(match.group(1))
        remainder = match.group(2)
        extra = UNICODE_FRACTIONS.get(remainder, 0.0)
        return integer + (thirty_seconds + extra) / 32
    except:
        return np.nan

df["Bid_clean"] = df["Bid"].apply(us_price_to_decimal)
df["Ask_clean"] = df["Ask"].apply(us_price_to_decimal)

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
    "Yield", "Price"
)

# ==================================================
# 6. QUOTED VALUE
# ==================================================
df.loc[df["Quote_type"] == "Price", "Quoted_value"] = (
    df.loc[df["Quote_type"] == "Price", ["Bid", "Ask"]]
      .applymap(us_price_to_decimal)
      .mean(axis=1)
)

df.loc[df["Quote_type"] == "Yield", "Quoted_value"] = (
    df.loc[df["Quote_type"] == "Yield", ["Bid", "Ask"]]
      .astype(float)
      .mean(axis=1)
)

# ==================================================
# 7. TIME TO MATURITY
# ==================================================
df["Maturity_Date"] = pd.to_datetime(df["Maturity_Date"])
df["Valuation_Date"] = pd.to_datetime(df["Valuation_Date"])
df["T"] = (df["Maturity_Date"] - df["Valuation_Date"]).dt.days / 365.25
df = df[df["T"] > 0]

# ==================================================
# 8. INSTRUMENT CLASS
# ==================================================
df["Instrument_class"] = np.where(df["Coupon_rate"] == 0, "Zero", "Coupon")

clean_df = df[[
    "RIC", "Name", "Coupon_rate", "Quoted_value",
    "Quote_type", "T", "Instrument_class"
]].dropna().reset_index(drop=True)

# ==================================================
# 9. SPOT RATE CONSTRUCTION
# ==================================================
def spot_from_zero_price(price, T):
    DF = price / 100
    return -np.log(DF) / T

def ytm_continuous(price, coupon_rate, T):
    face = 100
    freq = 2
    c = face * coupon_rate / 100 / freq

    def f(r):
        pv = sum(c * np.exp(-r * k / freq)
                 for k in range(1, int(np.round(T * freq)) + 1))
        pv += face * np.exp(-r * T)
        return pv - price

    return brentq(f, -0.5, 0.5)

spots = []

for _, row in clean_df.iterrows():
    if row["Instrument_class"] == "Zero":
        if row["Quote_type"] == "Yield":
            r = row["Quoted_value"] / 100      # direct spot
        else:
            r = spot_from_zero_price(row["Quoted_value"], row["T"])
    else:
        r = ytm_continuous(row["Quoted_value"], row["Coupon_rate"], row["T"])

    spots.append(r)

clean_df["Spot"] = spots

# ==================================================
# 10. NELSON–SIEGEL OLS
# ==================================================
def ns_terms(T, lam):
    x = lam * T
    f1 = np.where(T == 0, 1.0, (1 - np.exp(-x)) / x)
    f2 = f1 - np.exp(-x)
    return f1, f2

def nelson_siegel(T, b0, b1, b2, lam):
    f1, f2 = ns_terms(T, lam)
    return b0 + b1 * f1 + b2 * f2

lambda_grid = np.linspace(0.1, 5.0, 80)
best_sse = np.inf

for lam in lambda_grid:
    X, y = [], []
    for _, row in clean_df.iterrows():
        f1, f2 = ns_terms(row["T"], lam)
        X.append([1, f1, f2])
        y.append(row["Spot"])

    X = np.array(X)
    y = np.array(y)

    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = nelson_siegel(clean_df["T"], *betas, lam)
    sse = np.sum((y - fit) ** 2)

    if sse < best_sse:
        best_sse = sse
        best_params = (*betas, lam)

b0, b1, b2, lam = best_params

# ==================================================
# 11. PLOT
# ==================================================
T_grid = np.linspace(0.05, clean_df["T"].max(), 300)
curve = nelson_siegel(T_grid, b0, b1, b2, lam)

plt.figure(figsize=(8,5))
plt.scatter(clean_df["T"], 100 * clean_df["Spot"], label="Market spots")
plt.plot(T_grid, 100 * curve, label="Nelson–Siegel", linewidth=2)
plt.xlabel("Maturity (years)")
plt.ylabel("Spot rate (%)")
plt.title("Nelson–Siegel Yield Curve (OLS)")
plt.legend()
plt.grid(True)
plt.show()

print("\nNelson–Siegel parameters")
print(f"beta0 = {b0:.6f}")
print(f"beta1 = {b1:.6f}")
print(f"beta2 = {b2:.6f}")
print(f"lambda = {lam:.4f}")
print(f"SSE = {best_sse:.3e}")
