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
    STRIPS y T-Bills ya vienen en decimal.
    """
    if isinstance(price, (int, float)):
        return price

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

# --------------------------------------------------
# 4. Precio mid
# --------------------------------------------------
df["Price_mid"] = (df["Bid_clean"] + df["Ask_clean"]) / 2

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

# --------------------------------------------------
# 7. Limpieza final
# --------------------------------------------------
clean_df = df[[
    "RIC",
    "Name",
    "Instrument_Type",
    "Coupon_rate",
    "Price_mid",
    "T",
    "Instrument_class"
]].dropna()

clean_df.reset_index(drop=True, inplace=True)

# --------------------------------------------------
# 8. Resultado
# --------------------------------------------------
print(clean_df.head())
