import re
import numpy as np
import pandas as pd

def clean_ph(value: str):
    """Extract numeric pH values from mixed strings; handle invalid or missing gracefully."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()

    # Handle textual descriptors
    if "nd" in text or "ud" in text or "n.d." in text or "unknown" in text:
        return np.nan
    if "negligible" in text:
        return 0.0

    # Regex for number (integer or float)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def clean_temperature(value: str):
    """Standardize temperature to °C from strings like '25C', '298K', or 'room temperature'."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()

    # Handle qualitative cases
    if "room" in text:
        return 22.0  # approximate room temperature
    if "nd" in text or "unknown" in text:
        return np.nan

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan

    try:
        num = float(match.group(1))
    except ValueError:
        return np.nan

    # Convert from K to C if unit suggests Kelvin
    if "k" in text and "°c" not in text:
        num = num - 273.15

    return num


def add_temperature_ph(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized pH_value and temperature_C columns to the dataset."""
    df["pH_value"] = df.get("pH", np.nan).apply(clean_ph)
    df["temperature_C"] = df.get("temperature", np.nan).apply(clean_temperature)
    return df
