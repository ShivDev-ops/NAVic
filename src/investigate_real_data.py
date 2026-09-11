"""
One-off investigation script: confirms real IO-VNBD file format assumptions
before they get baked into common.py.
"""
import pandas as pd
import numpy as np


def load_csv(path):
    df = pd.read_csv(path, encoding="latin1", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def check_clock_offset(s_path, v_path, candidate_shifts=(0, 3600, -3600)):
    s = load_csv(s_path)
    v = load_csv(v_path)

    s["dt"] = pd.to_datetime(
        s["DATE (YYYY-MO-DD HH-MI-SS_SSS)"], format="%Y-%m-%d %H:%M:%S:%f"
    )
    s["sec_since_midnight"] = (
        s["dt"].dt.hour * 3600
        + s["dt"].dt.minute * 60
        + s["dt"].dt.second
        + s["dt"].dt.microsecond / 1e6
    )

    for shift in candidate_shifts:
        t_shifted = s["sec_since_midnight"] - shift
        v_speed_interp = np.interp(
            t_shifted, v["Time Since Start of Day (seconds)"], v["Velocity (km/hr)"]
        )
        s_speed = s["GPS SPEED (Kmh)"]
        mask = s_speed.notna()
        corr = np.corrcoef(s_speed[mask], pd.Series(v_speed_interp)[mask])[0, 1]
        print(f"shift={shift:+}s -> correlation={corr:.4f}")


if __name__ == "__main__":
    check_clock_offset("../data/raw/S-M.csv", "../data/raw/V-M.csv")
