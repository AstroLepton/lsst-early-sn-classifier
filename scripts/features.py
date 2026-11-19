'''
Features of one truncated light curve.

Missing inputs stay NaN. A zero would mean "we measured zero", which is
a different statement, and LightGBM already accepts NaN.

Night 0 has no forced photometry, so pre_chi2 and pre_snr are NaN there.
'''

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

# Same nanojansky zeropoint as the light-curve tables.
AB_ZEROPOINT = 31.4

FEATURE_NAMES = [
    "n_det", "n_bands", "t_span", "log_peak_flux", "rise_rate",
    "colour_gr", "colour_ri", "pre_chi2", "pre_snr", "neg_frac",
    "any_neg_det", "host_sep", "mwebv", "abs_gal_b",
]


def abs_galactic_latitude(ra, dec):
    """|Galactic latitude| in degrees. ra and dec are arrays in degrees."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    out = np.full(np.broadcast(ra, dec).shape, np.nan)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(ok):
        return out
    coord = SkyCoord(ra[ok], dec[ok], unit="deg", frame="icrs")
    out[ok] = np.abs(coord.galactic.b.deg)
    return out


def _blank():
    return {name: np.nan for name in FEATURE_NAMES}


def _mag(flux):
    return AB_ZEROPOINT - 2.5 * np.log10(flux)


def _rise_rate(det):
    """Magnitude brightened per day in the band with the most detections.

    Positive means the object got brighter. A Type Ia rises at about
    0.1 to 0.3 mag/day; a flare is steeper.
    """
    if len(det) < 2:
        return np.nan
    counts = det.groupby("band").size()
    band = sorted(counts.index, key=lambda name: (-int(counts[name]), str(name)))[0]
    series = det.loc[det["band"] == band].sort_values("mjd")
    first = series.iloc[0]
    last = series.iloc[-1]
    dt = float(last["mjd"] - first["mjd"])
    if dt <= 0 or first["flux"] <= 0 or last["flux"] <= 0:
        return np.nan
    if not np.isfinite(first["flux"]) or not np.isfinite(last["flux"]):
        return np.nan
    return float((_mag(first["flux"]) - _mag(last["flux"])) / dt)


def _colour(lc, band_1, band_2):
    """-2.5 log10(f1/f2) for the closest positive detections within 1 day."""
    a = lc[(lc["isDetection"]) & (lc["band"] == band_1) & (lc["flux"] > 0)]
    b = lc[(lc["isDetection"]) & (lc["band"] == band_2) & (lc["flux"] > 0)]
    if a.empty or b.empty:
        return np.nan
    a = a.sort_values("mjd")
    b = b.sort_values("mjd")
    dt = np.abs(a["mjd"].to_numpy()[:, None] - b["mjd"].to_numpy()[None, :])
    i, j = np.unravel_index(np.argmin(dt), dt.shape)
    if dt[i, j] > 1.0:
        return np.nan
    return float(-2.5 * np.log10(a["flux"].iloc[i] / b["flux"].iloc[j]))


def _pre_detection(lc, det):
    """Forced points in [t0 - 30, t0). Empty when the curve has none."""
    if det.empty:
        return lc.iloc[0:0]
    t0 = float(det["mjd"].min())
    pre = lc[(~lc["isDetection"]) & (lc["mjd"] >= t0 - 30.0) & (lc["mjd"] < t0)]
    ok = np.isfinite(pre["flux"]) & np.isfinite(pre["fluxErr"]) & (pre["fluxErr"] > 0)
    return pre.loc[ok]


def compute_features(lc, host_sep, mwebv, abs_gal_b):
    """One truncated light curve plus the object's fixed context columns.

    lc columns: mjd, band, flux, fluxErr, isDetection.
    host_sep, mwebv and abs_gal_b are the same at every night.
    """
    row = _blank()
    row["host_sep"] = host_sep
    row["mwebv"] = mwebv
    row["abs_gal_b"] = abs_gal_b
    if lc.empty:
        return row

    det = lc.loc[lc["isDetection"]]
    row["n_det"] = int(len(det))
    row["n_bands"] = int(det["band"].nunique()) if len(det) else 0
    if len(det) >= 2:
        row["t_span"] = float(det["mjd"].max() - det["mjd"].min())
    peak = det["flux"].max() if len(det) else np.nan
    if np.isfinite(peak) and peak > 0:
        row["log_peak_flux"] = float(np.log10(peak))
    row["rise_rate"] = _rise_rate(det)
    row["colour_gr"] = _colour(lc, "g", "r")
    row["colour_ri"] = _colour(lc, "r", "i")

    pre = _pre_detection(lc, det)
    if len(pre):
        flux = pre["flux"].to_numpy()
        sigma = pre["fluxErr"].to_numpy()
        row["pre_chi2"] = float(np.mean((flux / sigma) ** 2))
        weight = 1.0 / sigma ** 2
        # Inverse-variance mean, divided by the error on that mean.
        row["pre_snr"] = float(np.sum(flux * weight) / np.sqrt(np.sum(weight)))

    usable = lc[np.isfinite(lc["flux"]) & np.isfinite(lc["fluxErr"]) & (lc["fluxErr"] > 0)]
    if len(usable):
        row["neg_frac"] = float(np.mean(usable["flux"] < -3.0 * usable["fluxErr"]))
    if len(det):
        row["any_neg_det"] = float((det["flux"] < 0).any())
    return row
