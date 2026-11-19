'''
Plot the light curve of one Lasair object.

Set OBJECT_ID below, then run:
    python scripts/light_curves.py

Two figures are written. One is the full detection curve in AB magnitude.
The other is the same object cut by truncate() at nights N = 0, 2 and 5,
in flux, so the non-detections stay on the plot. Brighter is up on the
magnitude figure. Flux is in nanojansky.
'''

import json  # read the saved object record

import matplotlib.pyplot as plt  # draw the plot
import numpy as np  # log10 for the magnitude conversion
import pandas as pd  # table of detections
from paths import FIGS, INTERIM, RAW
from truncate import truncate

# Paste a diaObjectId here. This one is a TNS SN Ia with a rising light curve.
OBJECT_ID = "170019717277810735"
# Nights passed to truncate() for the comparison figure.
TRUNCATE_NIGHTS = (0, 2, 5)

# Colour used for each LSST filter. A band not listed gets a default blue.
BAND_COLOR = {
    "u": "#7b2cbf",
    "g": "#2ca02c",
    "r": "#d62728",
    "i": "#ff7f0e",
    "z": "#8c564b",
    "y": "#4d4d4d",
}

# Converts flux in nanojansky into AB magnitude.
AB_ZEROPOINT = 31.4


def load_object(object_id):
    """Read object_<id>.json, or download it if it is not here yet."""
    filename = RAW / "lasair" / "objects" / f"object_{object_id}.json"
    if filename.exists():
        print(f"Reading {filename.name}")
        with open(filename) as f:
            return json.load(f)  # the full record, as a Python dict

    # File is missing, so download just this one object.
    print(f"{filename.name} not found, downloading from Lasair")
    import lasair
    from settings import API_TOKEN

    client = lasair.lasair_client(API_TOKEN, endpoint="https://api.lasair.lsst.ac.uk/api")
    obj = client.object(object_id, lasair_added=True)  # full record plus light curve
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2, default=str)  # save it so the next run can reuse it
    print(f"Saved {filename.name}")
    return obj


def detections_table(obj):
    """Detections as a table of time, band, flux, and AB magnitude."""
    # diaSourcesList is the light curve: one row per detection.
    rows = obj.get("diaSourcesList") or []
    if not rows:
        raise SystemExit("This object has no diaSourcesList, so there is no light curve to plot.")

    df = pd.DataFrame(rows)  # list of detections -> table
    df = df.rename(columns={"midpointMjdTai": "mjd"})  # shorter name for the time column
    # Stop if a column the plot needs is missing.
    for col in ("mjd", "psfFlux", "psfFluxErr", "band"):
        if col not in df.columns:
            raise SystemExit(f"Light curve is missing the '{col}' column.")

    # Force these columns to be numbers. Bad values become NaN.
    df["psfFlux"] = pd.to_numeric(df["psfFlux"], errors="coerce")
    df["psfFluxErr"] = pd.to_numeric(df["psfFluxErr"], errors="coerce")
    df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")

    n_all = len(df)  # how many detections we started with
    # Magnitude needs a positive flux. Negative difference flux has no magnitude.
    df = df[df["psfFlux"] > 0].dropna(subset=["mjd", "psfFlux", "psfFluxErr", "band"])
    if df.empty:
        raise SystemExit(f"None of the {n_all} detections have positive flux.")

    # AB mag = -2.5 log10(flux) + zeropoint.
    df["mag"] = -2.5 * np.log10(df["psfFlux"]) + AB_ZEROPOINT
    # Error on the magnitude, from the flux error.
    df["magErr"] = (2.5 / np.log(10)) * (df["psfFluxErr"] / df["psfFlux"])
    df = df.sort_values("mjd")  # earliest detection first
    print(f"{len(df)} detections with positive flux ({n_all - len(df)} left out)")
    return df


def plot_lightcurve(df, object_id, classification):
    """Magnitude versus time, one colour per band. Brighter is up."""
    fig, ax = plt.subplots(figsize=(9, 5))  # one figure, one set of axes

    # Draw each filter as its own set of points.
    for band, group in df.groupby("band", sort=True):
        color = BAND_COLOR.get(str(band), "#1f77b4")  # default blue if the band is new
        ax.errorbar(
            group["mjd"], group["mag"], yerr=group["magErr"],  # time, mag, error bar
            fmt="o", ms=4, lw=0.8, capsize=2, color=color,
            label=f"{band} ({len(group)})",  # legend shows the band and how many points
        )

    title = f"Light curve {object_id}"
    if classification:
        title += f"  ({classification})"  # add Sherlock's class if we have one
    ax.set_title(title)
    ax.set_xlabel("MJD")  # time in modified Julian date
    ax.set_ylabel("AB magnitude")
    ax.invert_yaxis()  # smaller magnitude is brighter, so it goes up
    ax.legend(title="band", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()  # stop labels being cut off

    filename = FIGS / f"lightcurve_{object_id}.png"
    fig.savefig(filename, dpi=150)
    print(f"Saved {filename.name}")
    return fig


def plot_truncated(object_id):
    """One SN at nights 0, 2 and 5, from the shared Lasair light-curve table.

    The time axis is days after the first detection, and it is the same on
    every panel, out to the night-5 cut. A dashed line marks that panel's cut.
    Filled points are detections. Hollow points are forced photometry.
    Night 0 has no hollow points: truncate() drops them.
    """
    oid = int(object_id)
    lc = pd.read_parquet(
        INTERIM / "lightcurves_lasair.parquet",
        filters=[("objectId", "in", [oid])],
    )
    if lc.empty:
        raise SystemExit(f"{object_id} is not in lightcurves_lasair.parquet")
    objects = pd.read_parquet(
        INTERIM / "objects_lasair.parquet",
        columns=["objectId", "trueClass"],
        filters=[("objectId", "in", [oid])],
    )
    true_class = objects["trueClass"].iloc[0] if len(objects) else None
    label = true_class if pd.notna(true_class) else "unlabelled"

    t0 = lc.loc[lc["isDetection"], "mjd"].min()
    # Same vertical scale on every panel, taken from the longest cut.
    longest = truncate(lc, max(TRUNCATE_NIGHTS))
    low = (longest["flux"] - longest["fluxErr"]).min()
    high = (longest["flux"] + longest["fluxErr"]).max()
    pad = 0.05 * (high - low) if high > low else 1.0
    # A little space before the first detection, and just past the last cut.
    x_right = max(TRUNCATE_NIGHTS) + 0.5

    fig, axes = plt.subplots(1, len(TRUNCATE_NIGHTS), figsize=(12, 4), sharex=True, sharey=True)
    for ax, n in zip(axes, TRUNCATE_NIGHTS):
        part = truncate(lc, n).copy()
        part["day"] = part["mjd"] - t0
        # Nothing may land on or after the cut.
        assert part["day"].max() < n + 0.5
        if n < 1:
            assert part["isDetection"].all()
        for band, group in part.groupby("band", sort=True):
            color = BAND_COLOR.get(str(band), "#1f77b4")
            det = group[group["isDetection"]]
            non = group[~group["isDetection"]]
            ax.errorbar(non["day"], non["flux"], yerr=non["fluxErr"], fmt="o",
                        mfc="none", ms=4, lw=0.6, capsize=1, color=color)
            ax.errorbar(det["day"], det["flux"], yerr=det["fluxErr"], fmt="o",
                        ms=4, lw=0.6, capsize=1, color=color, label=band)
        ax.axvline(n + 0.5, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.4)
        n_det = int(part["isDetection"].sum())
        n_forced = int((~part["isDetection"]).sum())
        det_word = "detection" if n_det == 1 else "detections"
        ax.set_title(f"N = {n}\n{n_det} {det_word}, {n_forced} forced")
        ax.set_xlabel("days after first detection")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("flux (nJy)")
    axes[0].set_xlim(-0.3, x_right + 0.3)
    axes[0].set_ylim(low - pad, high + pad)
    axes[-1].legend(title="band", frameon=False, fontsize=8)
    fig.suptitle(f"{label}  {object_id}", y=1.02)
    fig.tight_layout()

    filename = FIGS / f"lightcurve_{object_id}_n0_n2_n5.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename.name}")
    return filename


def main():
    object_id = str(OBJECT_ID).strip()  # drop any accidental spaces
    if not object_id:
        raise SystemExit('Set OBJECT_ID at the top of light_curves.py, e.g. OBJECT_ID = "313756671828033555"')

    plot_truncated(object_id)  # nights 0, 2 and 5, from the parquet table

    obj = load_object(object_id)  # JSON on disk, or a fresh download
    df = detections_table(obj)  # the points that get plotted

    # Sherlock's class lives inside the extra Lasair block, if it is there.
    sherlock = (obj.get("lasairData") or {}).get("sherlock") or {}
    classification = sherlock.get("classification")

    print("Detections per band:")
    print(df.groupby("band").size().to_string(), "\n")  # how many points in u, g, r, ...

    # These are the columns written to the CSV. The plot uses the table, not this file.
    columns = ["mjd", "band", "psfFlux", "psfFluxErr", "mag", "magErr"]
    csv_name = INTERIM / f"lightcurve_{object_id}.csv"
    df[columns].to_csv(csv_name, index=False)
    print(f"Saved {csv_name.name}")

    plot_lightcurve(df, object_id, classification)
    plt.show()  # open the plot window


# Run main only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
