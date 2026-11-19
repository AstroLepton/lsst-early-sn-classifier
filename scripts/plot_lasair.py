'''
Plot Lasair light curves from the shared parquet tables.

Set MODE below, then run from the repo root:
    python scripts/plot_lasair.py

Reads objects_lasair.parquet and lightcurves_lasair.parquet.
Flux is in nanojansky, the same unit as the ELAsTiCC plots.
'''

import pandas as pd  # read the Parquet tables
import matplotlib.pyplot as plt  # draw the plots
from paths import FIGS, INTERIM

# Colour used for each LSST filter, same as plot_elasticc.py.
BAND_COLOR = {"u": "#7b2cbf", "g": "#2ca02c", "r": "#d62728",
              "i": "#ff7f0e", "z": "#8c564b", "y": "#4d4d4d"}

# Which objects to plot:
# "ids"    -> the objects listed in OBJECT_IDS
# "class"  -> the first N_PER_CLASS objects of CLASS_NAME
# "random" -> one random object per TNS type
MODE = "ids"
OBJECT_IDS = [170019717277810735] 
CLASS_NAME = "SN Ia"  # SN Ia, SN II, SN Ic-BL, TDE, AGN, ...
N_PER_CLASS = 20  # how many to plot in "class" mode
OUT_DIR = FIGS  # where the PNGs go
# Days to show before the first and after the last detection. None shows every point.
# Lasair has no measurements before the first detection, so the left side is empty.
ZOOM_DAYS = (50, 150)


def file_class(true_class):
    """TNS type safe to put in a file name. SN Ib/c must not make a folder."""
    if pd.isna(true_class) or str(true_class).strip() == "":
        return "unlabelled"
    return str(true_class).replace("/", "-")


# One row per object. trueClass is the TNS type, and two objects have none.
objects = pd.read_parquet(INTERIM / "objects_lasair.parquet")
if MODE == "ids":
    wanted = [int(i) for i in OBJECT_IDS]
    picks = objects[objects["objectId"].isin(wanted)]  # only the ids I listed
elif MODE == "class":
    picks = objects[objects["trueClass"] == CLASS_NAME].head(N_PER_CLASS)  # first N of one type
else:
    # One random object from each TNS type. random_state=0 means the same pick every run.
    # Rows with no type are left out of this mode.
    labelled = objects[objects["trueClass"].notna()]
    picks = labelled.groupby("trueClass").sample(1, random_state=0)
if picks.empty:
    raise SystemExit("No matching objects; check OBJECT_IDS or CLASS_NAME.")

ids = picks["objectId"].tolist()
lc = pd.read_parquet(INTERIM / "lightcurves_lasair.parquet",
                     filters=[("objectId", "in", ids)])

OUT_DIR.mkdir(exist_ok=True)  # make the plot folder if it is not there yet
# One plot per object.
for _, obj in picks.iterrows():
    points = lc[lc["objectId"] == obj["objectId"]]  # this object's measurements only
    if points.empty:
        print(f"{obj['objectId']} has no light-curve rows, skipped")
        continue
    fig, ax = plt.subplots(figsize=(9, 4))  # one figure, one set of axes
    # Draw each filter as its own set of points.
    for band, group in points.groupby("band"):
        color = BAND_COLOR.get(band, "#1f77b4")
        det = group[group["isDetection"]]  # diaSources
        non = group[~group["isDetection"]]  # forced points that are not a detection
        # Non-detections: small, faint, hollow circles.
        ax.errorbar(non["mjd"], non["flux"], yerr=non["fluxErr"], fmt="o",
                    mfc="none", ms=3, alpha=0.3, color=color)
        # Detections: solid circles. Only these get a legend entry.
        ax.errorbar(det["mjd"], det["flux"], yerr=det["fluxErr"], fmt="o",
                    ms=4, color=color, label=band)
    ax.axhline(0, color="k", lw=0.5)  # zero flux line
    detected = points[points["isDetection"]]
    if ZOOM_DAYS and not detected.empty:
        # x: from ZOOM_DAYS[0] days before the first detection to ZOOM_DAYS[1] after the last.
        ax.set_xlim(detected["mjd"].min() - ZOOM_DAYS[0],
                    detected["mjd"].max() + ZOOM_DAYS[1])
        # y: fit only the points inside that window, error bars included.
        in_view = points[points["mjd"].between(*ax.get_xlim())]
        low = (in_view["flux"] - in_view["fluxErr"]).min()
        high = (in_view["flux"] + in_view["fluxErr"]).max()
        pad = 0.05 * (high - low) if high > low else 1.0  # 5% gap above and below
        ax.set_ylim(low - pad, high + pad)
    label = obj["trueClass"] if pd.notna(obj["trueClass"]) else "unlabelled"
    ax.set_title(f"{label}  {obj['objectId']}")
    ax.set_xlabel("MJD")  # time in modified Julian date
    ax.set_ylabel("flux (nJy)")
    ax.legend(ncol=6, frameon=False, fontsize=8)  # filters in one row
    fig.tight_layout()  # stop labels being cut off
    fig.savefig(OUT_DIR / f"{file_class(obj['trueClass'])}_{obj['objectId']}.png", dpi=150)
    plt.close(fig)  # free the memory before the next object

print(f"Saved {len(picks)} plots to {OUT_DIR}/")
