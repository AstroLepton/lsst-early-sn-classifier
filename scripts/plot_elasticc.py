import random  # not used yet
import pandas as pd  # read the Parquet tables
import matplotlib.pyplot as plt  # draw the plots
from paths import FIGS, INTERIM

# Colour used for each LSST filter, same as light_curves.py.
BAND_COLOR = {"u": "#7b2cbf", "g": "#2ca02c", "r": "#d62728",
              "i": "#ff7f0e", "z": "#8c564b", "y": "#4d4d4d"}

# Which objects to plot:
# "ids"    -> the objects listed in OBJECT_IDS
# "class"  -> the first N_PER_CLASS objects of CLASS_NAME
# "random" -> one random object per class
MODE = "random"
OBJECT_IDS = []  # e.g. [30239566, 53608147] for a good SN Ia and SN II
CLASS_NAME = "SNIa-SALT3"   # SNIa-SALT3, SNII-Templates, CLAGN, RRL
N_PER_CLASS = 20  # how many to plot in "class" mode
OUT_DIR = FIGS  # where the PNGs go
# Days to show before the first and after the last detection; None shows the whole survey.
ZOOM_DAYS = (50, 150)

# Small table, one row per object: objectId, ebv, trueClass.
objects = pd.read_parquet(INTERIM / "objects.parquet")
if MODE == "ids":
    picks = objects[objects["objectId"].isin(OBJECT_IDS)]  # only the ids I listed
elif MODE == "class":
    picks = objects[objects["trueClass"] == CLASS_NAME].head(N_PER_CLASS)  # first N of one class
else:
    # One random object from each class. random_state=0 means the same pick every run.
    picks = objects.groupby("trueClass").sample(1, random_state=0)
if picks.empty:
    raise SystemExit("No matching objects; check OBJECT_IDS or CLASS_NAME.")

ids = picks["objectId"].tolist()
# The light-curve file is 1.6 GB, so only load the rows for the picked objects.
lc = pd.read_parquet(INTERIM / "lightcurves.parquet",
                     filters=[("objectId", "in", ids)])

OUT_DIR.mkdir(exist_ok=True)  # make the plot folder if it is not there yet
# One plot per object.
for _, obj in picks.iterrows():
    points = lc[lc["objectId"] == obj["objectId"]]  # this object's measurements only
    fig, ax = plt.subplots(figsize=(9, 4))  # one figure, one set of axes
    # Draw each filter as its own set of points.
    for band, group in points.groupby("band"):
        color = BAND_COLOR[band]
        det = group[group["isDetection"]]  # clearly seen (roughly SNR > 5)
        non = group[~group["isDetection"]]  # measured, but too faint to count
        # Non-detections: small, faint, hollow circles.
        ax.errorbar(non["mjd"], non["flux"], yerr=non["fluxErr"], fmt="o",
                    mfc="none", ms=3, alpha=0.3, color=color)
        # Detections: solid circles. Only these get a legend entry.
        ax.errorbar(det["mjd"], det["flux"], yerr=det["fluxErr"], fmt="o",
                    ms=4, color=color, label=band)
    ax.axhline(0, color="k", lw=0.5)  # zero flux line
    detected = points[points["isDetection"]]
    # Zoom onto the event, since a SN only lasts weeks of a 3-year light curve.
    if ZOOM_DAYS and not detected.empty:
        # x: from ZOOM_DAYS[0] days before the first detection to ZOOM_DAYS[1] after the last.
        ax.set_xlim(detected["mjd"].min() - ZOOM_DAYS[0],
                    detected["mjd"].max() + ZOOM_DAYS[1])
        # y: fit only the points inside that window, error bars included.
        in_view = points[points["mjd"].between(*ax.get_xlim())]
        low = (in_view["flux"] - in_view["fluxErr"]).min()
        high = (in_view["flux"] + in_view["fluxErr"]).max()
        pad = 0.05 * (high - low)  # 5% gap above and below
        ax.set_ylim(low - pad, high + pad)
    ax.set_title(f"{obj['trueClass']}  {obj['objectId']}")
    ax.set_xlabel("MJD")  # time in modified Julian date
    ax.set_ylabel("flux (nJy)")
    ax.legend(ncol=6, frameon=False, fontsize=8)  # filters in one row
    fig.tight_layout()  # stop labels being cut off
    fig.savefig(OUT_DIR / f"{obj['trueClass']}_{obj['objectId']}.png", dpi=150)
    plt.close(fig)  # free the memory before the next object

print(f"Saved {len(picks)} plots to {OUT_DIR}/")
