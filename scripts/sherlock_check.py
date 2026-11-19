'''
Measure how well Sherlock separates supernovae from impostors,
using objects whose true class is known from a TNS spectrum.

Input: tns_labelled.csv (made by lasair_phase1.py, Query 2)
Output: printed tables, and sherlock_performance.csv

The three numbers it measures, for each choice of which Sherlock
classes to keep:
  completeness  = of the real supernovae, what fraction Sherlock keeps
  purity        = of what Sherlock keeps, what fraction are real supernovae
  rejection     = of the real impostors, what fraction Sherlock removes

SN - offset from nucleus
NT (nuclear transient)- inside nucleas of galaxy
ORPHAN - no nucleus or transient (included in case too faint to be detected)

'''

import numpy as np  # square roots for the uncertainty
import pandas as pd  # read the labelled table and write the results
from labels import broad_class  # SN / not SN, shared with the ELAsTiCC labels
from paths import RAW, TABLES

# Which Sherlock classes count as "keep" in each cut we compare.
KEEP_CUTS = {
    "SN only": {"SN"},
    "SN + NT": {"SN", "NT"},
    "SN + NT + ORPHAN": {"SN", "NT", "ORPHAN"},
}


def fraction(k, n, z=1.0):  # k successes out of n trials; z=1 is about 68%
    '''k/n with its 1sigma (68%) Wilson uncertainty interval.

    The Wilson interval behaves sensibly for small n and for fractions
    near 0 or 1, unlike the simple sqrt(p(1-p)/n) error bar.
    '''
    if n == 0:
        return np.nan, np.nan, np.nan  # no objects, so no fraction
    p = k / n  # the plain fraction
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom  # Wilson centre
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom  # half-width
    return p, centre - half, centre + half  # fraction, low edge, high edge


def show(name, value):
    p, lo, hi = value  # unpack fraction and its range
    if np.isnan(p):
        return f"{name}: no objects"
    return f"{name}: {p:.3f}  (68% range {lo:.3f} to {hi:.3f})"



# Load and label

df = pd.read_csv(RAW / "lasair" / "tns_labelled.csv")
df = df.drop_duplicates(subset="tns_name")  # one row per TNS object, not per Lasair id
df["truth"] = df["type"].apply(broad_class)  # SN, not SN, or blank
df["classification"] = df["classification"].fillna("NULL")  # missing Sherlock class -> "NULL"

print("How each TNS type was grouped (check this looks right):")
print(df.assign(truth=df["truth"].fillna("left out"))
        .groupby(["truth", "type"], dropna=False).size().to_string(), "\n")

left_out = df["truth"].isna().sum()  # count uncertain labels before dropping them
df = df[df["truth"].notna()]  # keep only SN and not-SN
n_sn = (df["truth"] == "SN").sum()
n_not = (df["truth"] == "not SN").sum()
print(f"Usable objects: {len(df)}  ({n_sn} supernovae, {n_not} impostors); "
      f"{left_out} left out for uncertain labels\n")

print("True class (rows) vs Sherlock class (columns):")
print(pd.crosstab(df["truth"], df["classification"], margins=True), "\n")



# Score each cut
is_sn = df["truth"] == "SN"  # True on the rows that really are supernovae
results = []  # one dict per cut, written to the CSV at the end

for name, keep in KEEP_CUTS.items():
    kept = df["classification"].isin(keep)  # True where Sherlock's class is in this cut

    sn_kept = (kept & is_sn).sum()        # real SNe kept (good)
    sn_lost = (~kept & is_sn).sum()       # real SNe thrown away (bad)
    imp_kept = (kept & ~is_sn).sum()      # impostors let through (bad)
    imp_removed = (~kept & ~is_sn).sum()  # impostors removed (good)

    completeness = fraction(sn_kept, sn_kept + sn_lost)
    purity = fraction(sn_kept, sn_kept + imp_kept)
    rejection = fraction(imp_removed, imp_removed + imp_kept)

    print(f"Cut: keep {name}")
    print("  " + show("completeness", completeness))
    print("  " + show("purity      ", purity))
    print("  " + show("rejection   ", rejection))
    print()

    # Store the counts and the three scores, including the low and high edges.
    results.append({
        "cut": name,
        "sn_kept": sn_kept, "sn_lost": sn_lost,
        "impostors_kept": imp_kept, "impostors_removed": imp_removed,
        "completeness": completeness[0],
        "completeness_lo": completeness[1], "completeness_hi": completeness[2],
        "purity": purity[0],
        "purity_lo": purity[1], "purity_hi": purity[2],
        "rejection": rejection[0],
        "rejection_lo": rejection[1], "rejection_hi": rejection[2],
    })

out = TABLES / "sherlock_performance.csv"
pd.DataFrame(results).to_csv(out, index=False)
print(f"Saved {out.name}")
