'''
Build the ELAsTiCC feature table.

The split is chosen before any feature is computed. Object ids are capped
at 3,000 per true class, then divided 70/15/15 into train, validation and
test. The same object keeps that split at every night N.

One row per object per night is written to data/processed/features_elasticc.parquet.
A histogram of each feature at N = 2 is written beside it.

    python scripts/build_features.py
    python scripts/build_features.py --max-per-class 3000 --seed 0
'''

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from features import FEATURE_NAMES, abs_galactic_latitude, compute_features
from labels import elasticc_label
from paths import FIGS, INTERIM, PROCESSED
from truncate import NIGHTS, truncate

SEED = 0
MAX_PER_CLASS = 3000
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def sample_and_split(objects, max_per_class, seed):
    """Cap each class, then assign train / validation / test.

    Classes are visited in sorted order with one Generator, so the same
    seed always selects the same objects and the same split.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for true_class, group in objects.groupby("trueClass", sort=True):
        group = group.drop_duplicates("objectId")
        ids = group["objectId"].to_numpy().copy()
        rng.shuffle(ids)
        ids = ids[:max_per_class]
        n = len(ids)
        n_train = min(n, int(round(n * TRAIN_FRACTION)))
        n_val = min(n - n_train, int(round(n * VALIDATION_FRACTION)))
        n_test = n - n_train - n_val
        split = np.array(
            ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
        )
        # loc keeps the shuffled order, so split[i] belongs to ids[i].
        chosen = group.set_index("objectId").loc[ids].reset_index()
        if not np.array_equal(chosen["objectId"].to_numpy(), ids):
            raise RuntimeError(f"split order slipped for {true_class}")
        chosen["split"] = split
        chosen["is_sn"] = elasticc_label(true_class)
        parts.append(chosen)
        print(f"  {true_class}: {n} objects "
              f"({n_train} train, {n_val} validation, {n_test} test)", flush=True)
    return pd.concat(parts, ignore_index=True)


def read_light_curves(object_ids):
    """Photometry rows for these objects only. The file is scanned once."""
    wanted = pa.array(np.asarray(object_ids, dtype=np.int64))
    columns = ["objectId", "mjd", "band", "flux", "fluxErr", "isDetection"]
    source = pq.ParquetFile(INTERIM / "lightcurves_elasticc.parquet")
    kept = []
    for batch in source.iter_batches(columns=columns, batch_size=500_000):
        mask = pc.is_in(batch.column("objectId"), value_set=wanted)
        filtered = batch.filter(mask)
        if filtered.num_rows:
            kept.append(filtered)
    if not kept:
        return pd.DataFrame(columns=["objectId", "mjd", "band", "flux", "fluxErr", "isDetection"])
    table = pa.Table.from_batches(kept)
    print(f"  {table.num_rows} photometry rows for {len(object_ids)} objects", flush=True)
    return table.to_pandas()


def build(sample):
    """Features at every night. The split column is copied, not recomputed."""
    latitude = abs_galactic_latitude(sample["ra"], sample["dec"])
    sample = sample.copy()
    sample["abs_gal_b"] = latitude
    meta = sample.set_index("objectId")
    light_curves = read_light_curves(sample["objectId"])
    grouped = {
        oid: frame.drop(columns="objectId")
        for oid, frame in light_curves.groupby("objectId", sort=False)
    }
    missing = [oid for oid in meta.index if oid not in grouped]
    if missing:
        print(f"  {len(missing)} objects have no photometry rows", flush=True)

    rows = []
    for i, oid in enumerate(meta.index, start=1):
        info = meta.loc[oid]
        lc = grouped.get(oid)
        if lc is None:
            lc = pd.DataFrame(columns=["mjd", "band", "flux", "fluxErr", "isDetection"])
        for n in NIGHTS:
            features = compute_features(
                truncate(lc, n), info["host_sep"], info["mwebv"], info["abs_gal_b"],
            )
            features["objectId"] = int(oid)
            features["N"] = int(n)
            features["split"] = info["split"]
            features["trueClass"] = info["trueClass"]
            features["is_sn"] = int(info["is_sn"])
            rows.append(features)
        if i % 1000 == 0:
            print(f"  {i} / {len(meta)} objects", flush=True)
    columns = ["objectId", "N", "split", "trueClass", "is_sn", *FEATURE_NAMES]
    return pd.DataFrame(rows, columns=columns)


def plot_night2(features):
    """Histograms at N = 2. SN and not-SN on the same axes."""
    night = features.loc[features["N"] == 2]
    sn = night.loc[night["is_sn"] == 1]
    other = night.loc[night["is_sn"] == 0]
    n_cols = 4
    n_rows = int(np.ceil(len(FEATURE_NAMES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.6 * n_rows))
    for ax, name in zip(axes.ravel(), FEATURE_NAMES):
        a = sn[name].to_numpy(dtype=float)
        b = other[name].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) == 0 and len(b) == 0:
            ax.set_title(f"{name}\nno values")
            ax.axis("off")
            continue
        both = np.concatenate([a, b]) if len(a) and len(b) else (a if len(a) else b)
        # pre_chi2 has a thin tail past a few tens. Cutting at the 95th
        # percentile leaves the shift between the two samples visible.
        upper = 95 if name == "pre_chi2" else 99
        lo, hi = np.percentile(both, [1, upper])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(both)), float(np.max(both))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 40)
        if len(a):
            ax.hist(a, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.55, color="#d62728", label="SN")
            ax.axvline(np.median(a), color="#d62728", lw=0.8)
        if len(b):
            ax.hist(b, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.55, color="#1f77b4", label="not SN")
            ax.axvline(np.median(b), color="#1f77b4", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(FEATURE_NAMES):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("ELAsTiCC features at N = 2", y=1.01)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / "2026-11-19_features_n2.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")
    return path


def report_separation(features):
    """Medians at N = 2, so the SN / not-SN difference is on the screen."""
    night = features.loc[features["N"] == 2]
    print("\nN = 2 median")
    print(f"{'feature':<16} {'SN':>10} {'not SN':>10}")
    for name in ("pre_chi2", "neg_frac"):
        sn = night.loc[night["is_sn"] == 1, name].median()
        other = night.loc[night["is_sn"] == 0, name].median()
        print(f"{name:<16} {sn:10.3f} {other:10.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS,
                        help="objects kept from each ELAsTiCC class (default 3000)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="random seed for the sample and the split (default 0)")
    args = parser.parse_args()

    objects = pd.read_parquet(
        INTERIM / "objects_elasticc.parquet",
        columns=["objectId", "ra", "dec", "host_sep", "mwebv", "trueClass"],
    )
    print("Choosing the split before any features are computed", flush=True)
    sample = sample_and_split(objects, args.max_per_class, args.seed)
    # The split is fixed here. Nothing below is allowed to change it.
    split_by_id = sample.set_index("objectId")["split"]

    print("Computing features", flush=True)
    features = build(sample)
    mapped = features["objectId"].map(split_by_id)
    if not np.array_equal(mapped.to_numpy(), features["split"].to_numpy()):
        raise RuntimeError("a row's split does not match the object split")
    changed = features.groupby("objectId")["split"].nunique()
    if (changed != 1).any():
        raise RuntimeError("an object landed in more than one split")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "features_elasticc.parquet"
    features.to_parquet(out, index=False)
    print(f"Saved {len(features)} rows to {out.name}")
    report_separation(features)
    plot_night2(features)


if __name__ == "__main__":
    main()
