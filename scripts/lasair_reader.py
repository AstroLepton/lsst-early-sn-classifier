'''
Turn saved Lasair object records into the same tables as elasticc_reader.

Reads data/raw/lasair/objects/object_<id>.json. Does not call the Lasair API.

Writes three Parquet files:

lightcurves_lasair.parquet   one row per epoch
    objectId, mjd, band, flux, fluxErr, isDetection
objects_lasair.parquet       one row per object
    objectId, ra, dec, host_sep, mwebv, trueClass
redshifts_lasair.parquet     one row per object, kept for a later redshift test
    objectId, tns_z, sherlock_z, sherlock_photoz

psfFlux is already in nanojansky, so it is not multiplied by 36.308.
diaSourcesList is a detection. diaForcedSourcesList is not.
The saved records have no visit id, so a repeated epoch is matched on
mjd + band, and the detection is the row that is kept.

how to run (from the repo root)
    python scripts/lasair_reader.py
'''

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from paths import INTERIM, RAW

OBJECTS_DIR = RAW / "lasair" / "objects"

# Same light-curve and object columns as elasticc_reader.
LIGHTCURVE_SCHEMA = pa.schema([
    ("objectId", pa.int64()),
    ("mjd", pa.float64()),
    ("band", pa.string()),
    ("flux", pa.float64()),
    ("fluxErr", pa.float64()),
    ("isDetection", pa.bool_()),
])
OBJECT_SCHEMA = pa.schema([
    ("objectId", pa.int64()),
    ("ra", pa.float64()),
    ("dec", pa.float64()),
    ("host_sep", pa.float64()),
    ("mwebv", pa.float64()),
    ("trueClass", pa.string()),
])
# TNS.z, Sherlock z, and Sherlock photoZ. Nothing else.
REDSHIFT_SCHEMA = pa.schema([
    ("objectId", pa.int64()),
    ("tns_z", pa.float64()),
    ("sherlock_z", pa.float64()),
    ("sherlock_photoz", pa.float64()),
])

LIGHTCURVE_COLUMNS = [field.name for field in LIGHTCURVE_SCHEMA]
OBJECT_COLUMNS = [field.name for field in OBJECT_SCHEMA]


def as_float(value):
    """A JSON number to float. null, blank, and non-numeric become NaN."""
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def epochs(rows, is_detection):
    """One photometry list -> mjd, band, flux, fluxErr, isDetection."""
    empty = pd.DataFrame(columns=["mjd", "band", "flux", "fluxErr", "isDetection"])
    if not rows:
        return empty
    df = pd.DataFrame(rows).rename(columns={
        "midpointMjdTai": "mjd",
        "psfFlux": "flux",
        "psfFluxErr": "fluxErr",
    })
    for column in ("mjd", "band", "flux", "fluxErr"):
        if column not in df.columns:
            df[column] = np.nan
    df = df[["mjd", "band", "flux", "fluxErr"]].copy()
    df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df["fluxErr"] = pd.to_numeric(df["fluxErr"], errors="coerce")
    df["band"] = df["band"].astype("string").str.lower()
    df = df.dropna(subset=["mjd", "band"])
    # One row per epoch inside this list. A second detection at the same
    # mjd and band would be the same epoch.
    df = df.drop_duplicates(subset=["mjd", "band"], keep="first")
    df["isDetection"] = is_detection
    return df


def light_curve(obj):
    """Detections plus forced points that are not the same epoch.

    Returns the table and how many forced rows were dropped as duplicates.
    """
    detections = epochs(obj.get("diaSourcesList") or [], True)
    forced = epochs(obj.get("diaForcedSourcesList") or [], False)
    if detections.empty or forced.empty:
        out = pd.concat([detections, forced], ignore_index=True)
        return out, 0
    # No visit id in either list. Exact mjd + band is the same epoch.
    keys = detections[["mjd", "band"]].drop_duplicates()
    marked = forced.merge(keys, on=["mjd", "band"], how="left", indicator=True)
    dropped = int((marked["_merge"] == "both").sum())
    forced = marked[marked["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([detections, forced], ignore_index=True)
    return out, dropped


def object_row(obj):
    """One object -> object-table row and redshift-table row."""
    object_id = int(obj["diaObjectId"])
    dia = obj.get("diaObject") or {}
    lasair = obj.get("lasairData") or {}
    # A missing Sherlock block and a null separation are both "no host offset".
    sherlock = lasair.get("sherlock")
    if not isinstance(sherlock, dict) or not sherlock:
        host_sep = np.nan
        sherlock_z = np.nan
        sherlock_photoz = np.nan
    else:
        host_sep = as_float(sherlock.get("separationArcsec"))
        sherlock_z = as_float(sherlock.get("z"))
        sherlock_photoz = as_float(sherlock.get("photoZ"))
    tns = lasair.get("TNS") if isinstance(lasair.get("TNS"), dict) else {}
    true_class = tns.get("type")
    if isinstance(true_class, str):
        true_class = true_class.strip() or None
    else:
        true_class = None
    info = {
        "objectId": object_id,
        "ra": as_float(dia.get("ra")),
        "dec": as_float(dia.get("decl")),
        "host_sep": host_sep,
        "mwebv": as_float(lasair.get("ebv")),
        "trueClass": true_class,
    }
    redshift = {
        "objectId": object_id,
        "tns_z": as_float(tns.get("z")) if tns else np.nan,
        "sherlock_z": sherlock_z,
        "sherlock_photoz": sherlock_photoz,
    }
    return info, redshift


def main():
    paths = sorted(OBJECTS_DIR.glob("object_*.json"))
    if not paths:
        raise SystemExit(f"No object_*.json files in {OBJECTS_DIR}")

    light_parts = []
    objects = []
    redshifts = []
    n_dropped = 0
    for path in paths:
        with open(path) as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or "diaObjectId" not in obj:
            raise ValueError(f"{path.name} is not a Lasair object record")
        curve, dropped = light_curve(obj)
        curve.insert(0, "objectId", int(obj["diaObjectId"]))
        light_parts.append(curve[LIGHTCURVE_COLUMNS])
        info, redshift = object_row(obj)
        objects.append(info)
        redshifts.append(redshift)
        n_dropped += dropped

    lightcurves = pd.concat(light_parts, ignore_index=True)
    # The column list is fixed, so a redshift column cannot be appended by accident.
    objects = pd.DataFrame(objects, columns=OBJECT_COLUMNS)
    redshifts = pd.DataFrame(redshifts, columns=[field.name for field in REDSHIFT_SCHEMA])

    INTERIM.mkdir(parents=True, exist_ok=True)
    lc_path = INTERIM / "lightcurves_lasair.parquet"
    obj_path = INTERIM / "objects_lasair.parquet"
    z_path = INTERIM / "redshifts_lasair.parquet"
    pq.write_table(pa.Table.from_pandas(lightcurves, schema=LIGHTCURVE_SCHEMA, preserve_index=False), lc_path)
    pq.write_table(pa.Table.from_pandas(objects, schema=OBJECT_SCHEMA, preserve_index=False), obj_path)
    pq.write_table(pa.Table.from_pandas(redshifts, schema=REDSHIFT_SCHEMA, preserve_index=False), z_path)

    n_det = int(lightcurves["isDetection"].sum())
    print(f"{len(objects)} objects")
    print(f"{len(lightcurves)} epochs ({n_det} detections, {len(lightcurves) - n_det} forced only)")
    print(f"{n_dropped} forced rows dropped because they repeat a detection")
    print(f"host_sep missing: {int(objects['host_sep'].isna().sum())}")
    print(f"trueClass missing: {int(objects['trueClass'].isna().sum())}")
    print(f"Saved {lc_path.name}, {obj_path.name}, {z_path.name}")


if __name__ == "__main__":
    main()
