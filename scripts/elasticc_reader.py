'''
Turn ELAsTiCC2 training files (HEAD + PHOT + DUMP) into the shared tables.

Writes two Parquet files into OUT_DIR:

lightcurves.parquet  one row per photometry point
objectId, mjd, band, flux, fluxErr, isDetection, trueClass
objects.parquet      one row per object
objectId, ebv, trueClass

flux and fluxErr are in nanojansky, the same unit as Lasair psfFlux.
Redshifts (REDSHIFT_*, HOSTGAL_*PHOTOZ, DUMP ZCMB/GALZPHOT) are not carried
over, because real LSST alerts on Lasair have no host redshift yet.

how to run (from the repo root)
    python scripts/elasticc_reader.py                     # every ELASTICC2_TRAIN_02_* folder in data/raw/elasticc2
    python scripts/elasticc_reader.py --max-pairs 1       # quick test: first HEAD/PHOT pair per model
    python scripts/elasticc_reader.py ELASTICC2_TRAIN_02_RRL --out rrl_tables
'''

import argparse  # read the options typed after the script name
import io  # treat a string as a file, for pandas
from pathlib import Path  

import numpy as np 
import pandas as pd  
import pyarrow as pa  # Parquet column types
import pyarrow.parquet as pq  # write Parquet files piece by piece
from astropy.io import fits  # open the .FITS.gz files
from paths import INTERIM, RAW

# ELAsTiCC2 model folders, exactly as downloaded.
DATA = RAW / "elasticc2"
# Every model folder starts with this. Removing it leaves the class name.
PREFIX = "ELASTICC2_TRAIN_02_"

# FLUXCAL has zero point 27.5; Lasair nJy has 31.4.
NJY_PER_FLUXCAL = 10 ** ((31.4 - 27.5) / 2.5)  # about 36.31
DETECTION_BIT = 4096  # PHOTFLAG bit set when the point is a detection
#1024 saturated, 2048 trigger, meaning the object's first detectio, 4096 detected, signal to noise above 5sigma
END_OF_LIGHTCURVE_MJD = -777.0  # separator row between objects in PHOT

# Column names and types of the light-curve table (one row per point).
LIGHTCURVE_SCHEMA = pa.schema([
    ("objectId", pa.int64()),
    ("mjd", pa.float64()),
    ("band", pa.string()),
    ("flux", pa.float64()),
    ("fluxErr", pa.float64()),
    ("isDetection", pa.bool_()),
    ("trueClass", pa.string()),
])
# Column names and types of the object table (one row per object).
OBJECT_SCHEMA = pa.schema([
    ("objectId", pa.int64()),
    ("ebv", pa.float64()),
    ("trueClass", pa.string()),
])


def read_dump(path):
    """CID -> GENTYPE from the DUMP truth table (only the SN: data rows)."""
    names = None  # column names, from the VARNAMES line
    rows = []  # one text line per object
    with open(path) as f:
        for line in f:
            if line.startswith("VARNAMES:"):
                names = line.split()[1:]  # drop the word VARNAMES itself
            elif line.startswith("SN:"):
                rows.append(line[3:])  # drop the "SN:" at the front
    if names is None:
        raise ValueError(f"{path} has no VARNAMES line")
    # Columns are split by spaces. Only keep the id and the true type number.
    dump = pd.read_csv(
        io.StringIO("".join(rows)), sep=r"\s+", header=None,
        names=names, usecols=["CID", "GENTYPE"],
    )
    return dump.set_index("CID")["GENTYPE"]  # look up a type by object id


def native(column):
    """convert to the machine's byte order."""
    array = np.asarray(column)
    return array.astype(array.dtype.newbyteorder("="))  # "=" means this computer's order


def decode(column):
    """Fixed-width FITS strings -> stripped Python strings."""
    array = np.asarray(column)
    if array.dtype.kind == "S":  # raw bytes, so turn them into text first
        array = np.char.decode(array, "ascii")
    return np.char.strip(array.astype(str))  # "39592588        " -> "39592588"


def read_pair(head_path, phot_path, true_class, gentype_by_cid):
    """One HEAD/PHOT pair -> (light-curve DataFrame, object DataFrame)."""
    # HEAD: one row per object. Only take the columns I need (no redshifts).
    with fits.open(head_path) as hdul:
        head = hdul[1].data  # the table is in the 2nd part of the FITS file
        object_id = decode(head["SNID"]).astype(np.int64)
        ptr_min = native(head["PTROBS_MIN"]).astype(np.int64)  # first PHOT row of this object
        ptr_max = native(head["PTROBS_MAX"]).astype(np.int64)  # last PHOT row of this object
        nobs = native(head["NOBS"]).astype(np.int64)  # number of PHOT rows
        ebv = native(head["MWEBV"]).astype(np.float64)  # Milky Way dust
        sim_type = native(head["SIM_TYPE_INDEX"]).astype(np.int64)  # true type number

    # The pointers and NOBS should describe the same rows.
    if not np.array_equal(ptr_max - ptr_min + 1, nobs):
        raise ValueError(f"{head_path.name}: NOBS does not match PTROBS_MAX - PTROBS_MIN + 1")

    # Check HEAD against the DUMP truth table, object by object.
    gentype = gentype_by_cid.reindex(object_id).to_numpy()  # NaN if an id is not in DUMP
    if np.isnan(gentype.astype(float)).any():
        raise ValueError(f"{head_path.name}: some SNIDs are missing from the DUMP file")
    if not np.array_equal(gentype.astype(np.int64), sim_type):
        raise ValueError(f"{head_path.name}: DUMP GENTYPE disagrees with HEAD SIM_TYPE_INDEX")

    # PHOT: every measurement of every object in this pair, one after another.
    with fits.open(phot_path) as hdul:
        phot = hdul[1].data
        mjd_all = native(phot["MJD"]).astype(np.float64)
        band_all = decode(phot["BAND"])
        flux_all = native(phot["FLUXCAL"]).astype(np.float64)
        flux_err_all = native(phot["FLUXCALERR"]).astype(np.float64)
        photflag_all = native(phot["PHOTFLAG"]).astype(np.int64)

    # PTROBS are 1-based and inclusive. Build the PHOT row numbers for every
    # object at once, plus which object each row belongs to.
    # e.g. NOBS = [3, 2], starts = [0, 4] -> owner = [0,0,0,1,1], offset = [0,1,2,0,1],
    # rows = [0,1,2,4,5]
    starts = ptr_min - 1  # 1-based -> 0-based
    owner = np.repeat(np.arange(len(object_id)), nobs)  # which HEAD row each point is for
    offset = np.arange(nobs.sum()) - np.repeat(np.cumsum(nobs) - nobs, nobs)  # 0,1,2... per object
    rows = starts[owner] + offset  # PHOT row number of each point
    if rows.size and rows.max() >= len(mjd_all):
        raise ValueError(f"{head_path.name}: pointers run past the end of {phot_path.name}")

    # Drop the -777 separator rows. They are not real measurements.
    mjd = mjd_all[rows]
    keep = mjd != END_OF_LIGHTCURVE_MJD
    rows, owner = rows[keep], owner[keep]

    # The light-curve table, in the shared format.
    lightcurves = pd.DataFrame({
        "objectId": object_id[owner],
        "mjd": mjd_all[rows],
        # Lasair writes the LSST y filter in lower case.
        "band": np.char.lower(band_all[rows]),
        "flux": flux_all[rows] * NJY_PER_FLUXCAL,  # FLUXCAL -> nJy
        "fluxErr": flux_err_all[rows] * NJY_PER_FLUXCAL,
        "isDetection": (photflag_all[rows] & DETECTION_BIT) != 0,  # is the 4096 bit set?
        "trueClass": true_class,  # same for every row, it comes from the folder name
    })
    # The object table: one row per HEAD row.
    objects = pd.DataFrame({
        "objectId": object_id,
        "ebv": ebv,
        "trueClass": true_class,
    })
    return lightcurves, objects


def model_pairs(model_dir):
    """Sorted (HEAD, PHOT) paths in one model folder."""
    pairs = []
    for head_path in sorted(model_dir.glob("*_HEAD.FITS.gz")):  # 0001, 0002, ... 0040
        # The matching PHOT file has the same name with _PHOT instead of _HEAD.
        phot_path = head_path.with_name(head_path.name.replace("_HEAD.", "_PHOT."))
        if not phot_path.exists():
            raise FileNotFoundError(f"{phot_path} is missing")
        pairs.append((head_path, phot_path))
    return pairs


# The one .DUMP truth table in a model folder.
def find_dump(model_dir):
    dumps = sorted(model_dir.glob("*.DUMP"))
    if len(dumps) != 1:
        raise FileNotFoundError(f"expected one .DUMP in {model_dir}, found {len(dumps)}")
    return dumps[0]


# Read every pair of every model and write the two Parquet files.
def run(model_dirs, out_dir, max_pairs=None):
    out_dir.mkdir(parents=True, exist_ok=True)  # make the output folder if needed
    lc_path = out_dir / "lightcurves.parquet"
    obj_path = out_dir / "objects.parquet"

    # Writers add one pair at a time, so all 70 million rows never sit in memory at once.
    lc_writer = pq.ParquetWriter(lc_path, LIGHTCURVE_SCHEMA)
    obj_writer = pq.ParquetWriter(obj_path, OBJECT_SCHEMA)
    n_points = n_objects = 0  # running totals for the summary at the end
    try:
        for model_dir in model_dirs:
            # ELASTICC2_TRAIN_02_SNIa-SALT3 -> SNIa-SALT3
            true_class = model_dir.name.removeprefix(PREFIX)
            gentype_by_cid = read_dump(find_dump(model_dir))
            pairs = model_pairs(model_dir)[:max_pairs]  # [:None] keeps all 40
            print(f"{true_class}: {len(pairs)} HEAD/PHOT pairs, "
                  f"{len(gentype_by_cid)} objects in DUMP", flush=True)

            for head_path, phot_path in pairs:
                lightcurves, objects = read_pair(head_path, phot_path, true_class, gentype_by_cid)
                # Append this pair to the end of each file.
                lc_writer.write_table(pa.Table.from_pandas(
                    lightcurves, schema=LIGHTCURVE_SCHEMA, preserve_index=False))
                obj_writer.write_table(pa.Table.from_pandas(
                    objects, schema=OBJECT_SCHEMA, preserve_index=False))
                n_points += len(lightcurves)
                n_objects += len(objects)
                print(f"  {head_path.name}: {len(objects)} objects, "
                      f"{len(lightcurves)} points", flush=True)
    finally:
        # Always close, even after an error, or the Parquet files are left broken.
        lc_writer.close()
        obj_writer.close()

    print(f"\nSaved {n_points} points to {lc_path}")
    print(f"Saved {n_objects} objects to {obj_path}")


def main():
    # The options you can type after "python3 elasticc_reader.py".
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*",
                        help="model folders (default: every ELASTICC2_TRAIN_02_* folder in data/raw/elasticc2)")
    parser.add_argument("--out", default=str(INTERIM),
                        help="output folder (default: data/interim)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="only read the first N HEAD/PHOT pairs of each model")
    args = parser.parse_args()

    if args.models:
        model_dirs = []
        for m in args.models:
            p = Path(m)
            # A bare folder name is looked up next to the other raw ELAsTiCC2 files.
            if not p.is_absolute() and not p.exists():
                p = DATA / p
            model_dirs.append(p.resolve())
    else:
        # No folders named, so use every model folder in data/raw/elasticc2.
        model_dirs = sorted(p for p in DATA.glob(PREFIX + "*") if p.is_dir())
    if not model_dirs:
        raise SystemExit(f"No {PREFIX}* folders found in {DATA}")

    run(model_dirs, Path(args.out).resolve(), args.max_pairs)


# Run main only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
