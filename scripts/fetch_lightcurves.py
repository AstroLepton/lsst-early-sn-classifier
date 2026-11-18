'''
Download the full Lasair record, including the light curve, for every
object in tns_all.csv.

Each object is saved as object_<id>.json in data/raw/lasair/objects.
Files that are already there are skipped, so you can stop this
and run it again until every object is saved:

    python scripts/fetch_lightcurves.py
'''
import json  #write the downloaded record as a JSON file
import time  # so we pause between requests

import lasair
import pandas as pd  # read tns_all.csv
from lasair.lasair import LasairError  # error type Lasair raises
from paths import RAW
from settings import API_TOKEN  # your private API token

# Web address of the Lasair API.
ENDPOINT = "https://api.lasair.lsst.ac.uk/api"
# List of objects to download.
CSV_PATH = RAW / "lasair" / "tns_all.csv"
# Where each object_<id>.json is written.
OUT_DIR = RAW / "lasair" / "objects"

# wait this many seconds after each object, so we do not hit the limit as fast.
PAUSE_SECONDS = 0.5
#how many times to retry one object before skipping it.
ATTEMPTS = 3
# Give up on a single request after this many seconds.
REQUEST_TIMEOUT = 30


# Used to stop the whole run when Lasair says "wait an hour".
class RateLimit(Exception):
    pass


def already_saved(path):
    # True only if the file exists and is not empty.
    return path.exists() and path.stat().st_size > 0


def fetch_one(client, object_id, path):
    last_error = None  # remember the last failure if all retries fail
    for attempt in range(1, ATTEMPTS + 1):  # try 1, then 2, then 3
        try:
            # Download the full object, including the light curve.
            obj = client.object(object_id, lasair_added=True)
            # Refuse to save a reply that is not a real object record.
            if not isinstance(obj, dict) or "diaObjectId" not in obj:
                raise LasairError(f"unexpected reply: {obj}")
            # Write the record to disk.
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, default=str)
            return True  # this object is done
        except RateLimit:
            raise  # do not retry; the hourly limit stops the whole run
        except Exception as error:
            message = str(error)
            # Lasair's hourly cap. Stop instead of hammering the API.
            if "Request limit exceeded" in message or "429" in message:
                raise RateLimit(message)
            last_error = error
            print(f"  {object_id} attempt {attempt} failed: {error}", flush=True)
            time.sleep(2 * attempt)  # wait longer before the next try
    # All retries failed. Leave this object for a later run.
    print(f"  giving up on {object_id} this run: {last_error}", flush=True)
    return False


def object_path(object_id):
    # object_1700....json inside the light-curve folder.
    return OUT_DIR / f"object_{object_id}.json"


def main():
    OUT_DIR.mkdir(exist_ok=True)  # create the folder if it is not there yet
    table = pd.read_csv(CSV_PATH)  # one row per TNS object
    # The id column, as text, so it matches the file names.
    object_ids = table["diaObjectId"].astype(str).tolist()

    pending = []  # objects that still need a download
    for object_id in object_ids:
        path = object_path(object_id)
        if already_saved(path):
            continue  # this one is already in the folder
        pending.append((object_id, path))

    print(
        f"{len(object_ids)} objects in tns_all.csv, "
        f"{len(object_ids) - len(pending)} already saved, "
        f"{len(pending)} to fetch",
        flush=True,
    )
    if not pending:
        return  # nothing left to download

    # Log in to Lasair. timeout stops a hung request.
    client = lasair.lasair_client(
        API_TOKEN, endpoint=ENDPOINT, timeout=REQUEST_TIMEOUT
    )
    saved = 0
    failed = 0
    for index, (object_id, path) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {object_id}", flush=True)
        try:
            ok = fetch_one(client, object_id, path)
        except RateLimit as error:
            # Stop here. The next run will skip files already saved.
            print(
                "Lasair request limit reached. "
                "Rerun this script in about an hour; saved files will be skipped.",
                flush=True,
            )
            print(f"  {error}", flush=True)
            break
        if ok:
            saved += 1
            print(f"  saved {path.name}", flush=True)
        else:
            failed += 1
        time.sleep(PAUSE_SECONDS)  # short pause before the next object

    # How many of the original list still have no file.
    still = sum(
        1
        for object_id in object_ids
        if not already_saved(object_path(object_id))
    )
    print(
        f"Done this run: saved {saved}, failed {failed}, still to fetch: {still}",
        flush=True,
    )


# Run main only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
