'''

  1.Query 1 (cleaned supernova candidates) and saves it to candidates.csv
  2.Query 2 (objects with a TNS spectroscopic class) and saves it to tns_labelled.csv
  3.If OBJECT_ID is set, downloads that object's full record to object_<id>.json

how to run
  pip install lasair pandas
      API_TOKEN = "paste your token here"
'''

import json  # write one object's full record to a JSON file
import lasair  # talk to the Lasair API
import pandas as pd  # tables, CSV files, and the class comparison
from paths import RAW  # folder paths live in one place
from settings import API_TOKEN  # your private API token

# Web address of the Lasair API.
ENDPOINT = "https://api.lasair.lsst.ac.uk/api"

# Log in once. Every query and the single-object download use this client.
L = lasair.lasair_client(API_TOKEN, endpoint=ENDPOINT)


def run_query(selected, tables, conditions, page_size=1000, max_pages=10):
    # Run a Lasair query and page through the results.
    # The API returns at most `page_size` rows per call, so we ask for
    # page 1, then page 2, and so on, until a page comes
    # back short. The conditions must end with a fixed ORDER BY so that
    # pages don't overlap or skip rows.
    rows = []  # every row from every page ends up here
    for page in range(max_pages):
        # offset skips the rows already collected.
        batch = L.query(selected, tables, conditions,
                        limit=page_size, offset=page * page_size)
        rows.extend(batch)
        print(f"  page {page + 1}: {len(batch)} rows (running total {len(rows)})")
        if len(batch) < page_size:
            break  # short page means this was the last one
    else:
        # Hit max_pages while every page was still full, so some rows were not fetched.
        print(f"  stopped after {max_pages} pages; raise max_pages to get more")
    return pd.DataFrame(rows)



# Query 1: cleaned supernova candidates.
# A TNS name is copied if the object has one. A name is not required.

print("Query 1: cleaned candidates")

candidates = run_query(
    selected=(
        # Position, dates, detection counts, latest flux, optional TNS name,
        # and Sherlock's class plus how far it is from the matched galaxy.
        "objects.diaObjectId, objects.ra, objects.decl, "
        "objects.firstDiaSourceMjdTai, objects.lastDiaSourceMjdTai, "
        "objects.nDiaSources, objects.nPosDiaSourcesNights, "
        "objects.latest_psfFlux, objects.tns_name, "
        "sherlock_classifications.classification, "
        "sherlock_classifications.separationArcsec"
    ),
    tables="objects, sherlock_classifications",
    conditions=(
        # Keep Sherlock's supernova-like classes only.
        "sherlock_classifications.classification IN ('SN', 'NT', 'ORPHAN') "
        # At least two nights with a positive detection.
        "AND objects.nPosDiaSourcesNights >= 2 "
        "ORDER BY objects.diaObjectId"
    ),
)

candidates_path = RAW / "lasair" / "candidates.csv"
candidates.to_csv(candidates_path, index=False)
print(f"Saved {len(candidates)} rows to {candidates_path.name}\n")

if not candidates.empty:
    print("Sherlock classes among candidates:")
    # How many rows fell in SN, NT, and ORPHAN.
    print(candidates["classification"].value_counts(), "\n")


# Query 2: objects with a TNS spectroscopic class (for validating Sherlock).
# This is not "the labelled rows of candidates.csv". It is its own search.
print("Query 2: TNS-labelled objects")

tns = run_query(
    selected=(
        "objects.diaObjectId, objects.tns_name, "
        "crossmatch_tns.tns_prefix, crossmatch_tns.type, crossmatch_tns.z, "
        "sherlock_classifications.classification, "
        "sherlock_classifications.separationArcsec, "
        "objects.nPosDiaSourcesNights"
    ),
    # Need the TNS catalogue and Sherlock's class on the same row.
    tables="objects, sherlock_classifications, crossmatch_tns",
    conditions=(
        "objects.tns_name = crossmatch_tns.tns_name "
        # A name is not enough. The TNS row must have a type.
        "AND crossmatch_tns.type IS NOT NULL "
        "AND crossmatch_tns.type != '' "
        "ORDER BY objects.diaObjectId"
    ),
)

tns_path = RAW / "lasair" / "tns_labelled.csv"
tns.to_csv(tns_path, index=False)
print(f"Saved {len(tns)} rows to {tns_path.name}\n")

if not tns.empty:
    # Rows: the true (spectroscopic) class from TNS.
    # Columns: what Sherlock guessed from position alone.
    print("TNS true class (rows) vs Sherlock class (columns):")
    print(pd.crosstab(tns["type"], tns["classification"], margins=True), "\n")


# One full object record, including its light curve.
# Paste a diaObjectId here (from candidates.csv or elsewhere).
# Leave it blank to skip this step.
OBJECT_ID = "313756671828033555"  # e.g. "170019696262250510"

object_id = str(OBJECT_ID).strip()
if object_id:
    print(f"Fetching full record for object {object_id}")

    # lasair_added=True includes the extra Lasair summary and the detections.
    obj = L.object(object_id, lasair_added=True)

    filename = RAW / "lasair" / "objects" / f"object_{object_id}.json"
    with open(filename, "w") as f:
        json.dump(obj, f, indent=2, default=str)  # default=str handles odd value types
    print(f"Saved to {filename.name}")

    # Print the shape of the record so you can see where the light curve sits.
    print("Top-level keys:", list(obj.keys()))
    for key, value in obj.items():
        if isinstance(value, dict):
            print(f"  {key} contains: {list(value.keys())}")
        elif isinstance(value, list):
            print(f"  {key} is a list of {len(value)} items")
else:
    print("No OBJECT_ID set, so no full object record was downloaded.")
