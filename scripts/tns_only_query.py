import lasair  # talk to the Lasair API
import pandas as pd  # turn the query rows into a table and a CSV
from paths import RAW
from settings import API_TOKEN  # your private API token

# Web address of the Lasair API.
ENDPOINT = "https://api.lasair.lsst.ac.uk/api"

# Log in once. Every query below uses this client.
L = lasair.lasair_client(API_TOKEN, endpoint=ENDPOINT)



def run_query(selected, tables, conditions, page_size=1000, max_pages=10):
    rows = []  # every row from every page ends up here
    for page in range(max_pages):  # page 0, 1, 2, ... up to max_pages
        # Ask for one page. offset skips the rows we already have.
        batch = L.query(selected, tables, conditions,
                        limit=page_size, offset=page * page_size)
        rows.extend(batch)  # add this page onto the full list
        print(f"  page {page + 1}: {len(batch)} rows (running total {len(rows)})")
        if len(batch) < page_size:
            break  # short page means there are no more rows
    else:
        # The loop finished without a short page, so the cap was hit.
        print(f"  stopped after {max_pages} pages; raise max_pages to get more")
    return pd.DataFrame(rows)  # list of dicts -> table



# Query 3: every Lasair object that has a real TNS type.
# This is a separate search. It is not a filter of candidates.csv.

tns_all = run_query(
    selected=(
        # Columns to bring back: id, TNS name, spectroscopic type, redshift.
        "objects.diaObjectId, objects.tns_name, "
        "crossmatch_tns.type, crossmatch_tns.z"
    ),
    # objects is the Lasair list. crossmatch_tns is the TNS catalogue.
    tables="objects, crossmatch_tns",
    conditions=(
        # Match an object to TNS by name.
        "objects.tns_name = crossmatch_tns.tns_name "
        # Keep only rows that have a type written down.
        "AND crossmatch_tns.type IS NOT NULL "
        "AND crossmatch_tns.type != '' "
        # Fixed order so later pages do not skip or repeat rows.
        "ORDER BY objects.diaObjectId"
    ),
)
# Write the table. index=False means no extra numbered column.
tns_all.to_csv(RAW / "lasair" / "tns_all.csv", index=False)
