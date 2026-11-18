Photometric supernova classifier for early LSST alerts. Real objects come from the Lasair broker, with TNS spectroscopic types where they exist, and the training set is simulated ELAsTiCC2 light curves. Both are turned into one shared table of flux versus time so a classifier can be trained on the simulations and tested on the real alerts.

Create the environment from the repository root, then activate it:

```
conda env create -f environment.yml
conda activate lsst-sn
```

The Lasair API token lives in `scripts/settings.py` as `API_TOKEN`. That file is gitignored. Do not commit it.

```
lsst-early-sn-classifier/
├── scripts/            # all code, flat; paths come from paths.py
├── data/raw/           # downloads, never edited (gitignored)
├── data/interim/       # light curves in the shared table format (gitignored)
├── data/processed/     # day-N tables for models (gitignored)
├── results/figures/    # plots
├── results/tables/     # result tables
├── models/             # trained models (gitignored)
├── notebooks/          # exploration only
├── literature/         # paper PDFs (gitignored)
└── logbook/            # lab notes
```

Run every script from the repository root.

- `python scripts/lasair_phase1.py` — queries Lasair for cleaned supernova candidates and for TNS-labelled objects, and can download one full object record. This calls the Lasair API.
- `python scripts/tns_only_query.py` — queries every Lasair object that has a TNS type and writes `tns_all.csv`. This calls the Lasair API.
- `python scripts/fetch_lightcurves.py` — downloads the full record, including the light curve, for each object in `tns_all.csv`. This calls the Lasair API.
- `python scripts/sherlock_check.py` — scores Sherlock's completeness, purity, and rejection against the TNS labels, and writes `sherlock_performance.csv`.
- `python scripts/light_curves.py` — plots one Lasair light curve from the saved object JSON. Set `OBJECT_ID` at the top of the script. It calls the API only when that JSON is not already on disk.
- `python scripts/download_docs.py` — downloads the selected ELAsTiCC2 training folders from NERSC into `data/raw/elasticc2`.
- `python scripts/elasticc_reader.py` — reads the ELAsTiCC2 HEAD, PHOT, and DUMP files and writes `lightcurves.parquet` and `objects.parquet` into `data/interim`. The full run is large; `--max-pairs 1` reads only the first file pair of each model.
- `python scripts/plot_elasticc.py` — plots a few ELAsTiCC2 light curves from the interim parquet tables.
- `scripts/paths.py` — defines every folder path. Scripts import it; it is not run on its own.
