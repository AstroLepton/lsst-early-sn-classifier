Photometric supernova classifier for early LSST alerts. Real objects come from the Lasair broker, with TNS spectroscopic types where they exist, and the training set is simulated ELAsTiCC2 light curves. Both are turned into one shared table of flux versus time so a classifier can be trained on the simulations and tested on the real alerts.

The training data are simulated light curves (ELAsTiCC). The test data are real LSST alerts from Lasair. A TNS spectrum gives the true type when one exists.

From this folder:

```
conda env create -f environment.yml
conda activate lsst-sn
```


## Scripts

- `download_docs.py` downloads the simulated light curves. Data: ELAsTiCC files from NERSC.
- `elasticc_reader.py` turns those files into one light-curve table and one object table. Data: ELAsTiCC FITS files.
- `lasair_phase1.py` gets a list of supernova candidates, and a list of objects that already have a spectrum type. Data: Lasair, over the web.
- `tns_only_query.py` gets every Lasair object that has a TNS type. Data: Lasair, over the web.
- `fetch_lightcurves.py` downloads the light curve for each object in that list. Data: Lasair, over the web.
- `lasair_reader.py` turns the downloaded Lasair objects into the same kind of tables as the simulations. Data: saved Lasair JSON files.
- `plot_elasticc.py` plots a few simulated light curves. Data: ELAsTiCC tables.
- `plot_lasair.py` plots a few real light curves. Data: Lasair tables.
- `light_curves.py` plots one real object, and the same object cut at night 0, 2 and 5. Data: one Lasair JSON file and the Lasair table.
- `sherlock_check.py` checks how often Sherlock is right, using the spectrum types as the truth. Data: the TNS-labelled list.
- `labels.py` says whether a class counts as a supernova. Other scripts import this. Data: ELAsTiCC model names and TNS types.
- `truncate.py` cuts a light curve at the end of night N after the first detection. Data: one light-curve table.

- `paths.py` holds the folder paths. Other scripts import this. Data: none.
