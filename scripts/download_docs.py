'''Download selected ELASTICC2 training models '''

import requests 
from html.parser import HTMLParser  # read the links out of a folder listing page
from pathlib import Path
from urllib.parse import quote, unquote  # turn names into URL-safe text and back

from paths import RAW

# Web address of the ELAsTiCC2 training sample. Every model folder sits under this.
BASE = (
    "https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/"
    "ELASTICC2_TRAINING_SAMPLE_2/"
)

# Class label and the directory that holds that model's FITS files.
# The counts next to each class (172180, 78306, 38722, 20048) are the
# number of light curves written into these files, not the file count.
MODELS = (
    ("SN Ia", "ELASTICC2_TRAIN_02_SNIa-SALT3"),
    ("SN II", "ELASTICC2_TRAIN_02_SNII-Templates"),
    ("AGN", "ELASTICC2_TRAIN_02_CLAGN"),
    ("Variable star", "ELASTICC2_TRAIN_02_RRL"),
)

# Save into data/raw/elasticc2, next to the files already downloaded.
OUT = RAW / "elasticc2"


# Collects the file names from a folder listing page.
class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []  # every file name found on the page

    # Called for every HTML tag on the page.
    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return  # only <a> tags are links
        href = dict(attrs).get("href", "")  # where the link points
        # Skip sort links , the parent folder, and sub-folders.
        if not href or href.startswith(("?", "/", "../")) or href.endswith("/"):
            return
        self.names.append(unquote(href))  # %5F -> _ so it is the real file name


# Every file name inside one model folder on the portal.
def list_files(dirname: str) -> list[str]:
    response = requests.get(BASE + dirname + "/", timeout=60)  # the folder listing page
    response.raise_for_status()  # stop if the page did not load
    parser = LinkParser()
    parser.feed(response.text)  # pull the links out of the HTML
    return parser.names


# Download one file, unless a complete copy is already on disk.
def download_file(dirname: str, name: str, dest: Path) -> None:
    url = BASE + dirname + "/" + quote(name)  # quote turns odd characters into %xx
    # stream=True so a big file comes in pieces, not all into memory at once.
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", 0))  # file size on the server
        # Same size on disk already means it finished last time, so skip it.
        if dest.exists() and expected and dest.stat().st_size == expected:
            print(f"  skip {name}")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)  # make the model folder if needed
        # Write to name.part first, so a half-finished file never looks complete.
        partial = dest.with_name(dest.name + ".part")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MB at a time
                if chunk:
                    handle.write(chunk)
        partial.replace(dest)  # finished, so rename .part to the real name
    size_mb = dest.stat().st_size / 1e6
    print(f"  saved {name} ({size_mb:.1f} MB)")


# Download every file in one model folder.
def download_model(label: str, dirname: str) -> None:
    names = list_files(dirname)
    print(f"\n{label}: {dirname} ({len(names)} files)")
    folder = OUT / dirname  # e.g. data/raw/elasticc2/ELASTICC2_TRAIN_02_RRL
    for index, name in enumerate(names, start=1):
        print(f"[{index}/{len(names)}]", end="")  # progress, e.g. [12/84]
        download_file(dirname, name, folder / name)


# Run the downloads only when this file is executed, not when it is imported.
if __name__ == "__main__":
    for label, dirname in MODELS:
        download_model(label, dirname)
    print("\ndone")
