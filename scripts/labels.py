'''
One definition of "supernova" for both datasets.

ELAsTiCC models are the folder names in A_MODEL_SUMMARY.TXT, without the
ELASTICC2_TRAIN_01_ or ELASTICC2_TRAIN_02_ prefix. 1 is a supernova, 0 is not.
A name that is not in the dictionary raises KeyError.

TNS types use the same rule as the old sherlock_check grouping:
a type is a supernova when it starts with "SN" or "SLSN".
Anything else with a real type is not a supernova.
'''

import pandas as pd

# Folder prefixes used by the two ELAsTiCC2 training releases.
# The class name after the prefix is the dictionary key.
_PREFIXES = ("ELASTICC2_TRAIN_02_", "ELASTICC2_TRAIN_01_")

# +HostXT_V19 is the same supernova with extra host-galaxy dust, not a new class.
ELASTICC_IS_SN = {
    # Type Ia, including the faint 91bg and the Iax subclass.
    "SNIa-SALT3": 1,
    "SNIa-91bg": 1,
    "SNIax": 1,
    # Stripped-envelope supernovae.
    "SNIb-Templates": 1,
    "SNIb+HostXT_V19": 1,
    "SNIc-Templates": 1,
    "SNIc+HostXT_V19": 1,
    "SNIcBL+HostXT_V19": 1,
    "SNIIb+HostXT_V19": 1,
    # Hydrogen-rich supernovae, including the narrow-line IIn models.
    "SNII-NMF": 1,
    "SNII-Templates": 1,
    "SNII+HostXT_V19": 1,
    "SNIIn-MOSFIT": 1,
    "SNIIn+HostXT_V19": 1,
    # Superluminous and pair-instability supernovae.
    "SLSN-I+host": 1,
    "SLSN-I_no_host": 1,
    "PISN": 1,
    # Not a supernova. CART is decided in the logbook entry 2026-11-19_cart-label.
    "CART": 0,
    "KN_K17": 0,
    "KN_B19": 0,
    "ILOT": 0,
    "TDE": 0,
    "CLAGN": 0,
    "Mdwarf-flare": 0,
    "dwarf-nova": 0,
    "uLens-Binary": 0,
    "uLens-Single_PyLIMA": 0,
    "uLens-Single-GenLens": 0,
    "EB": 0,
    "Cepheid": 0,
    "RRL": 0,
    "d-Sct": 0,
}

# TNS types that don't say what the transient is, so they can't be used as truth.
UNUSABLE_TYPES = {"Other", "Galaxy"}


def elasticc_label(folder_name):
    """1 if this ELAsTiCC model is a supernova, otherwise 0.

    folder_name may be the class (SNIa-SALT3) or the full folder
    (ELASTICC2_TRAIN_02_SNIa-SALT3). An unknown class raises KeyError.
    """
    name = str(folder_name).strip()
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name not in ELASTICC_IS_SN:
        raise KeyError(
            f"No label for ELAsTiCC model {folder_name!r}. "
            "Add it to ELASTICC_IS_SN in labels.py before using it."
        )
    return ELASTICC_IS_SN[name]


def broad_class(tns_type):
    """Turn a detailed TNS type into 'SN', 'not SN', or None (leave out)."""
    if pd.isna(tns_type):
        return None  # blank cell
    t = str(tns_type).strip()
    if t == "" or "?" in t or t in UNUSABLE_TYPES:
        return None  # missing or uncertain label
    if t.startswith("SN") or t.startswith("SLSN"):
        return "SN"  # SN Ia, SN II, SN Ic-BL, SLSN-I, ...
    return "not SN"  # TDE, AGN, CV, Nova, Varstar, Impostor-SN, Gap, ...
