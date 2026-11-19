# 19 Nov 2026 — CART label

CART is labelled 0. It is not a supernova in this project.

The rule for both datasets is the TNS name, the same test `broad_class` already used: a type counts as a supernova only when it starts with `SN` or `SLSN`. ELAsTiCC models were filled from `A_MODEL_SUMMARY.TXT` (32 models). Ia, Iax, 91bg, Ib, Ic, Ic-BL, II, IIn, IIb, SLSN-I and PISN are 1. AGN, TDE, kilonova, ILOT, M-dwarf flare, dwarf nova, microlensing, eclipsing binary, Cepheid, RR Lyrae and δ Scuti are 0. There is no Mira model in that file, so Mira has no key. A folder that is not in the dictionary raises `KeyError`.

CART is the calcium-rich gap transient. It is not in the supernova list above. TNS does not type these events `SN …`. In the Lasair objects already on disk, the matching spectroscopic class is `Gap` (6 objects), and `Gap` does not start with `SN` or `SLSN`, so `broad_class` already returns `not SN`. Labelling the CART simulations as supernovae would train on them as SN and then score the real Gap objects as not SN.

Calcium-rich events are sometimes described as a faint thermonuclear supernova. That is a statement about the explosion, not about the TNS type. This project follows the TNS type, so CART stays 0.
