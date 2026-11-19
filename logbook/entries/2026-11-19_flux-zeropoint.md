# 19 Nov 2026 — ELAsTiCC flux zeropoint

Closed. `FLUXCAL` and `FLUXCALERR` are converted to nanojansky with one factor,

```
10^((31.4 - 27.5) / 2.5) = 36.307805
```

which is the 36.308 quoted from the two zeropoints in `A_FORMAT.TXT`. `elasticc_reader.py` multiplies `FLUXCAL` by that factor once and `FLUXCALERR` by the same factor once. It does not multiply again.

The check is an r-band point within 0.5 day of `SIM_PEAKMJD`, in the first SNIa-SALT3 HEAD/PHOT pair (`ELASTICC2_TRAIN_02_NONIaMODEL0-0001`). Magnitude from the converted flux, `31.4 - 2.5 log10(flux_nJy)`, matches magnitude from the SNANA zeropoint, `27.5 - 2.5 log10(FLUXCAL)`, for every object below. Both the flux ratio and the error ratio equal 36.307805, not its square.

| SNID | SIM_PEAKMAG_r | mag from nJy | difference | SNR |
|---|---:|---:|---:|---:|
| 31375154 | 18.266 | 18.268 | +0.003 | 189 |
| 57782528 | 20.027 | 20.027 | +0.0005 | 98 |
| 7151399 | 22.594 | 22.701 | +0.108 | 19 |
| 20039454 | 23.478 | 23.256 | −0.222 | 8.5 |
| 70771399 | 23.961 | 23.890 | −0.072 | 6.0 |
| 2830634 | 26.001 | 26.012 | +0.011 | 1.3 |

The two bright supernovae land on `SIM_PEAKMAG_r` to a few millimag. The fainter ones miss by tenths of a magnitude, which is the noise on that epoch (the SNR column), not a second factor of 36.308. Of the 359 SNe Ia in this file with an r point that close to peak, the median difference from `SIM_PEAKMAG_r` is +0.004 mag.

Reading the written table back for SNID 57782528 gives the same ratio, 36.307805, on both `flux` and `fluxErr`.
