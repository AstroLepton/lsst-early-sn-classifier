'''
Cut a light curve to the end of night N after the first detection.

t0 is the time of the first detection. Night N ends at

    cut = t0 + N + 0.5

N = 0 keeps only the detections before that cut, the first alert.
N >= 1 also keeps non-detections from 30 days before t0 up to the cut.

The nights used here are 0, 1, 2, 3 and 5.

Check the cut against a fake light curve:

    python scripts/truncate.py
'''

import pandas as pd

#Nights at which a classifier will be asked for a label.
NIGHTS = (0, 1, 2, 3, 5)


def truncate(lc, N):
    """Rows available at the end of night N. lc needs mjd and isDetection."""
    t0 = lc.loc[lc.isDetection, "mjd"].min()
    cut = t0 + N + 0.5
    det = lc[lc.isDetection & (lc.mjd < cut)]
    if N < 1:
        return det                      # first alert: no forced photometry
    forced = lc[~lc.isDetection & (lc.mjd >= t0 - 30) & (lc.mjd < cut)]
    return pd.concat([det, forced]).sort_values("mjd")


def _fake_light_curve():
    """One object, t0 = 1000. Points sit on the cuts so each side can be checked.

    A point exactly at a cut is not kept: the test uses mjd < cut.
    A forced point exactly 30 days before t0 is kept: the test uses mjd >= t0 - 30.
    """
    rows = [
        # mjd, isDetection
        (969.0, False),    # 31 days before t0: outside the forced window
        (970.0, False),    # exactly t0 - 30
        (999.9, False),    # before the first detection, inside the window
        (1000.0, True),    # first detection, t0
        (1000.4, True),    # same night, before the N=0 cut at 1000.5
        (1000.5, True),    # exactly the N=0 cut
        (1000.6, True),    # after night 0
        (1000.7, False),   # forced point on the next night
        (1001.4, True),    # before the N=1 cut at 1001.5
        (1001.5, False),   # exactly the N=1 cut; kept on later nights
        (1001.6, True),    # after night 1, before the N=2 cut at 1002.5
        (1002.4, False),
        (1002.5, True),    # exactly the N=2 cut
        (1003.4, True),    # before the N=3 cut at 1003.5
        (1003.5, False),   # exactly the N=3 cut
        (1005.4, True),    # before the N=5 cut at 1005.5
        (1005.5, True),    # exactly the N=5 cut
        (1005.6, False),   # after night 5
    ]
    return pd.DataFrame({
        "objectId": 1,
        "mjd": [mjd for mjd, _ in rows],
        "band": "r",
        "flux": 1.0,
        "fluxErr": 0.1,
        "isDetection": [flag for _, flag in rows],
    })


# mjd of the rows night N must return, in time order.
# Worked out from t0 = 1000 and cut = t0 + N + 0.5, not from truncate().
_EXPECTED = {
    0: [1000.0, 1000.4],
    1: [970.0, 999.9, 1000.0, 1000.4, 1000.5, 1000.6, 1000.7, 1001.4],
    2: [970.0, 999.9, 1000.0, 1000.4, 1000.5, 1000.6, 1000.7, 1001.4,
        1001.5, 1001.6, 1002.4],
    3: [970.0, 999.9, 1000.0, 1000.4, 1000.5, 1000.6, 1000.7, 1001.4,
        1001.5, 1001.6, 1002.4, 1002.5, 1003.4],
    5: [970.0, 999.9, 1000.0, 1000.4, 1000.5, 1000.6, 1000.7, 1001.4,
        1001.5, 1001.6, 1002.4, 1002.5, 1003.4, 1003.5, 1005.4],
}


def _check():
    lc = _fake_light_curve()
    for n in NIGHTS:
        out = truncate(lc, n)
        got = [round(mjd, 1) for mjd in out["mjd"]]
        assert got == _EXPECTED[n], f"N={n}: got {got}, expected {_EXPECTED[n]}"
        assert list(out["mjd"]) == sorted(out["mjd"]), f"N={n} is not in time order"
        if n < 1:
            assert out["isDetection"].all(), "N=0 must be detections only"
        else:
            # The forced point 31 days before t0 stays out. The one at 30 days stays in.
            assert 969.0 not in set(out["mjd"])
            assert 970.0 in set(out["mjd"])
            assert not out.loc[out["mjd"] == 970.0, "isDetection"].iloc[0]
    # Rows given out of order still come back sorted once forced points are included.
    reversed_lc = lc.iloc[::-1]
    ordered = truncate(reversed_lc, 1)["mjd"].tolist()
    assert ordered == sorted(ordered)
    print(f"truncate: ok for N = {', '.join(str(n) for n in NIGHTS)}")


if __name__ == "__main__":
    _check()
