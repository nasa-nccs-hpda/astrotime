import time, argparse, traceback, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union
import xarray as xa

rootdir = "/explore/nobackup/people/bppowel1/timehascome/"
dset = "astro_signals_with_noise"
#rootdir = "/explore/nobackup/projects/ilab/data/astrotime/sinusoids/"
# rootdir = "/Users/tpmaxwel/Data/astro_sigproc/sinusoids"

fstart = 0
nfiles = 10
ncfilesize = 1000

for archive_idx in range(fstart, fstart+nfiles):
    npz_path = f"{rootdir}/{dset}_{archive_idx}.npz"
    t0 = time.time()
    data=np.load( npz_path, allow_pickle=True )
    signals: np.ndarray = data["signals"]
    times: np.ndarray = data["times"]
    types: np.ndarray = data["types"]
    periods: np.ndarray = data["periods"]
    archive_size: int = signals.shape[0]
    files_per_archive: int = archive_size // ncfilesize
    print( f"Loaded data[{archive_idx}] from {npz_path}, archive_size={archive_size}, ncfilesize={ncfilesize}, files_per_archive={files_per_archive}, in {time.time()-t0:.3f}s" )

    for file_idx in range(files_per_archive):
        try:
            t1 = time.time()
            brng: Tuple[int,int] = (file_idx * ncfilesize, (file_idx+1) * ncfilesize)
            xvars = {}
            for vid, sidx in enumerate(range(*brng)):
                signal,time,period,stype = signals[sidx].astype(np.float32), times[sidx].astype(np.float32), periods[sidx], types[sidx]
                nan_mask = (np.isnan(signal) | np.isnan(time))
                signal, time, tcoord = signal[~nan_mask], time[~nan_mask], f"t{vid}"
                xvars[ f"s{vid}" ] = xa.DataArray(signal, dims=[tcoord], coords={tcoord:time}, attrs=dict(period=period, type=stype) )
            ncf_idx = archive_idx * files_per_archive + file_idx
            ncpath = f"{rootdir}/nc/{dset}_{ncf_idx}.nc"
            xa.Dataset( xvars ).to_netcdf(ncpath)
            print( f" ----> Wrote file {ncpath} in {(time.time()-t1):.2f}s" )
        except Exception as e:
            print(f"FILE {archive_idx}-{file_idx} Error-> {e}")
            traceback.print_exc(limit=100)
            break

