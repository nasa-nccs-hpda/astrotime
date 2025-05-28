import time, argparse, traceback, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union
import xarray as xa

rootdir = "/explore/nobackup/projects/ilab/data/astrotime/synthetic/"
dset = "astro_signals_with_noise"

nfiles = 10
ncfilesize = 1000
archive_size = 100000
files_per_archive: int = archive_size // ncfilesize

for archive_idx in range(nfiles):
    npz_path = f"{rootdir}/npz/{dset}_{archive_idx}.npz"
    t0 = time.time()
    data=np.load( npz_path, allow_pickle=True, mmap_mode="r" )
    print( f"Loaded data[{archive_idx}] from {npz_path}, archive_size={archive_size}, ncfilesize={ncfilesize}, files_per_archive={files_per_archive}, in {time.time()-t0:.3f}s" )

    xvars = {}
    for vid in range(archive_size):
        signal, tvar, period, stype = data["signals"][vid].astype(np.float32), data["times"][vid].astype(np.float32), data["periods"][vid], data["types"][vid]
        tcoord, sname = f"t{archive_idx}{vid}", f"s{archive_idx}{vid}"
        xvars[sname] = xa.DataArray(signal, dims=[tcoord], coords={tcoord: tvar}, attrs=dict(period=period, type=stype))
        if (vid > 0) and ((vid % 10) == 0): print(".",end="")
        if (vid>0) and ((vid % ncfilesize) == 0):
            file_idx = (vid-1)//files_per_archive + archive_idx*files_per_archive
            ncpath = f"{rootdir}/nc/{dset}-{file_idx}.nc"
            xa.Dataset( xvars ).to_netcdf(ncpath)
            print( f"\n ----> Wrote file {ncpath}" )


