import time, os, shutil, argparse, traceback, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union
import xarray as xa

rootdir = "/explore/nobackup/projects/ilab/data/astrotime/synthetic/"
dset = "astro_signals_with_noise"

nfiles = 10
ncfilesize = 1000
archive_size = 100000
files_per_archive: int = archive_size // ncfilesize

for archive_idx in range(1,nfiles):
    archive_path = f"{rootdir}/npz/{dset}_{archive_idx}.npz"
    tmp_path = f"{os.path.expanduser('~')}/{dset}.npz"
    shutil.copyfile( archive_path, tmp_path)
    data = np.load( tmp_path, allow_pickle=True ) # , mmap_mode='r+' )
    archive_segment_idx = 0
    os.remove(tmp_path)
    print( f"Loaded data[{archive_idx}] from {archive_path}, archive_size={archive_size}, ncfilesize={ncfilesize}, files_per_archive={files_per_archive}" )
    y,t,p,st = data["signals"], data["times"], data["periods"], data["types"]
    xvars = {}
    for vid in range(archive_size):
        signal, tvar, period, stype = y[vid].astype(np.float32), t[vid].astype(np.float32), p[vid], st[vid]
        tcoord, sname = f"t{archive_idx}{vid}", f"s{archive_idx}{vid}"
        xvars[sname] = xa.DataArray(signal, dims=[tcoord], coords={tcoord: tvar}, attrs=dict(period=period, type=stype))
        if (vid % 100) == 99: print(".",flush=True)
        else:                 print(".", end="",flush=True)
        if (vid>0) and ((vid % ncfilesize) == 0):
            file_idx = archive_segment_idx + archive_idx*files_per_archive
            ncpath = f"{rootdir}/nc/{dset}-{file_idx}.nc"
            xa.Dataset( xvars ).to_netcdf(ncpath)
            xvars, archive_segment_idx = {}, archive_segment_idx+1
            print( f"\n ----> Wrote file {ncpath}" )


