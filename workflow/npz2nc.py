import time, os, shutil, sys, argparse, traceback, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union
import xarray as xa

rootdir = "/explore/nobackup/projects/ilab/data/astrotime/synthetic/"
dset = "astro_signals_with_noise"

narchives = 10
ncfilesize = 1000
archive_size = 100000
files_per_archive: int = archive_size // ncfilesize
start_archive = 0

for archive_idx in range(start_archive,narchives):
    file_idx = archive_idx * files_per_archive
    archive_path = f"{rootdir}/npz/{dset}_{archive_idx}.npz"
    tmp_path = f"{os.path.expanduser('~')}/tmp/{dset}.npz"
    if os.path.exists(tmp_path): os.remove(tmp_path)
    shutil.copyfile( archive_path, tmp_path)
    data = np.load( tmp_path, allow_pickle=True )
    print( f"Loaded data[{archive_idx}] from {archive_path}, archive_size={archive_size}, ncfilesize={ncfilesize}, files_per_archive={files_per_archive}" )
    y,t,p,st = data["signals"], data["times"], data["periods"], data["types"]

    xvars, var_idx = {}, 0
    for vid in range(archive_size):
        ncpath = f"{rootdir}/nc/{dset}-{file_idx}.nc"
        if not os.path.exists(ncpath):
            signal, tvar, period, stype = y[vid].astype(np.float32), t[vid].astype(np.float32), p[vid], st[vid]
            tcoord, sname = f"t{var_idx}", f"s{var_idx}"
            xvars[sname] = xa.DataArray(signal, name=sname, dims=[tcoord], coords={tcoord: tvar}, attrs=dict(period=period, type=stype))
        if var_idx == ncfilesize-1:
            nvars = len(xvars)
            if nvars > 0:
                xa.Dataset( xvars ).to_netcdf(ncpath)
                print(f" * A{archive_idx}.V{vid//nvars}: Wrote {nvars} vars to file: {ncpath}")
            xvars, file_idx, var_idx = {}, file_idx+1, 0
        else:
            var_idx += 1

    os.remove(tmp_path)


