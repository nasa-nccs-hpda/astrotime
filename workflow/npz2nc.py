import time, argparse, traceback, numpy as np
from typing import Any, Dict, List, Tuple, Type, Optional, Union
from astrotime.config.context import ConfigContext
from scipy.signal import argrelmax, find_peaks
import xarray as xa

rootdir = "/explore/nobackup/projects/ilab/data/astro_sigproc/sinusoids"
argparser = argparse.ArgumentParser(description=f'Execute workflow')
argparser.add_argument('-nf', '--nfiles',  default=1,  type=int,  help="Number of filea to process")
argparser.add_argument('-f0', '--fstart', default=0,  type=int, help="Start file index")
argparser.add_argument('-gpu', '--gpu', nargs='?',  default=-1, type=int, help="GPU ID to use")
argparser.add_argument('-ws', '--world_size', nargs='?', default=1, type=int, help="Number of gpus to use in training")
argparser.add_argument('-p',  '--port', nargs='?', default=23467, type=int, help="Port to use in DDP")
clargs: argparse.Namespace = argparser.parse_args()

configuration = dict( task="freqid", platform="explore", training="sigproc", transform="wwz", dataset="sinusoid", model="wwznet" )
ccustom: Dict[str,Any] = { }
cc: ConfigContext = ConfigContext.initialize( "signal-analysis", configuration, clargs, ccustom )
file_size: int = cc.cfg.dataset.file_size

def compute_period( t: np.array, s: np.array ) -> float:
    peaks, props = find_peaks(s, distance=10)
    return t[peaks[1]] - t[peaks[0]]


for archive_idx in range(clargs.fstart, clargs.fstart+clargs.nfiles):
    npz_path = f"{rootdir}/npz/sinusoids_{archive_idx}.npz"
    t0 = time.time()
    data=np.load( npz_path, allow_pickle=True )
    sinusoids: np.ndarray = data["sinusoids"]
    times: np.ndarray = data["times"]
    periods: np.ndarray = data["periods"]
    archive_size: int = sinusoids.shape[0]
    files_per_archive: int = archive_size // file_size
    print( f"Loaded data[{archive_idx}] from {npz_path}, archive_size={archive_size}, file_size={file_size}, files_per_archive={files_per_archive}, in {time.time()-t0:.3f}s" )

    for file_idx in range(files_per_archive):
        try:
            elem_list = []
            t1 = time.time()
            brng: Tuple[int,int] = (file_idx * file_size, (file_idx+1) * file_size)
            series_lens: np.array = np.array( [ times[sidx].size for sidx in range(*brng) ] )
            maxlen: int = series_lens.max()
            for sidx in range(*brng):
                s,t,p = sinusoids[sidx].astype(np.float32), times[sidx].astype(np.float32), periods[sidx]
                nan_mask = (np.isnan(s) | np.isnan(t))
                s, t = s[~nan_mask], t[~nan_mask]
                slen = s.size
                t = np.pad( t, (0,maxlen-slen), 'constant', constant_values=np.nan )
                s = np.pad( s, (0,maxlen-slen), 'constant', constant_values=np.nan )
                elem_list.append( (s,t,p,slen) )
            elem_list.sort( key=lambda e:e[3], reverse=True )
            ls, lt, lp, lslen = zip(*elem_list)
            coords = dict(elem=archive_size*archive_idx+np.arange(*brng), time=np.arange(maxlen))
            y = xa.DataArray( np.stack( ls ), dims=['elem','time'], coords=coords )
            t = xa.DataArray( np.stack( lt ), dims=['elem','time'], coords=coords )
            p = xa.DataArray( np.array( lp ), dims=['elem'], coords=dict(elem=coords['elem']) )
            if archive_idx + file_idx == 0:
                for i in range(0,y.shape[0],100):
                    cp: float = compute_period(t.values[i,:], y.values[i,:])
                    print(f" *------->> F{file_idx}.E{brng}[{i}]: {p[i]:.3f} <-> {cp:.3f}")
            f = 1/p
            slen = xa.DataArray( np.array( lslen, dtype=np.int64), dims=['elem'], coords=dict(elem=coords['elem']) )
            dset = xa.Dataset( dict( y=y, t=t, p=p, f=f, slen=slen ), coords=coords )
            ncf_idx = archive_idx * files_per_archive + file_idx


            ncpath = f"{rootdir}/nc/padded_sinusoids_{ncf_idx}.nc"
            if file_idx % 10 == 0:
                print(f"Writing to file-{archive_idx}:{file_idx} ({ncpath}): y{list(y.shape)}, t{list(t.shape)}, p{list(p.shape)}, slen{list(slen.shape)}, coords: elem{list(coords['elem'].shape)}, time{list(coords['time'].shape)}")
            dset.to_netcdf(ncpath)
            if file_idx % 10 == 0:
                print( f" ----> Wrote {file_size} arrays in {(time.time()-t1):.2f}s, maxlen={maxlen}, slen: {lslen[0]} -> {lslen[-1]}" )
        except Exception as e:
            print(f"FILE {archive_idx}-{file_idx} Error-> {e}")
            traceback.print_exc(limit=100)
            break

