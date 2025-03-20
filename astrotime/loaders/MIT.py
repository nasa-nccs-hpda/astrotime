import time, numpy as np, xarray as xa
from astrotime.loaders.base import DataLoader
from typing import List, Optional, Dict, Type, Union, Tuple
from astrotime.util.logging import exception_handled
import os
import pandas as pd
from glob import glob
from omegaconf import DictConfig, OmegaConf
import logging

class MITLoader(DataLoader):

	def __init__(self, cfg: DictConfig ):
		super().__init__()
		self.cfg = cfg
		self.sector_range = cfg.sector_range
		self.current_sector = None
		self.dataset = None

	def TICS( self, sector_index: int ) -> List[str]:
		bls_dir = f"{self.cfg.dataset_root}/sector{sector_index}/bls"
		files = glob("*.bls", root_dir=bls_dir )
		print( f"Get TICS from {bls_dir}, nfiles: {len(files)}")
		return [ f.split('.')[0] for f in files ]

	def bls_file_path( self, sector_index: int, TIC: str ) -> str:
		return f"{self.cfg.dataset_root}/sector{sector_index}/bls/{TIC}.bls"

	def lc_file_path( self, sector_index: int, TIC: str ) -> str:
		return f"{self.cfg.dataset_root}/sector{sector_index}/lc/{TIC}.txt"

	@exception_handled
	def load_sector(self, sector: int) -> bool:
		if (self.current_sector != sector) or (self.dataset is None):
			TICS: List[str] = self.TICS(sector)
			periods = []
			times = []
			fluxes = []
			sns = []
			for TIC in TICS:
				data_file = self.bls_file_path(sector,TIC)
				dfbls = pd.read_csv( data_file, header=None, names=['Header', 'Data'] )
				dfbls = dfbls.set_index('Header').T
				period = np.float64(dfbls['per'].values[0])
				sn = np.float64(dfbls['sn'].values[0])
				dflc = pd.read_csv( self.lc_file_path(sector,TIC), header=None, sep='\s+')
				time = dflc[0].values
				flux = dflc[1].values
				periods.append(period)
				sns.append(sn)
				times.append(time)
				fluxes.append(flux)
			print( f"periods[{len(periods)}]")
			print(f"sns[{len(sns)}]")
			print(f"times[{len(times)}]")
			print(f"fluxes[{len(fluxes)}]")

# 			pd.read_csv( path + sector + '/bls/ ' + TIC + '.bls', header=None, names=['Header', 'Data'])
#
# 			if file_path is not None:
# 				t0 = time.time()
# 				self.dataset: xa.Dataset = xa.open_dataset(file_path, engine="netcdf4")
# 				slen: int = self.dataset['slen'].values.min()
# 				y: xa.DataArray = self.dataset['y'].isel(time=slice(0, slen))
# 				t: xa.DataArray = self.dataset['t'].isel(time=slice(0, slen))
# 				p: xa.DataArray = self.dataset['p']
# 				f: xa.DataArray = self.dataset['f']
# 				self.current_file = file_index
# 				self._nelements = self.dataset.sizes['elem']
# 				self.dataset = xa.Dataset(dict(y=y, t=t, p=p, f=f))
# 				self.log.info(f"Loaded {self._nelements} sinusoids in {time.time() - t0:.3f} sec from file: {file_path}, freq range = [{f.values.min():.3f}, {f.values.max():.3f}]")
# 				return True
# 		return False
#
#
# 		self.periods =[]
# 		self.times =[]
# 		self.fluxes =[]
# 		self.sns =[]
#
# for sector in tqdm(sectors):
# 	blspat h =pat h +secto r +'/bls/'
# 	file s =os.listdir(blspath)
# 	for f in tqdm(files):
#
# 		try:
#
# 			TI C =f.split('.')[0]
# 			dfbls =pd.read_csv(path+sector+'/bls/ '+TIC+'.bls',header=None,names=['Header' ,'Data'])
# 			dfbls = dfbls.set_index('Header').T
# 			period = np.float64(dfbls['per'].values[0])
# 			sn = np.float64(dfbls['sn'].values[0])
# 			dflc =pd.read_csv(header=None,delim_whitespace=True)
# 			time =dflc[0].values
# 			flux =dflc[1].values
# 			periods.append(period)
# 			sns.append(sn)
# 			times.append(time)
# 			fluxes.append(flux)
#
# 		except:
#
# 			print('error reading file')