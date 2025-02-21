import logging, torch, math, csv
from astrotime.util.logging import lgm, exception_handled, log_timing
from io import TextIOWrapper
from astrotime.config.context import ConfigContext, cfg
import os, time, yaml, numpy as np
from collections import deque
from astrotime.config.context import cfg
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Sequence, Mapping, Callable
from omegaconf import DictConfig
from astrotime.util.config import TSet

def tofloat( x ) -> float:
	if type(x) == torch.Tensor: return float(x.detach().cpu().item())
	else:                       return float(x)

def pkey( tset: TSet, ltype: str ): return '-'.join([tset.value,ltype])

def tidx() -> int:
	return int(time.time()/10)

def version_test( test: str ):
	try:
		tset = TSet(test)
		return 0
	except ValueError:
		return 1

def get_temporal_features( time: np.ndarray = None ) -> Optional[np.ndarray]:
	if time is None: return None
	sday, syear, t0, pi2 = [], [], time[0], 2 * np.pi
	for idx, t in enumerate(time):
		td: float = float((t - t0) / np.timedelta64(1, 'D'))
		sday.append((np.sin(td * pi2), np.cos(td * pi2)))
		ty: float = float((t - t0) / np.timedelta64(365, 'D'))
		syear.append([np.sin(ty * pi2), np.cos(ty * pi2)])
	# print( f"{idx}: {pd.Timestamp(t).to_pydatetime().strftime('%m/%d:%H/%Y')}: td[{td:.2f}]=[{sday[-1][0]:.2f},{sday[-1][1]:.2f}] ty[{ty:.2f}]=[{syear[-1][0]:.2f},{syear[-1][1]:.2f}]" )
	tfeats = np.concatenate([np.array(tf,dtype=np.float32) for tf in [sday, syear]], axis=1)
	return tfeats.reshape(list(tfeats.shape) + [1, 1])

def rrkey( tset: TSet, **kwargs ) -> str:
	epoch = kwargs.get('epoch', -1)
	epstr = f"-{epoch}" if epoch >= 0 else ''
	return f"{tset.value}{epstr}"

class ResultRecord(object):

	def __init__(self, tset: TSet, epoch: float, losses: Dict[str,float] ):
		self.losses: Dict[str,float] = losses
		self.accuracy: float = 0.0 if (self.ratio is None) else 1.0/self.ratio
		self.epoch: float = tofloat(epoch)
		self.tset: TSet = tset

	@classmethod
	def parse(cls, rec: List[str]) -> "ResultRecord":
		tset = TSet(rec[0])
		epoch = float(rec[1])
		losses: Dict[str,float]  = {}
		for item in rec:
			toks = item.split(':')
			if len(toks) == 2:
				losses[ toks[0].strip() ] = float(toks[1])
		return ResultRecord( tset, epoch, losses )

	@property
	def ratio(self):
		return self.losses.get('ratio',None)

	@property
	def slosses(self)-> List[str]:
		print( 'self.losses :' + str(self.losses) )
		return [f"{k}: {v:.4f}" for k, v in self.losses.items() if v is not None]

	def serialize(self) -> List[str]:
		return [ self.tset.value, f"{self.epoch:.3f}" ] + self.slosses

	def __str__(self):
		return f" --- TSet: {self.tset.value}, Epoch: {self.epoch:.3f},  Losses: {self.slosses}"

class ResultFileWriter:

	def __init__(self, file_path: str):
		self.file_path = file_path
		self._csvfile: TextIOWrapper = None
		self._writer: csv.writer = None

	@property
	def csvfile(self) -> TextIOWrapper:
		if self._csvfile is None:
			self._csvfile = open(self.file_path, 'a', newline='\n')
		return self._csvfile

	def refresh(self):
		if self._csvfile is not None:
			os.rename(self.file_path, f"{self.file_path}.{tidx()}")
		self._csvfile = None

	@property
	def csvwriter(self) -> csv.writer:
		if self._writer is None:
			self._writer = csv.writer(self.csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		return self._writer

	def write_entry(self, entry: List[str]):
		self.csvwriter.writerow(entry)

	def close(self):
		if self._csvfile is not None:
			self._writer = None
			self._csvfile.close()
			self._csvfile = None


class ResultFileReader:

	def __init__(self, file_paths: List[str] ):
		self.file_paths = file_paths
		self._csvfiles: List[TextIOWrapper] = None
		self._readers: List[csv.reader] = None

	@property
	def csvfiles(self) -> List[TextIOWrapper]:
		if self._csvfiles is None:
			self._csvfiles = []
			for file_path in self.file_paths:
				try:
					self._csvfiles.append( open( file_path, 'r', newline='' ) )
				except FileNotFoundError:
					pass
		return self._csvfiles

	@property
	def csvreaders(self) -> List[csv.reader]:
		if self._readers is None:
			self._readers = []
			for csvfile in self.csvfiles:
				self._readers.append( csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL) )
		return self._readers

	def close(self):
		if self._csvfiles is not None:
			self._readers = None
			for csvfile in self._csvfiles:
				csvfile.close()
			self._csvfiles = None

class ResultsAccumulator(object):

	def __init__(self, **kwargs):
		self.results: List[ResultRecord] = []
		self._writer: Optional[ResultFileWriter] = None
		self._reader: Optional[ResultFileReader] = None

	@property
	def reader(self) -> ResultFileReader:
		if self._reader is None:
			self._reader = ResultFileReader( [ self.result_file_path(model_specific=True) ] )
		return self._reader

	@property
	def writer(self) -> ResultFileWriter:
		if self._writer is None:
			self._writer = ResultFileWriter( self.result_file_path() )
		return self._writer

	def result_file_path(self, model_specific = True ) -> str:
		results_save_dir = f"{self.save_dir}/{self.task.task}_result_recs"
		os.makedirs(results_save_dir, exist_ok=True)
		model_id = f"_{self.model.model}" if model_specific else ""
		return f"{results_save_dir}/{self.dataset.source}_{model_id}_losses.csv"

	def refresh_state(self):
		rfile =self.result_file_path()
		if os.path.exists( rfile ):
			os.remove( rfile )

	def close(self):
		if self._reader is not None:
			self._reader.close()
			self._reader = None
		if self._writer is not None:
			self._writer.close()
			self._writer = None
		del self.results
		self.results = []
		torch.cuda.empty_cache()


	def record_losses(self, tset: TSet, epoch: float, losses: Dict[str,float], flush=True):
		rr: ResultRecord = ResultRecord(tset, epoch, losses )
		self.results.append( rr )
		if flush: self.flush()

	def serialize(self)-> Dict[ str, Tuple[str,float,float] ]:
		sr =  { k: rr.serialize() for k, rr in self.results }
		return sr

	def flush(self):
		self.save()
		self.close()

	@exception_handled
	def save(self):
		print( f" ** Saving {len(self.results)} training stats to {self.result_file_path()}")
		for result in self.results:
			self.writer.write_entry( result.serialize() )

	def load_row(self, row: List[str]):
		rec = ResultRecord.parse(row)
		if rec is not None:
			self.results.append(rec)

	def load_results( self ):
		for reader in self.reader.csvreaders:
			for row in reader:
				self.load_row(row)
		print(f" ** Loading training stats ({len(self.results)} recs) from {self.result_file_path()}")

	def get_plot_data(self) -> Tuple[Dict[TSet,np.ndarray],Dict[TSet,np.ndarray]]:
		model_data = {}
		print(f"get_plot_data: {len(self.results)} results")
		for tset in [TSet.Train, TSet.Validation]:
			result_data = model_data.setdefault( tset, {} )
			for result in self.results:
				if (result.tset == tset) and (result.ratio is not None):
					result_data[ result.epoch ] = 1.0/result.ratio

		x, y = {}, {}
		for tset in model_data.keys():
			result_data = dict(sorted(model_data[ tset ].items()))
			x[tset] = np.array(list(result_data.keys()))
			y[tset] = np.array(list(result_data.values()))

		return x, y

class LossAccumulator:

	def __init__(self, **kwargs ):
		self._batch_losses: Dict[str,List[float]] = {}
		self._means = {}
		self._mean_length = kwargs.get("mean_length",25)

	def __getitem__(self,ltype: str) -> float:
		return self.accumulate_loss(ltype)

	@property
	def types(self) -> List[str]:
		return list(self._batch_losses.keys())

	def _means_accum(self, ltype: str) -> deque:
		return self._means.setdefault( ltype, deque([]) )

	def nlosses(self, ltype: str) -> int:
		return len( self._batch_losses[ltype] )

	def batch_losses(self, ltype: str) -> List[float]:
		return self._batch_losses.setdefault(ltype, [])

	def clear_batch_losses(self, ltype: str):
		self._batch_losses[ltype] = []

	def register_loss(self, ltype: str, loss: float|torch.Tensor) -> float:
		if type(loss) is torch.Tensor: loss = loss.detach().item()
		self.batch_losses(ltype).append(loss)
		self._add_mean_loss(ltype, loss)
		return loss

	def _add_mean_loss(self, ltype: str, loss: float):
		means: deque = self._means_accum(ltype)
		if len(means) >= self._mean_length: means.popleft()
		means.append(loss)

	def running_mean(self, ltype: str) -> float:
		ma: deque = self._means_accum(ltype)
		return 0.0 if (len(ma)==0) else np.array(ma).mean()

	def accumulate_losses(self)-> Dict[str,float]:
		alosses = {ltype: self.accumulate_loss(ltype) for ltype in self.types}
		return { k:v for k,v in alosses.items() if (v is not None) }

	def accumulate_loss(self, ltype: str) -> Optional[float]:
		losses = self.batch_losses(ltype)
		if len(losses) > 0:
			accum_loss = np.array(losses).mean()
			self.clear_batch_losses(ltype)
			return tofloat(accum_loss)


