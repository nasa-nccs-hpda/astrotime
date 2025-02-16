import sys, argparse
from argparse import Namespace
import xarray.core.coordinates
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Optional, Union, Hashable
from astrotime.util.logging import lgm, exception_handled, log_timing
from datetime import date, timedelta, datetime
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
import hydra, traceback, os
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
DataCoordinates = Union[DataArrayCoordinates,DatasetCoordinates]
default_args = Namespace(gpu=0,world_size=1,port=0)

def cfg() -> DictConfig:
    return ConfigContext.cfg

def get_device():
    return ConfigContext.device

def config() -> Dict:
    return ConfigContext.configuration

def cid() -> str:
    return ConfigContext.cid

def cfgdir() -> str:
    cdir = Path(__file__).parent.parent.parent / "config"
    print( f'cdir = {cdir}')
    return str(cdir)

def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description=f'Execute corrdiff workflow')
    argparser.add_argument('scenario',  help="Name of preprocessed input dataset" )
    return argparser.parse_args()

class ConfigContext(initialize):
    cfg: Optional[DictConfig] = None
    cid: Optional[str] = None
    cname: Optional[str] = None
    defaults: Dict = {}
    configuration: Dict = {}
    device = None

    def __init__(self, ccustom: Dict = None, device = None ):
        assert self.cfg is None, "Only one ConfigContext instance is allowed at a time"
        ConfigContext.configuration = dict(**self.defaults)
        if ccustom is not None: ConfigContext.configuration.update(ccustom)
        ConfigContext.device = device
        self.training: str = self.get_config('training')
        self.platform: str = self.get_config('platform')
        self.dataset: str = self.get_config('dataset')
        self.task: str = self.get_config('task')
        self.transform: str = self.get_config('transform')
        self.config_path: str = self.get_config('config_path', "../../config")
        super(ConfigContext, self).__init__(version_base=None, config_path=self.config_path)

    @classmethod
    def initialize(cls, cname: str, device, configuration: Dict[str,Any], ccustom: Dict[str,Any]=None ):
        cls.set_defaults(cname, **configuration)
        return cls.activate_global(ccustom,device)

    @property
    def model(self):
        if "arch" in cfg().model: return ".".join([cfg().model.model, cfg().model.arch])
        else:                     return cfg().model.model

    def to_config_file(self, **kwargs) -> str:
        cfg_data = DictConfig( { s:self.cfg[s] for s in self.defaults.keys() } )
        cfg_data['rootcfg'] = DictConfig( self.defaults )
        OmegaConf.resolve(cfg_data)
        print( f" ___________ Config data ___________ ")
        keys = kwargs.get( 'sections', self.defaults.keys())
        cfg_content = {}
        for s in keys:
            print(f"\n\t {s}:  {cfg_data[s]}" )
            cfg_content[s] = cfg_data[s]
        print( f" save config: keys = {list(cfg_content.keys())}" )
        OmegaConf.save( DictConfig(cfg_content), self.cfg_cache_file )
        return self.cfg_cache_file

    # def init_lightning(self, cli: LightningCLI ):
    #     os.makedirs(self.cfg.task.log_dir, exist_ok=True)
    #     os.makedirs(cli.trainer.default_root_dir, exist_ok=True)
    #     normalization = cli.datamodule.output_transforms
    #     mean_norm, std_norm = normalization.mean, normalization.std
    #     mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    #     cli.model.set_denormalization(mean_denorm, std_denorm)
    #     cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    #     cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    #     cli.model.set_val_clim(cli.datamodule.val_clim)
    #     cli.model.set_test_clim(cli.datamodule.test_clim)

    @classmethod
    def get_config( cls, name: str, default: Any = None ):
        return cls.configuration.get( name, cls.defaults.get(name,default) )

    @classmethod
    def set_defaults(cls, cname: str, **kwargs):
        cls.cname = cname
        cls.defaults = kwargs

    @property
    def cfg_file( self ):
        currdir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath( os.path.join(currdir, self.config_path,  f"{self.cname}.yaml") )

    @property
    def cfg_cache_file( self ):
        cdir = f"{self.cfg.platform.cache}/config"
        os.makedirs(cdir, exist_ok=True)
        return f"{cdir}/{self.cname}.yaml"

    @classmethod
    def deactivate(cls):
        cls.cfg = None

    @classmethod
    def activate_global(cls, ccustom: Dict, device, rank: int = 0 ) -> 'ConfigContext':
        cc = ConfigContext( rank, ccustom, device )
        cc.activate()
        lgm().init_logging(rank)
        return cc

    def activate(self):
        assert ConfigContext.cfg is None, "Context already activated"
        cfg = ConfigContext.cfg = self.load()
        print( f" ___________ Config context ___________ {self.model}")
        ConfigContext.cid =  '-'.join( [self.cname, self.dataset, self.model, self.task, self.transform,  self.training] )
        if self.rank == 0: print( f"Activating {self.cname}: '{self.cfg_file}', keys = {list(self.cfg.keys())}")
        cfg.training.name = self.defaults['training']
        cfg.platform.name = self.defaults['platform']
        cfg.task.task = self.defaults['task']
        cfg.task.dataset = self.defaults['dataset']
        cfg.task.training_version = self.cid

    def load(self) -> DictConfig:
        assert self.cfg is None, "Another Config context has already been activateed"
        if not GlobalHydra().is_initialized():
            hydra.initialize(version_base=None, config_path=self.config_path)
        cfg = hydra.compose(config_name=self.cname, overrides=[f"{ov[0]}={ov[1]}" for ov in self.configuration.items()])
        #OmegaConf.resolve(cfg)
        return cfg

    def __enter__(self, *args: Any, **kwargs: Any):
       super(ConfigContext, self).__enter__(*args, **kwargs)
       self.activate()
       return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
       super(ConfigContext, self).__exit__(exc_type, exc_val, exc_tb)
       self.deactivate()
       if exc_type is not None:
           traceback.print_exception( exc_type, value=exc_val, tb=exc_tb)


def cfg2meta(csection: str, meta: object, on_missing: str = "ignore"):
    csections = csection.split(".")
    cmeta = cfg().get(csections[0])
    if (len(csections) > 1) and (cmeta is not None): cmeta = cmeta.get(csections[1])
    if cmeta is None:
        print( f"Warning: section '{csection}' does not exist in configuration" )
        return None
    for k,v in cmeta.items():
        valid = True
        if (getattr(meta, k, None) is None) and (on_missing != "ignore"):
            msg = f"Attribute '{k}' does not exist in metadata object"
            if on_missing.startswith("warn"): print("Warning: " + msg)
            elif on_missing == "skip": valid = False
            elif on_missing.startswith("excep"): raise Exception(msg)
            else: raise Exception(f"Unknown on_missing value in cfg2meta: {on_missing}")
        if valid: setattr(meta, k, v)
    return meta

def cfg2args( csection: str, pnames: List[str] ) -> Dict[str,Any]:
    csections = csection.split(".")
    cmeta = cfg().get(csections[0])
    if (len(csections) > 1) and (cmeta is not None): cmeta = cmeta.get(csections[1])
    args = {}
    if cmeta is None:
        print( f"Warning: section '{csection}' does not exist in configuration" )
    else:
        for pn in pnames:
            if pn in cmeta.keys():
                aval = cmeta.get(pn)
                if str(aval) == "None": aval = None
                args[pn] = aval
    return args

def cfg_date( csection: str ) -> date:
    dcfg = cfg().get(csection)
    return date( dcfg.year, dcfg.month, dcfg.day )

def start_date( task: DictConfig )-> Optional[datetime]:
    startdate = task.get('start_date', None)
    if startdate is None: return None
    toks = [ int(tok) for tok in startdate.split("/") ]
    return  datetime( month=toks[0], day=toks[1], year=toks[2] )

def dateindex(d: datetime, task: DictConfig) -> int:
    sd: date = start_date(task)
    dt: timedelta = d - sd
    hours: int = (dt.seconds // 3600) + (dt.days * 24)
    # print( f"dateindex: d[{d.strftime('%H:%d/%m/%Y')}], sd[{sd.strftime('%H:%d/%m/%Y')}], dts={dt.seconds}, hours={hours}")
    return hours + 1

def index_of_value( array: np.ndarray, target_value: float ) -> int:
    differences = np.abs(array - target_value)
    return differences.argmin()

def closest_value( array: np.ndarray, target_value: float ) -> float:
    differences = np.abs(array - target_value)
    print( f"Closest value: array{array.shape}, target={target_value}, differences type: {type(differences)}")
    return  float( array[ differences.argmin() ] )

def get_coord_bounds( coord: np.ndarray ) -> Tuple[float, float]:
    dc = coord[1] - coord[0]
    return  float(coord[0]), float(coord[-1]+dc)

def get_dims( coords: DataCoordinates, **kwargs ) -> List[str]:
    dims = kwargs.get( 'dims', ['x','y'] )
    dc: List[Hashable] = list(coords.keys())
    if 'x' in dc:
        return dims
    else:
        cmap: Dict[str, str] = cfg().task.coords
        vs: List[str] = list(cmap.values())
        if vs[0] in dc:
            return [ cmap[k] for k in dims ]
        else:
            raise Exception(f"Data Coordinates {dc} do not exist in configuration")

def get_roi( coords: DataCoordinates ) -> Dict:
    return { dim: get_coord_bounds( coords[ dim ].values ) for dim in get_dims(coords) }

def get_data_coords( data: xarray.DataArray, target_coords: Dict[str,float] ) -> Dict[str,float]:
    return { dim: closest_value( data.coords[ dim ].values, cval ) for dim, cval in target_coords.items() }

def cdelta(dset: xarray.DataArray):
    return { k: float(dset.coords[k][1]-dset.coords[k][0]) for k in dset.coords.keys() if dset.coords[k].size > 1 }

def cval( data: xarray.DataArray, dim: str, cindex ) -> float:
    coord : np.ndarray = data.coords[ cfg().task.coords[dim] ].values
    return float( coord[cindex] )

def get_data_indices( data: Union[xarray.DataArray,xarray.Dataset], target_coords: Dict[str,float] ) -> Dict[str,int]:
    return { dim: index_of_value( data.coords[ dim ].values, coord_value ) for dim, coord_value in target_coords.items() }



