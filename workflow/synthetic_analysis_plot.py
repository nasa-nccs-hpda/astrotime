from astrotime.util.series import TSet
from astrotime.loaders.synthetic import SyntheticLoader
from astrotime.encoders.wavelet import WaveletAnalysisLayer, embedding_space
from astrotime.plot.analysis import DatasetPlot, EvaluatorPlot
from astrotime.config.context import astrotime_initialize
from astrotime.plot.base import SignalPlotFigure
from astrotime.trainers.model_evaluator import ModelEvaluator
import torch
from hydra import initialize, compose

version = "synthetic_period"
overrides = []
initialize(version_base=None, config_path="../config" )
cfg = compose( config_name=version, overrides=overrides )
device: torch.device = astrotime_initialize(cfg,version+".plot")
sector = cfg.data.sector_range[0]
cfg.platform['gpu'] = -1

data_loader = SyntheticLoader( cfg.data )
data_loader.initialize( TSet.Train )
embedding_space_array, embedding_space_tensor = embedding_space( cfg.transform, device )

dplot = DatasetPlot(f"{version}: Lightcurves", data_loader, sector )
embedding = WaveletAnalysisLayer( 'analysis', cfg.transform, embedding_space_tensor, device )
evaluator = ModelEvaluator( cfg, version, data_loader, embedding, device )
wplot = EvaluatorPlot("WWAnalysis Transform", evaluator, sector )

fig = SignalPlotFigure([dplot,wplot])
fig.show()