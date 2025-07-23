
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="../../../../config", config_name="sinusoid_period")
def my_app(cfg: DictConfig) -> None:
	print( cfg )

if __name__ == "__main__":
	my_app()

