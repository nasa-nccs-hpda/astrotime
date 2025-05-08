from torch import Tensor
from torch.nn.modules import Module

class PeakFinder(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input
