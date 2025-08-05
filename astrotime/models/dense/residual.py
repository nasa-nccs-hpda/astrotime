from torch import Tensor, device, nn


class ResidualBlock(nn.Module):

  def __init__(  self, input_dims: int, hidden_dims: int, output_dims: int ):
    super(ResidualBlock, self).__init__()
    self.input_dims: int = input_dims
    self.hidden_dims: int = hidden_dims
    self.output_dims: int = output_dims

    self.hidden_layer:   nn.Module  = nn.Sequential( nn.Linear(input_dims, hidden_dims), nn.SiLU() )
    self.output_layer:   nn.Module  = nn.Linear(hidden_dims, output_dims)
    self.residual_layer: nn.Module  = nn.Linear(input_dims, output_dims)

  def forward(self, x: Tensor ) -> Tensor:
    hidden: Tensor    = self.hidden_layer(x)
    output: Tensor    = self.output_layer(hidden)
    residual: Tensor  = self.residual_layer(x)
    return output + residual