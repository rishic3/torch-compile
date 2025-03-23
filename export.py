import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

if __name__ == "__main__":
    example_args = (torch.randn(10, 10), torch.randn(10, 10))

    print("==" * 10)
    print("Example args:")
    print("==" * 10)
    print(example_args)

    exported_program: torch.export.ExportedProgram = export(
        Mod(), args=example_args
    )

    print("==" * 10)
    print("Exported program:")
    print("==" * 10)
    print(exported_program)