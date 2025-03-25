import torch
import torch.fx

# TORCH_LOGS=graph_code python fx_inductor.py

class MyModule(torch.nn.Module):
    def forward(self, x):
        y = torch.sin(x)
        z = y * 2
        return z + 1

m = MyModule()
traced = torch.fx.symbolic_trace(m)
print(traced.graph)

print("==" * 20)

@torch.compile
def foo(x):
    y = torch.sin(x)
    z = y * 2
    return z + 1

print(foo(torch.randn(10)))
