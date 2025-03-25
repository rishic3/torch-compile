import torch

# TORCH_LOGS=graph_code python dynamo.py

@torch.compile
def mse(x, y):
    z = (x - y) ** 2
    return z.sum()

@torch.compile
def fn(x, n):
    y = x ** 2
    if n >= 0:
        return (n + 1) * y
    else:
        return y / n

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    
    print("==" * 20)
    print(f"Testing simple mse:")
    print("==" * 20)
    x = torch.randn(200)
    y = torch.randn(200)
    mse(x, y)

    print("==" * 20)
    print(f"Testing control flow:")
    print("==" * 20)
    fn(x, 1)
    fn(x, -1)
