import torch
import sys

print("="*80)
print("GPU CHECK")
print("="*80)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU computation successful!")
        print(f"  Result device: {z.device}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("\n⚠️  No GPU detected!")
    print("Possible reasons:")
    print("  1. CUDA is not installed")
    print("  2. PyTorch was installed without CUDA support")
    print("  3. No NVIDIA GPU is available")
    print("\nTo install PyTorch with CUDA support:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("="*80)
