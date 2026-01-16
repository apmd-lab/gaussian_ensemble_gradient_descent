
import sys
import os
import torch
import numpy as np

# Adjust path to import necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from gegd.parameter_processing.bilevel_design_generator import conditional_generator, filter_and_project

def verify_batch_consistency():
    Nx = 10
    Ny = 10
    symmetry = 0
    periodic = False
    mfs = 1
    padding = -np.ones((Nx + 2*mfs, Ny + 2*mfs))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running on device: {device}")
    
    # Create two random inputs
    np.random.seed(42)
    x1_np = filter_and_project(np.random.uniform(-1, 1, size=(Nx*Ny)), periodic, Nx, Ny, mfs/2/np.sqrt(2), 8)/10
    x2_np = filter_and_project(np.random.uniform(-1, 1, size=(Nx*Ny)), periodic, Nx, Ny, mfs/2/np.sqrt(2), 8)/10
    
    x1 = torch.tensor(x1_np, device=device, dtype=torch.float64)
    x2 = torch.tensor(x2_np, device=device, dtype=torch.float64)
    
    # SETUP 1: Single processing
    print("Running single processing...")
    gen_single = conditional_generator(Nx, Ny, symmetry, periodic, padding, mfs + 2, x1, batch_processing=False)
    
    out1_single = gen_single.generate_near_binary_no_grad(x1)
    out2_single = gen_single.generate_near_binary_no_grad(x2)
    
    # SETUP 2: Batch processing
    print("Running batch processing...")
    batch_input = torch.stack([x1, x2])
    gen_batch = conditional_generator(Nx, Ny, symmetry, periodic, padding, mfs + 2, batch_input, batch_processing=True)
    
    out_batch = gen_batch.generate_near_binary_no_grad(batch_input)
    
    print(f"Batch output shape: {out_batch.shape}")
    
    # Compare results
    diff1 = torch.max(torch.abs(out_batch[0] - out1_single))
    diff2 = torch.max(torch.abs(out_batch[1] - out2_single))
    
    print(f"Max diff sample 1: {diff1}")
    print(f"Max diff sample 2: {diff2}")
    
    if diff1 < 1e-6 and diff2 < 1e-6:
        print("VERIFICATION PASSED: Batch processing matches single processing.")
    else:
        print("VERIFICATION FAILED: Results diverge.")

if __name__ == "__main__":
    verify_batch_consistency()
