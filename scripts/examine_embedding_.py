import torch
import time

device = torch.device("cuda:0")
print(f"Using device: {device}")

# Allocate a tensor
dummy = torch.randn((4096, 4096), device=device)
try:
    while True:
        # Active GPU work phase
        work_duration = 80  # seconds
        work_start = time.time()
        while time.time() - work_start < work_duration:
            for _ in range(2):  # Light computation
                dummy = dummy @ dummy
                dummy = torch.relu(dummy)
                dummy = dummy / 2.0
            time.sleep(0.2)  # Keep GPU load moderate

        # Sleep phase
        rest_duration = 20  # seconds
        time.sleep(rest_duration)

except KeyboardInterrupt:
    print("Stopped")

# Clean up
dummy = None
torch.cuda.empty_cache()
