import torch


def detect():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s) available.")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA devices available.")
