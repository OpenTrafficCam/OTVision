import torch

if __name__ == "__main__":
    print(f"This system has cuda: {torch.cuda.is_available()}")
