import torch

if __name__ == '__main__':
    checkpoint = torch.load('./checkpoints/LoopLlmModel.pt', weights_only=False, map_location=torch.device('cpu'))
    print(checkpoint["step"])
