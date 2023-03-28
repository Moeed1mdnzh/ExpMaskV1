import torch 
import numpy as np 
import cv2 

sample = np.array([35, 56, 54, 56, 44, 65, 36, 76, 52, 77])
state_dict = torch.load("models/MaskGenerator.pt", map_location="cpu")
class Reshape(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), *self.size)

model = torch.nn.Sequential(
    torch.nn.Linear(10, 512*27*22),
    Reshape((512, 27, 22)),
    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
    torch.nn.ConvTranspose2d(512, 512, (3, 3), padding=1),
    torch.nn.SELU(),
    torch.nn.ConvTranspose2d(512, 256, (3, 3), padding=1),
    torch.nn.SELU(),
    torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
    torch.nn.ConvTranspose2d(256, 256, (3, 3), padding=1),
    torch.nn.SELU(),
    torch.nn.ConvTranspose2d(256, 128, (3, 3), padding=1),
    torch.nn.SELU(),
    torch.nn.ConvTranspose2d(128, 3, (1, 1)),
    torch.nn.Tanh()
)
model.load_state_dict(state_dict)
model.eval()
sample = torch.tensor(sample).reshape(1, 10).to(torch.float32)
pred = model(sample).squeeze()
pred = (pred + 1.0)/2.0
pred = pred.permute(1, 2, 0).detach().cpu().numpy()
pred = pred * 255
pred = cv2.cvtColor(np.uint8(pred), cv2.COLOR_RGB2BGR)
cv2.imshow("WIN", pred)
cv2.waitKey(0)
