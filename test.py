import torch 
import numpy as np 
import cv2 

image = cv2.imread("face.jpg")
image = cv2.resize(image, (88, 108))
clone = image.copy()
state_dict = torch.load("models/LandmarkFinder.pt", map_location="cpu")
sample = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sample = np.interp(np.float32(sample), (0, 255), (-1, +1))
sample = torch.tensor(sample).permute(2, 0, 1).unsqueeze(dim=0)
model = torch.nn.Sequential(torch.nn.Conv2d(3, 512, (3, 3), padding=1),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d((2, 2), stride=2),
                            torch.nn.Conv2d(512, 256, (3, 3), padding=1),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
                            torch.nn.ReLU(),
                            torch.nn.MaxPool2d((2, 2), stride=2),
                            torch.nn.Flatten(),
                            torch.nn.Linear(256*27*22, 10))
model.load_state_dict(state_dict)
model.eval()
pred = model(sample.to(torch.float32))
pred = pred.detach().cpu().numpy()
pred = np.int32(pred)
pred = pred[0]
print(pred)
for i in range(0, 10, 2):
  pts = pred[i: i+2]
  cv2.circle(image, pts, 2, (0, 0, 255), -1)

cv2.imshow("", image)
cv2.imshow("a", clone)
cv2.waitKey(0)