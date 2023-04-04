# ExpMask
**ExpressionMask** or **ExpMask** in short is a deep learning project with the purpose of generating a newer face with the exact same expressions as your face although as you may have noticed, this is the first version of the AI meaning that the results may not be accurate and satisfactory enough but it is certainly going to be improved adequately in the later versions as expected.

## Data
Our data consisted of 88x108x3 images of various faces from the CelebA dataset which were then passed to mediapipe models to extract landmarks.

## Model
The general task of our deep learning model is to take your face and extract some landmarks from your face and then feed them to an AutoEncoder in order to give us an outcome with the same features as the original face.
### LandmarkPredictor
We contemplated on using two models for this task.The first one being a landmark predictor from which we took **mediapipe's** models 
### MaskGenerator
For the mask generator model we wrote a convolutional autoencoder with the following architecture
```python
Upsample2x2 -> ConvTranspose2d-512 -> ReLU -> ConvTranspose2d-256 -> ReLU -> Upsample2x2 -> ConvTranspose2d-256 -> ReLU -> ConvTranspose2d-128 -> ReLU -> ConvTranspose2d-3 -> Tanh
```
### Configs
We used Adam as the optimizer with the learning rate of 0.0001 and the batch size of 64 for 50 epochs and MSE as the loss function. 

## Steps
### Installation
Open up command line  and clone the repo through <br/>
`git clone https://github.com/Moeed1mdnzh/ExpMaskV1` <br/>
and then install the required packages by using
```python
pip install -r requirements.txt
```
Also to mention that our model runs with the help of pytorch which is not mentioned in the requirements file so you must consider installing pytorch and torchvision from [https://pytorch.org/](https://pytorch.org) in order to enjoy the AI.
### Running
The process of running the program is just as basic as cutting a butter.
```python 
python main.py -c 0 -m False -d 1
```
1. **-c or camera is the number of the camera out of the other cameras connected to your machine.**<br/>
2. **-m or mirror is a boolean determining wether to flip the frames or not.** <br/>
3. **-d or delay is determines the fps but it's not the fps itself.**
### Guide-To-Use
After you succeeded to run the file, you should then be facing a similar screen to this
![](https://github.com/Moeed1mdnzh/ExpMaskV1/blob/master/images/PageA.jpg) <br/>
The interaction with the app seems understandable enough except there's as well a slider on top of the screen which is not illustrated in this image which helps you to change the size of the green bounding box in the image.<br/><br/>
Now once you've pressed ***C*** the program takes the photo in the bounding box and is given to the landmark predictor and mask generator model to present us the following results <br/><br/>
![](https://github.com/Moeed1mdnzh/ExpMaskV1/blob/master/images/PageB.jpg)<br/>
Eventually you can save the results by pressing on ***S*** and the photos will be automatically saved as *Original.jpg*, *Landmarks.jpg* and *Mask.jpg* in the project folder.You could also return to the main screen by pressing ***R*** if you're not comfortable with your current taken image.
## Insights
The model is not capable of producing faces that have heavy similarities to the original faces due to the limited data but GANs(Generative Adversarial Networks) and more features about the face are going to be included in the second version of this AI.
