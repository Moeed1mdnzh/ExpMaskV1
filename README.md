# ExpMask
**ExpressionMask** or **ExpMask** in short is a deep learning project with the purpose of generating a newer face with the exact same expressions as your face although as you may have noticed, this is the first version of the AI meaning that the results may not be accurate and satisfactory enough but it is certainly going to be improved adequately in the later versions as expected.

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
1. *c or camera is the number of the camera out of the other cameras connected to your machine.**<br/>
2-**-m or mirror is a boolean determining wether to flip the frames or not.** 
<br/>
3-**-d or delay is determines the fps but it's not the fps itself.**
