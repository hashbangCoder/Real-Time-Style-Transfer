# Real Time Style Transfer 

This is the Torch Implementation for [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf) paper. 

### Results

Preliminary working example and output has been uploaded. I'll update soon with better ones.

Trained on 256x256 resized images

Style Image - Content Image - Output (256x256) - Output (512x512)

<img src="https://github.com/hashbangCoder/Real-Time-Style-Transfer/blob/master/style_image.jpg?raw=true" width="300" height="300">
<img src="https://github.com/hashbangCoder/Real-Time-Style-Transfer/blob/master/test_image.jpg?raw=true" width="300" height="300">
<img src="https://github.com/hashbangCoder/Real-Time-Style-Transfer/blob/master/Output_1/testOutIterend.jpg?raw=true" width="300" height="300">
<img src= width="300" height="300">


### What has been implemented
- Only Style Transfer has been implemented so far
- Updated code to reflect changes in the paper for removing border artifacts
- Video version coming soon. Check out [this](https://www.youtube.com/watch?v=h0jH0bJIvcM&feature=youtu.be) example in Chainer



### Requirements

Unfortunately it is not fully CPU compatible yet and requires a GPU to run

- Torch Packages - Image,XLua,Cutorch,Cunn,Optim,

- cuDNN/CUDA


### Details
Trained on MS COCO TrainSet (~80,000 images) over two epochs on a NVIDIA TitanX gpu. Takes about ~6 hours







