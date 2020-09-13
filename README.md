# Image to Image Translation with Pix2Pix GAN
Paper To Code implementation of pix2pix GAN in native pytorch

Original Paper- [ArxViv](https://arxiv.org/abs/1611.07004)

Code packaging in progress... Will be updated soon....

Pls checkout the [medium article](https://medium.com/@noufalsamsudin/image-to-image-translations-for-colorizing-videos-image-restoration-and-face-generation-14b7d7a40b34?sk=a73efb61df685f717459ab89e8f2a0ba) for a quick overview.

### Summary

Train Pix2Pix GAN models for:
1. Colorizing and enhancing an old, black & white grainy footage
2. Inpainting images to fix scratches, missing pixels etc.
3. Generating faces from doodles


### Usecases

#### 1. Video restoration of Malcolm X interview

 I download 2 videos ([Pursuit of Happiness last scenes](https://www.youtube.com/watch?v=x8-7mHT9edg&ab_channel=CiprianVatamanu), [Funniest moments in talkshow](https://www.youtube.com/watch?v=hO5Fp9ZLFqE&ab_channel=ComedySpace)), split the video into frames, and generate a grainy b/w image using some basic PIL functions. So I now have my image pairs for my training data:
 
 ![Gif of train videos](https://github.com/kvsnoufal/pix2pix/blob/master/docs/malcolm_train_gif.gif)
 
 ##### Results:

![Gif of results](https://github.com/kvsnoufal/pix2pix/blob/master/docs/malcolm_result_gif.gif)


Checkout youtube for video with audio - recommended:
[Youtube Link](https://youtu.be/Fhb1uHk80XQ)

#### 2.Inpainting images to fix scratches, missing pixels etc.

For training, I used VOC2012 dataset, and generated a “distorted version” of each image by randomly cropping pixels and adding black lines and blobs. Tested it on a held out portion of the dataset.

![Pic of results](https://github.com/kvsnoufal/pix2pix/blob/master/docs/inpainting_results.jpeg)


#### 3.Generating faces from doodles

For training, I generated a “doodle” for each face in the 10k faces dataset using a combination of face-landmark detection feature in opencv and Holy Nested Edge detection.

For testing, I set up webcam to read hand-drawn doodles on post-it notes.

##### Results:

![Gif of doodle results](https://github.com/kvsnoufal/pix2pix/blob/master/docs/doodleface_gif.gif)


Check out youtube video :
[Youtube Link](https://youtu.be/92gwRyJS8m8)







