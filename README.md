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
 
 ![Gif of train videos](https://github.com/kvsnoufal/pix2pix/docs/malcolm_train_gif.gif)



