# Latent Coordinate Networks for Image and Video Memorization

We use a coordinate MLP with a learned latent code attached to the input in order to learn multiple images and videos with a single network. Having first validated that such an MLP is able to learn many images at once, we turn to learning multiple videos. We show only our video results below:

### Using a Coordinate MLP to Remember a Single Video
First, we validate that a coordinate MLP is able to learn an implicit representation for video. We use the same positional encoding scheme for images and video, only varying the smaller dimension of the \mathbf{B} matrix between 2 and 3. 

|Ground Truth| No Pos. Enc. | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|---|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/gt.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/gt.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|

### Using a Coordinate MLP to Remember Pairs of Videos
We then attempt to learn pairs of videos. We notice that the resulting video quality is significantly decreased from the single-video training. Due to GPU constraints, we decrease the resolution of the two videos by a factor of 2. Results are shown below:

|Ground Truth| No Pos. Enc. | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|---|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/gt.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/gt.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/none/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss1.0/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss10.0/videos/jelly.gif" width="150">|<img src="[https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss100.0/videoMLP_Test_3000.gif](https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss100.0/videos/jelly.gif)" width="150">|

## 2. Adding Latent Vectors for Input Images

## 3. Adding Latent Vectors for Input Videos

