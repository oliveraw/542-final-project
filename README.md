# Latent Coordinate Networks for Image and Video Memorization

We use a coordinate MLP with a learned latent code attached to the input in order to learn multiple images and videos with a single network. We use selected videos from the [WAIC-TSR](https://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/supplementary/Dataset.html) dataset, showing only our video results below:

### Using a Coordinate MLP to Remember Multiple Videos
We use the same positional encoding scheme as the 2d image MLP, $\gamma(\mathbf{x}) = \[\sin(2\pi\mathbf{Bx}), \cos(2\pi\mathbf{Bx})\]^T$, only changing the smaller dimension of the $\mathbf{B}$ matrix from 2 to 3. We vary the hyperparameter $\sigma$, the standard deviation of elements in the positional encoding matrix $\mathbf{B}$. Notice the blurriness of the video generated without positional encoding, as well as the "static" texture of the $\sigma = 100$ positional encoding.

**🔴 Note: if you are on mobile it may be helpful to zoom in on the videos. 🔴**

#### Results After Learning 2 Videos
| Ground Truth | No Pos. Enc. | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|---|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/none/videos/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss1.0/videos/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss10.0/videos/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss100.0/videos/water.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/none/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss1.0/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss10.0/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/9000/gauss100.0/videos/jelly.gif" width="150">|

#### Results After Learning 4 Videos
| Ground Truth | No Pos. Enc. | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|---|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/none/videos/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss1.0/videos/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss10.0/videos/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss100.0/videos/water.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/none/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss1.0/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss10.0/videos/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss100.0/videos/jelly.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/billiards.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/none/videos/billiard.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss1.0/videos/billiard.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss10.0/videos/billiard.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss100.0/videos/billiard.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/running_women.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/none/videos/running_women.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss1.0/videos/running_women.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss10.0/videos/running_women.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/9000/gauss100.0/videos/running_women.gif" width="150">|

#### Interpolation Between Latent Codes After Learning 2 Videos
Similar to our experiments for images, we also interpolate between latent codes and show the results below:
|| 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
|No Pos. Enc.|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/none/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/none/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/none/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/none/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/none/1.0.gif" width="150">|
|$\sigma = 1$|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss1.0/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss1.0/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss1.0/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss1.0/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss1.0/1.0.gif" width="150">|
|$\sigma = 10$|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss10.0/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss10.0/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss10.0/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss10.0/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss10.0/1.0.gif" width="150">|
|$\sigma = 100$|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss100.0/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss100.0/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss100.0/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss100.0/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water-jelly-9000iters/interpolations/gauss100.0/1.0.gif" width="150">|

#### Interpolation Between Latent Codes After Learning 4 Videos
Curiously, we notice that the interpolations after learning 4 videos are higher quality, having more faithful color and shape reconstruction than the above. 
|| 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
|No Pos. Enc.|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/none/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/none/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/none/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/none/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/none/1.0.gif" width="150">|
|$\sigma = 1$|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss1.0/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss1.0/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss1.0/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss1.0/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss1.0/1.0.gif" width="150">|
|$\sigma = 10$|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss10.0/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss10.0/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss10.0/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss10.0/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss10.0/1.0.gif" width="150">|
|$\sigma = 100$|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss100.0/0.0.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss100.0/0.25.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss100.0/0.5.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss100.0/0.75.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/4vids-9000iters/interpolations/gauss100.0/1.0.gif" width="150">|

#### Results After Learning a Single Video
To compare against the multi-video scenario, we also train MLPs which learn only a single video at a time. 

| Ground Truth | No Pos. Enc. | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|---|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/water.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/gt/jelly.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|
