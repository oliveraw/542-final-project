# Latent Coordinate Networks for Image and Video Memorization

We use a coordinate MLP with a learned latent code attached to the input in order to learn multiple images and videos with a single MLP.


|Ground Truth| No Pos. Enc. | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|---|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/gt.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|
|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/gt.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/none/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss1.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss10.0/videoMLP_Test_3000.gif" width="150">|<img src="https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/jelly/3000/gauss100.0/videoMLP_Test_3000.gif" width="150">|


For training, we use.... We use videos at half-resolution for the train set, and test on videos at full resolution. 

We observe artifacts indicating overfitting for $\sigma = 100$

## 2. Adding Latent Vectors for Input Images

## 3. Adding Latent Vectors for Input Videos

