# Latent Coordinate Networks for Images and Video

This project investigates the use of latent noise vectors for remembering image and video information. There are multiple stages to the project.

## 1. Video Data
First, we train a videoMLP to validate that a coordinate network can remember video data. We use the random positional encoding described by Tancik et al. to embed our video data as follows. 

Formally, we would like to learn a function $f: \mathbb{R}^3 \rightarrow \mathbb{R}^3$, where our input vector $\mathbf{x}$ is of the form $(x, y, t)$, and the output vector represents an RGB color, $(r, g, b)$. 

We apply a positional encoding, $\mathbf{B}$, which is a Gaussian random noise matrix, with varying standard deviations, following the approach of (Tancik). We use standard deviations of $\sigma = \{1, 10, 100\}$ for the positional encoding. Our results are shown below:

| No Positional Encoding | $\sigma = 1$ | $\sigma = 10$ | $\sigma = 100$ |
|---|---|---|---|
|![](https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/none/videoMLP_Test_3000.gif)|![](https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss1.0/videoMLP_Test_3000.gif)|![](https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss10.0/videoMLP_Test_3000.gif)|![](https://github.com/oliveraw/542-final-project/blob/master/results/videoMLP/water/3000/gauss100.0/videoMLP_Test_3000.gif)|



For training, we use.... We use videos at half-resolution for the train set, and test on videos at full resolution. 

We observe artifacts indicating overfitting for $\sigma = 100$

## 2. Adding Latent Vectors for Input Images

## 3. Adding Latent Vectors for Input Videos

