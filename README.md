# aind-exaspim-soma-detection

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## Overview

To do...

<p>
  <img src="imgs/pipeline.png" width="900" alt="pipeline">
  <br>
  <b> Figure: </b>Visualization of soma detection pipeline.
</p>

Add code block that shows how to run code...

## Inference

### Step 1: Proposal Generation

The objective of this step is to generate initial proposals for potential soma locations by detecting blob-like structures in the image. Our proposal detection algorithm includes the following steps:

<blockquote>
  <p>a. Smooth image with Gaussian filter to reduce false positives.</p>
  <p>b. Laplacian of Gaussian (LoG) to enhance regions where the gradient changes dramatically, then apply a maximum filter.</p>
  <p>c. Generate initial set of proposals by detecting local maximas in LoG image that lie outside of the image margins.</p>
  <p>d. Shift each proposal to the brightest voxel in its neighborhood. If the brightness is below a threshold, reject the proposal.</p>
</blockquote>

<p>
  <img src="imgs/proposals_example.png" width="900" alt="proposals">
  <br>
  <b> Figure: </b>Example of proposals generated across a large region.
</p>

### Step 2: Filter Proposals with Heuristics

To do...

### Step 3: Classify Proposals with Convolutional Neural Network (CNN)

To do...

## Train Classification Model

To do...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```
