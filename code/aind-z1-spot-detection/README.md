# aind-z1-spot-detection

Large-scale detection of spot-like structures. The algorithms included in this package are:

**Traditional algorithm**: Method based on the Laplacian of Gaussians technique. These are the image processing steps that happen in every chunk:
1. Laplacian of Gaussians to enhance regions where the intensity changes dramatically (higher gradient).
2. Percentile to get estimated background image.
3. Combination of logical ANDs to filter the LoG image using threshold values and non-linear maximum filter.
4. After identifying initial spots (ZYX points) from 1-3 steps, we prune blobs close to each other within a certain radius $$r$$ using a kd-tree.
5. We take each of these pruned spots and get their contexts which is a 3D image of size $$context = (radius + 1, radius + 1, radius + 1)$$.
6. Finally, we fit a gaussian to each of the spots using its context to be able to prune false positives.

This is a traditional-based algorithm, therefore, parameter tunning is required to make it work.

The output of this algorithm is a CSV file with the following columns:

- Z: Z location of the spot.
- Y: Y location of the spot.
- X: X location of the spot.
- Z_center: Z center of the spot during the guassian fitting, useful for demixing.
- Y_center: Y center of the spot during the guassian fitting, useful for demixing.
- X_center: X center of the spot during the guassian fitting, useful for demixing.
- dist: Euclidean distance or L2 norm of the ZYX center vector, $`norm = \sqrt{z^2 + y^2 + x^2}`$.
- r: Pearson correlation coefficient between integrated 3D gaussian and the 3D context where the spot is located.
- SEG_ID (optional): When a segmentation mask is provided, a new column is added with the segmentation ID of the detected spot.

We also generate a numpy array with the values.

## Optional mask
An additional segmentation mask can be provided to only detect spots in given areas. In our case, the segmentation mask provided is a cell segmentation mask of the entire dataset, we used a  [large-scale version of cellpose](https://github.com/AllenNeuralDynamics/aind-z1-cell-segmentation) to generate the segmentation masks.
