"""
Utility functions for segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import segmentation as seg
from scipy import ndimage
from pathlib import Path


def _dataset_base_dir():
    # Use a path relative to this file, so the code works from the notebook
    # as well as from scripts started in the project root.
    return Path(__file__).resolve().parents[1] / 'data' / 'dataset_brains'


def available_brain_slices():
    # Find all subject/slice combinations that have a ground-truth image.
    # The file names are formatted like 1_1_gt.tif.
    base_dir = _dataset_base_dir()
    slices = []

    for gt_file in sorted(base_dir.glob('*_*_gt.tif')):
        image_number, slice_number, *_ = gt_file.stem.split('_')
        slices.append((int(image_number), int(slice_number)))

    return slices


def create_all_datasets(task='tissue'):
    # Create one dataset per available subject/slice pair. This keeps the
    # images separated, which is useful later when testing generalization.
    datasets = {}

    for image_number, slice_number in available_brain_slices():
        datasets[(image_number, slice_number)] = create_dataset(image_number, slice_number, task)

    return datasets


def create_training_dataset(image_numbers, slice_numbers, task='tissue'):
    # Create one training dataset by combining multiple subject/slice datasets.
    # For example, image_numbers=[1, 2] and slice_numbers=[1, 2, 3] creates one
    # dataset containing all pixels from those six images.
    image_numbers = np.atleast_1d(image_numbers)
    slice_numbers = np.atleast_1d(slice_numbers)

    all_features = []
    all_labels = []
    feature_labels = None

    for image_number in image_numbers:
        for slice_number in slice_numbers:
            X, Y, current_feature_labels = create_dataset(int(image_number), int(slice_number), task)
            all_features.append(X)
            all_labels.append(Y)

            if feature_labels is None:
                feature_labels = current_feature_labels

    X_train = np.concatenate(all_features, axis=0)
    Y_train = np.concatenate(all_labels, axis=0)

    return X_train, Y_train, feature_labels


def _as_float_image(im):
    # Convert the image to float so filtering operations behave consistently.
    return im.astype(float)


def _feature_column(feature_image):
    # Store every image feature as one column in the pixel-wise feature matrix.
    return feature_image.flatten().T.reshape(-1, 1)


def extract_local_std_feature(im, sigma=2):
    # This feature measures local intensity variation around each pixel.
    # I use it as a simple texture feature: homogeneous tissue regions should
    # have lower values, while boundaries and locally varying regions should
    # have higher values. It only uses the image itself, not the ground truth.
    im = _as_float_image(im)
    local_mean = ndimage.gaussian_filter(im, sigma=sigma)
    local_mean_squared = ndimage.gaussian_filter(im**2, sigma=sigma)
    local_variance = local_mean_squared - local_mean**2
    local_variance = np.maximum(local_variance, 0)
    local_std = np.sqrt(local_variance)

    return _feature_column(local_std)


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    #------------------------------------------------------------------#
    # TODO: Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter
    pass
    #------------------------------------------------------------------#

    # return g

def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    # scater_data displays a scatterplot of at most 1000 samples from dataset X, and gives each point
    # a different color based on its label in Y

    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx,:]
        Y = Y[idx]

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'X, class '+str(i)
        ax.scatter(X[idx2,feature0], X[idx2,feature1], color=c, label=lbl)

    return ax


def create_dataset(image_number, slice_number, task):
    # create_dataset Creates a dataset for a particular subject (image), slice and task
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    # Output:
    # X           - Nxk feature matrix, where N is the number of pixels and k is the number of features
    # Y           - Nx1 vector with labels
    # feature_labels - kx1 cell array with descriptions of the k features

    #Extract features from the subject/slice
    X, feature_labels = extract_features(image_number, slice_number)

    #Create labels
    Y = create_labels(image_number, slice_number, task)

    return X, Y, feature_labels


def extract_features(image_number, slice_number):
    # extracts features for [image_number]_[slice_number]_t1.tif and [image_number]_[slice_number]_t2.tif
    # Input:
    # image_number - Which subject (scalar)
    # slice_number - Which slice (scalar)
    # Output:
    # X           - N x k dataset, where N is the number of pixels and k is the total number of features
    # features    - k x 1 cell array describing each of the k features

    base_dir = _dataset_base_dir()

    t1 = plt.imread(base_dir / f'{image_number}_{slice_number}_t1.tif')
    t2 = plt.imread(base_dir / f'{image_number}_{slice_number}_t2.tif')

    features = ()

    t1 = _as_float_image(t1)
    t2 = _as_float_image(t2)

    t1f = _feature_column(t1)
    t2f = _feature_column(t2)

    X = np.concatenate((t1f, t2f), axis=1)

    features += ('T1 intensity',)
    features += ('T2 intensity',)

    #------------------------------------------------------------------#
    # Smooth intensity features add local context around each pixel. This can
    # help because tissue labels are usually spatially coherent, not random.
    t1_smooth_1 = ndimage.gaussian_filter(t1, sigma=1)
    t1_smooth_3 = ndimage.gaussian_filter(t1, sigma=3)
    t2_smooth_1 = ndimage.gaussian_filter(t2, sigma=1)
    t2_smooth_3 = ndimage.gaussian_filter(t2, sigma=3)

    # Gradient magnitude is an edge-strength feature. It is useful near tissue
    # boundaries, where intensities change quickly between neighboring pixels.
    t1_gradient = ndimage.gaussian_gradient_magnitude(t1, sigma=1)
    t2_gradient = ndimage.gaussian_gradient_magnitude(t2, sigma=1)

    # The coordinate feature encodes distance from the image center. It gives
    # the classifier simple spatial information without using ground truth.
    coordinate_feature, _ = seg.extract_coordinate_feature(t1)

    # Local standard deviation is a texture feature. It can highlight areas
    # where the local intensity pattern is less homogeneous.
    t1_local_std = extract_local_std_feature(t1, sigma=2)

    extra_features = (
        _feature_column(t1_smooth_1),
        _feature_column(t1_smooth_3),
        _feature_column(t2_smooth_1),
        _feature_column(t2_smooth_3),
        _feature_column(t1_gradient),
        _feature_column(t2_gradient),
        coordinate_feature,
        t1_local_std,
    )

    X = np.concatenate((X,) + extra_features, axis=1)

    features += ('T1 Gaussian sigma 1',)
    features += ('T1 Gaussian sigma 3',)
    features += ('T2 Gaussian sigma 1',)
    features += ('T2 Gaussian sigma 3',)
    features += ('T1 gradient magnitude',)
    features += ('T2 gradient magnitude',)
    features += ('Distance to image center',)
    features += ('T1 local standard deviation',)
    #------------------------------------------------------------------#
    return X, features


def create_labels(image_number, slice_number, task):
    # Creates labels for a particular subject (image), slice and
    # task
    #
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    #
    # Output:
    # Y           - Nx1 vector with labels
    #
    # Original labels reference:
    # 0 background
    # 1 cerebellum
    # 2 white matter hyperintensities/lesions
    # 3 basal ganglia and thalami
    # 4 ventricles
    # 5 white matter
    # 6 brainstem
    # 7 cortical grey matter
    # 8 cerebrospinal fluid in the extracerebral space

    #Read the ground-truth image
    base_dir = _dataset_base_dir()

    I = plt.imread(base_dir / f'{image_number}_{slice_number}_gt.tif')

    if task == 'brain':
        Y = I>0
    elif task == 'tissue':
        # Convert the original MRBrainS labels to the tissue classes used in
        # this project: background/other, white matter, gray matter, and CSF.
        white_matter = np.isin(I, [2, 5])
        gray_matter = np.isin(I, [3, 7])
        csf = np.isin(I, [4, 8])
        background = np.isin(I, [0, 1, 6])

        # New labels:
        # 0 = background/other, 1 = white matter, 2 = gray matter, 3 = CSF.
        Y = np.copy(I)
        Y[background] = 0
        Y[white_matter] = 1
        Y[gray_matter] = 2
        Y[csf] = 3
    else:
        print(task)
        raise ValueError("Variable 'task' must be one of two values: 'brain' or 'tissue'")

    Y = Y.flatten().T
    Y = Y.reshape(-1,1)

    return Y


def dice_overlap(true_labels, predicted_labels, smooth=1.):
    # returns the Dice coefficient for two binary label vectors
    # Input:
    # true_labels         Nx1 binary vector with the true labels
    # predicted_labels    Nx1 binary vector with the predicted labels
    # smooth              smoothing factor that prevents division by zero
    # Output:
    # dice          Dice coefficient

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    #------------------------------------------------------------------#
    intersection = np.sum(t * p)
    dice = (2 * intersection + smooth) / (np.sum(t) + np.sum(p) + smooth)
    #------------------------------------------------------------------#
    return dice


def dice_multiclass(true_labels, predicted_labels):
    #dice_multiclass.m returns the Dice coefficient for two label vectors with
    #multiple classses
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # dice_score          Dice coefficient

    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)

    dice_score = np.empty((len(all_classes), 1))
    dice_score[:] = np.nan

    #Consider each class as the foreground class
    for i in np.arange(len(all_classes)):
        idx2 = indices2 == all_classes[i]
        lbl = 'X, class '+ str(all_classes[i])
        temp_true = true_labels.copy()
        temp_true[true_labels == all_classes[i]] = 1  #Class i is foreground
        temp_true[true_labels != all_classes[i]] = 0  #Everything else is background

        temp_predicted = predicted_labels.copy();
        temp_predicted[predicted_labels == all_classes[i]] = 1
        temp_predicted[predicted_labels != all_classes[i]] = 0
        dice_score[i] = dice_overlap(temp_true.astype(int), temp_predicted.astype(int))

    dice_score_mean = dice_score.mean()

    return dice_score_mean


def classification_error(true_labels, predicted_labels):
    # classification_error.m returns the classification error for two vectors
    # with labels
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # error         Classification error

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    #------------------------------------------------------------------#
    err = np.mean(t != p)
    #------------------------------------------------------------------#
    return err
