"""
Project code+scripts for 8BE030 course
"""

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg


def segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task='brain'):
    # segments the image based on your own method!
    # Input:
    # train_data_matrix   num_pixels x num_features x num_subjects matrix of
    # features
    # train_labels_matrix num_pixels x num_subjects matrix of labels
    # test_data           num_pixels x num_features test data
    # task           String corresponding to the segmentation task: either 'brain' or 'tissue'
    # Output:
    # predicted_labels    Predicted labels for the test slice

    #------------------------------------------------------------------#
    #TODO: Implement your method here

    print("train_data_matrix:", train_data_matrix.shape)
    print("train_labels_matrix:", train_labels_matrix.shape)    


    # combine train_data_matrix labels for data normalization
    train_data_reshaped, train_labels_reshaped = combine_subjects(train_data_matrix, train_labels_matrix)
    train_data_selection, train_labels_selection = balanced_pixel_subset(train_data_reshaped, train_labels_reshaped, max_per_class=1000)

    print("train_data_reshaped:", train_data_selection.shape)
    print("train_labels_reshaped:", train_labels_selection.shape)
    print("test_data:", test_data.shape)


    knn_segment = seg.segmentation_knn(train_data_selection, train_labels_selection, test_data, k=3)
    Nmeans_segment = seg.segmentation_nearest_centroid(train_data_selection, train_labels_selection, test_data)
    rmeans_segment = seg.radius_classifier(train_data_selection, train_labels_selection,test_data)

    print(rmeans_segment.shape)
    
    all_segments = np.stack([knn_segment,Nmeans_segment,rmeans_segment], axis = 1)
    print(all_segments.shape)

    values, counts = np.unique(all_segments, axis = 1, return_counts = True)
    predicted_labels = values[:, np.argmax(counts)]

    print(predicted_labels.shape)

    #------------------------------------------------------------------#
    return predicted_labels


def segmentation_demo():

    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'brain'

    #Load data
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

    # predicted_labels = seg.segmentation_knn(None, train_labels, None)

    # err = util.classification_error(test_labels, predicted_labels)
    # dice = util.dice_overlap(test_labels, predicted_labels)

    #Display results
    # true_mask = test_labels.reshape(240, 240)
    # predicted_mask = predicted_labels.reshape(240, 240)

    # fig = plt.figure(figsize=(8,8))
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(true_mask, 'gray')
    # ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    # print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    ## Compare methods
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    all_errors = np.empty([num_images,num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images,num_methods])
    all_dice[:] = np.nan

    all_subjects = np.arange(num_images)
    train_slice = 1
    task = 'brain'
    all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
    all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)

    #Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')
    

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.flatten()

    print('Finished loading data.\nStarting segmentation...')

    #Go through each subject, taking i-th subject as the test
    for i in np.arange(num_images):
        sub = i+1
        #Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        train_data_matrix = all_data_matrix[:,:,train_subjects]
        train_labels_matrix = all_labels_matrix[:,train_subjects]
        test_data = all_data_matrix[:,:,i]
        test_labels = all_labels_matrix[:,i]
        test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

        fig = plt.figure(figsize=(10,5))

        predicted_labels = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
        all_errors[i,0] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,0] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_1 = predicted_labels.reshape(im_size[0],im_size[1])
        ax1 = fig.add_subplot(121)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined k-NN'.format(sub))

        predicted_labels = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,task)
        all_errors[i,1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,1] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_2 = predicted_labels.reshape(im_size[0],im_size[1])
        ax2 = fig.add_subplot(122)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: My method'.format(sub))

def balanced_pixel_subset(X, y, max_per_class=600, random_state=0):
    """Return a class-balanced subset of pixels for faster training."""
    rng = np.random.default_rng(random_state)
    y = y.flatten().astype(int)
    selected = []

    for label in np.unique(y):
        label_idx = np.flatnonzero(y == label)
        n_keep = min(max_per_class, len(label_idx))
        selected.append(rng.choice(label_idx, size=n_keep, replace=False))

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return X[selected], y[selected]


def normalize_with_training_statistics(train_X, test_X):
    """Normalize train and test features using only the training distribution."""
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1
    return (train_X - mean) / std, (test_X - mean) / std


def candidate_feature_sets(feature_labels):
    """Feature subsets to compare during model selection."""
    n_features = len(feature_labels)
    sets = {
        "T1/T2 intensity": [0, 1],
        "intensity + smooth": [0, 1, 2, 3, 4, 5],
        "intensity + edges": [0, 1, 6, 7],
        "intensity + location": [0, 1, 8],
        "all features": list(range(n_features)),
    }

    if n_features > 9:
        sets["intensity + smooth + custom texture"] = [0, 1, 2, 3, 4, 5, 9]

    return sets


def predict_with_method(method_name, train_X, train_y, test_X):
    """Train one candidate segmentation method and predict test labels."""
    train_X_norm, test_X_norm = normalize_with_training_statistics(train_X, test_X)

    if method_name.startswith("kNN"):
        k = int(method_name.split("=")[1])
        clf = KNeighborsClassifier(n_neighbors=k)
    elif method_name == "Gaussian naive Bayes":
        clf = GaussianNB()
    else:
        raise ValueError(f"Unknown method: {method_name}")

    clf.fit(train_X_norm, train_y)
    return clf.predict(test_X_norm)


def evaluate_model_choices(train_subject, test_subject, slice_number=1, max_per_class=600, random_state=0):
    """Evaluate feature subsets and methods for two subjects.

    Returns
    -------
    best_choice : dict
        The best-ranked feature/method choice.
    results : list of dict
        All candidate results sorted from best to worst.
    """
    train_X, train_y, feature_labels = util.create_dataset(train_subject, slice_number, "tissue")
    test_X, test_y, _ = util.create_dataset(test_subject, slice_number, "tissue")

    train_y = train_y.flatten().astype(int)
    test_y = test_y.flatten().astype(int)
    feature_sets = candidate_feature_sets(feature_labels)
    methods = ["kNN k=1", "kNN k=3", "kNN k=7", "Gaussian naive Bayes"]

    results = []
    for feature_set_name, feature_indices in feature_sets.items():
        subset_train_X, subset_train_y = balanced_pixel_subset(
            train_X[:, feature_indices],
            train_y,
            max_per_class=max_per_class,
            random_state=random_state,
        )
        test_subset_X = test_X[:, feature_indices]

        for method in methods:
            predicted = predict_with_method(method, subset_train_X, subset_train_y, test_subset_X)
            error = util.classification_error(test_y, predicted)
            dice = util.dice_multiclass(test_y, predicted)
            results.append({
                "train_subject": train_subject,
                "test_subject": test_subject,
                "slice": slice_number,
                "features": feature_set_name,
                "method": method,
                "error": float(error),
                "dice": float(dice),
            })

    results = sorted(results, key=lambda r: (-r["dice"], r["error"]))
    return results[0], results


def print_top_results(results, n=8):
    print(f"{'rank':>4}  {'dice':>7}  {'error':>7}  {'method':<22}  features")
    print("-" * 80)
    for rank, row in enumerate(results[:n], start=1):
        print(f"{rank:>4}  {row['dice']:>7.4f}  {row['error']:>7.4f}  {row['method']:<22}  {row['features']}")


def combine_subjects(train_data_matrix, train_labels_matrix):
    """
    Combines subject dimension into a single dataset.

    Input:
        train_data_matrix: (num_pixels x num_features x num_subjects)
        train_labels_matrix: (num_pixels x num_subjects)

    Output:
        X: (num_pixels*num_subjects x num_features)
        y: (num_pixels*num_subjects,)
    """

    num_pixels, num_features, num_subjects = train_data_matrix.shape

    # reshape features: stack subjects vertically
    X = train_data_matrix.transpose(2, 0, 1).reshape(num_subjects * num_pixels, num_features)

    # flatten labels
    y = train_labels_matrix.T.reshape(num_subjects * num_pixels)

    return X, y