"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output


def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread("../data/image_data/1_1_t1.tif")
    Im = plt.imread("../data/image_data/1_1_t1_d.tif")

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0.0, 0.0, 0.0])

    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)

    # the learning rate
    mu = 0.001

    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter + 1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14, 6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(
        0.3,
        0.95,
        np.array2string(x, precision=5, floatmode="fixed"),
        bbox={"facecolor": "white", "alpha": 1, "pad": 10},
        transform=ax1.transAxes,
    )

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    (learning_curve,) = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Similarity")
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g * mu

        # for visualization of the result
        S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)

        clear_output(wait=True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode="fixed"))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)


def optimize(fun, x, num_iter, mu, tol=1e-6):

    similarity = np.full((num_iter, 1), np.nan)

    S_old = fun(x)

    for k in range(num_iter):

        x_old = x.copy()

        g = reg.ngradient(fun, x)

        x += g * mu

        S_new = fun(x)

        similarity[k] = S_new

        print(f"Iteration {k+1}/{num_iter}, Similarity: {S_new:.4f}")

        # stopping criteria

        param_change = np.linalg.norm(x - x_old)

        sim_change = abs(S_new - S_old)

        if param_change < tol or sim_change < tol:

            print(f"Converged at iteration {k+1}")

            break

        S_old = S_new

    return x, similarity


def intensity_based_registration(I, Im, method="rigid_cc", num_iter=200, mu=0.001):
    match method:
        case "rigid_cc":
            fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)
            x = np.array([0.0] * 3)
            y_lim = (0, 1)
            SCALING = 100
        case "affine_cc":
            fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
            x = np.array([0.0] * 7)
            y_lim = (0, 1)
            SCALING = 100
        case "affine_mi":
            fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)
            x = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            y_lim = (0, 1)
            SCALING = 100
        case _:
            print("invalid method chosen")
            return

    x, S = optimize(fun, x, num_iter, mu)
    print(f"final similarity: {S[-1][0]:.4f}")

    if len(x) == 3:
        T = reg.rotate(x[0])
        Th = util.t2h(T, x[1:] * SCALING)
    else:
        T = reg.rotate(x[0]).dot(reg.scale(x[1], x[2])).dot(reg.shear(x[3], x[4]))
        Th = util.t2h(T, x[5:] * SCALING)

    Im_t, _ = reg.image_transform(Im, Th)

    ### displaying the graphs

    fig = plt.figure(figsize=(14, 6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(Im_t, alpha=0.7)
    # parameters
    txt = ax1.text(
        0.3,
        0.95,
        np.array2string(x, precision=5, floatmode="fixed"),
        bbox={"facecolor": "white", "alpha": 1, "pad": 10},
        transform=ax1.transAxes,
    )

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=y_lim)

    iterations = np.arange(1, num_iter + 1)

    ax2.plot(iterations, S, lw=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Similarity")
    ax2.grid()


def generate_noisy_images():
    noise_levels = [0, 1, 2]
    layers = [1, 2, 3]
    patients = [1, 2, 3, 4, 5]
    scan_types = ["t1", "t2"]

    for patient in patients:
        for layer in layers:
            for scan_type in scan_types:
                original_image_name = (
                    f"../data/dataset_brains/{patient}_{layer}_{scan_type}.tif"
                )
                original_image = plt.imread(original_image_name)
                for noise_level in noise_levels:
                    noise = np.random.normal(0, 10 * noise_level, original_image.shape)
                    img_noised = np.clip(original_image + noise, 0, 255).astype(
                        np.uint8
                    )
                    if scan_type == "t2":
                        T = util.t2h(reg.identity(), np.array((120, 120)))
                        T = T.dot(util.t2h(reg.rotate(1 / 6 * np.pi), np.array([0, 0])))
                        T = T.dot(util.t2h(reg.identity(), np.array((-120, -120))))
                        img_noised, _ = reg.image_transform(img_noised, T)
                    plt.imsave(
                        f"../data/noisy_image_data/{patient}_{layer}_{scan_type}_{noise_level}.png",
                        img_noised,
                        cmap="gray",
                    )
                    print(f"{patient}_{layer}_{scan_type}_{noise_level}.png is saved")

    return
