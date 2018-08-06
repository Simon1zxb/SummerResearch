from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import numpy as np

def extract_patches(volume, patch_shape, extraction_step):
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    print("===== patches.shape is", patches.shape, ndim, "==========")
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches,) + patch_shape)


def build_set(T1_vols, label_vols, extraction_step=(9, 9, 9)):
    patch_shape = (27, 27, 27)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 1, 27, 27, 27))
    y = np.zeros((0, 9 * 9 * 9, num_classes))
    for idx in range(len(T1_vols)):
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        # This is my revise
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) >= 0)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 1, 27, 27, 27))))
        y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes))))

        for i in range(len(label_patches)):
            y[i + y_length, :, :] = np_utils.to_categorical(label_patches[i].flatten(), num_classes)

        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
        del T1_train

        # Sampling strategy: reject samples which labels are only zeros
        # T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step)
        # x[y_length:, 1, :, :, :] = T2_train[valid_idxs]
        # del T2_train
    return x, y


# Reconstruction utils
import itertools


def generate_indexes(patch_shape, expected_shape):
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i + 1] * (expected_shape[i] // patch_shape[i + 1]) for i in range(ndims - 1)]

    #idxs = [range(patch_shape[i + 1], poss_shape[i] - patch_shape[i + 1], patch_shape[i + 1]) for i in range(ndims - 1)]
    #Todo: change the forlume here to a correct way
    idxs = [range(0, 261, 9), range(0, 261, 9), range(0, 134, 9)]

    return itertools.product(*idxs)


def reconstruct_volume(patches, expected_shape):
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)):
        selection = [slice(coord[i], coord[i] + patch_shape[i + 1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

def padding(test_img, expected_shape = (279, 279, 153)):
    new_image = np.zeros(expected_shape)
    new_image[9:265, 9:265, 9:137] = test_img
    return new_image