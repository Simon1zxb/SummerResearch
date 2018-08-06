import nibabel as nib
import numpy as np

seed = 7
np.random.seed(seed)

def get_filename(folder_name, case_idx, input_name, loc='datasets'):
    if input_name == "data":
        pattern = '{0}/IBSR/IBSR-{1}/{2}/IBSR_{3}_ana.nii.gz'
    else:
        pattern = '{0}/IBSR/IBSR-{1}/{2}/IBSR_{3}_ana_brainmask.nii.gz'
    return pattern.format(loc, folder_name, input_name, case_idx)


def get_folder_name(case_idx):
    return 'Training' if case_idx < 17 else 'Testing'


def read_data(case_idx, input_name, loc='datasets'):
    folder_name = get_folder_name(case_idx)

    image_path = get_filename(folder_name, case_idx, input_name, loc)
    print("-----image_path is", image_path, "-----")
    data = nib.load(image_path)

    return data


def read_vol(case_idx, input_name, loc='datasets'):
    image_data = read_data(case_idx, input_name, loc)
    image_data = image_data.get_data()[:, :, :, 0]
    print("data.shape is:", image_data.shape)
    if input_name == "label":
        image_data = image_data.transpose((0, 2, 1))
    return image_data


def save_vol(segmentation, case_idx, loc='results'):
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, 'data')

    segmentation_vol = np.empty(input_image_data.shape)
    segmentation_vol[:256, :256, :128, 0] = segmentation

    filename = get_filename(set_name, case_idx, 'label', loc)
    print("filename is", filename)
    nib.save(nib.analyze.AnalyzeImage(
        segmentation_vol.astype('uint8'), input_image_data.affine), filename)