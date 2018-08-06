import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

from SummerResearch.utils.imageOp import *
from SummerResearch.utils.IO import *
from SummerResearch.net.NetModule import *

num_classes = 2

model_filename = 'models/iSeg2017/outrun_step_{}.h5'
csv_filename = 'log/iSeg2017/outrun_step_{}.cvs'
extraction_step = (9, 9, 9)

print("============= LOAD MODEL =============")
model = generate_model(num_classes)
model.load_weights(model_filename.format(1))
print("============= LOAD MODEL DONE =============")

# read Data
print("============= READ DATA =============")

T1_vols = np.zeros((16, 279, 279, 153))

for case_id in range(0, 16) :
    T1_vols[case_id, 9:265, 9:265, 9:137] = read_vol(case_id + 1, 'data')

print("============= READ DATA DONE =============")

# process Data

print("============= PRE-PROCESS DATA =============")
T1_mean = T1_vols.mean()
T1_std = T1_vols.std()

print("============= PRE-PROCESS DATA DONE =============")
T1_vols = np.zeros((279, 279, 153))

for case_idx in range(16, 17):
    start = time.time()
    T1_vols[9:265, 9:265, 9:137] = read_vol(case_idx, 'data')[:256, :256, :128]
    # T2_test_vol = read_vol(case_idx, 'T2')[:256, :256, :128]
    T1_vols = (T1_vols - T1_mean) / T1_std
    x_test = np.zeros((12615, 1, 27, 27, 27))
    x_test[:, 0, :, :, :] = extract_patches(T1_vols, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))

    pred = model.predict(x_test, verbose=2)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 9))
    segmentation = reconstruct_volume(pred_classes, (256, 256, 128))

    csf = np.logical_and(segmentation == 0, T1_vols != 0)

    segmentation[segmentation == 1] = 150
    segmentation[csf] = 10

    save_vol(segmentation, case_idx)
    end = time.time()
    print("Finished segmentation of case # {}".format(case_idx), "Cost time is : ", end - start)

print("Test done")


