import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from SummerResearch.utils.imageOp import *
from SummerResearch.utils.IO import *
from SummerResearch.net.NetModule import *

num_classes = 2
num_epoch = 20

patience = 10
model_filename = 'models/iSeg2017/outrun_step_{}.h5'
csv_filename = 'log/iSeg2017/outrun_step_{}.cvs'
extraction_step = (3, 3, 3)

validation_split = 0.2

# read Data
print("============= READ DATA =============")
T1_vols = np.empty((16, 279, 279, 153))
label_vols = np.empty((16, 279, 279, 153))

for case_id in range(0, 16) :
    T1_vols[case_id, 9:265, 9:265, 9:137] = read_vol(case_id + 1, 'data')
    label_vols[case_id, 9:265, 9:265, 9:137] = read_vol(case_id + 1, 'label')

print("============= READ DATA DONE =============")

# process Data

print("============= PRE-PROCESS DATA =============")
T1_mean = T1_vols.mean()
T1_std = T1_vols.std()
T1_vols = (T1_vols - T1_mean) / T1_std

print("============= PRE-PROCESS DATA DONE =============")

# build Dataset
print("============= PREPARE DATASET FOR TRAINING ============")

x_train, y_train = build_set(T1_vols, label_vols, extraction_step)

print("============= PREPARE DATASET DONE ============")

print("============= CONFIGURE CALLBACKS ============")

stopper = EarlyStopping(patience)

reducer = ReduceLROnPlateau(montitor='loss',
                            factor=0.1,
                            patience=5,
                            verbose=1,
                            mode='auto')

check_pointer = ModelCheckpoint(filepath=model_filename.format(1),
                               verbose=0,
                               save_best_only=True,
                               save_weights_only=True)

csv_logger = CSVLogger(csv_filename, separator=';')

callbacks = [stopper, reducer, check_pointer, csv_logger]

print("============= CONFIGURE CALLBACKS DONE ============")

print("============= START TRAINING ============")

model = generate_model(num_classes)

model.fit(x_train,
          y_train,
          batch_size=20,
          epochs=num_epoch,
          validation_split=validation_split,
          verbose=2,
          callbacks=callbacks)

print("============= FIRST TRAIN DONE ============")
del x_train
del y_train





