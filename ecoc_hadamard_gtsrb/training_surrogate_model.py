from utils.train_support import DataSequence, visualize_training_history
from utils.readTrafficSigns import Hadamard_labels
import numpy as np
import albumentations as aug
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
import os
from ecoc_hadamard_gtsrb.models import get_32class_surrogate_model

if __name__ == '__main__':
    # paras need pay attention every time
    BIT_MODEL_INDEX = 0  # starts from 1 to 32 other number indicates 32-class surrogate case
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32
    TRAIN_IMG_FOLDER = r'D:\LableCodingNetwork\database\GTSRB\Final_Training\Images'
    TEST_IMG_FOLDER = r'D:\LableCodingNetwork\database\GTSRB\Final_Test\Images'
    EPOCHS = 20

    # paras needs change every time
    ADAM_DECAY = 0.9
    RESULTSAVEPATH = r'D:\LableCodingNetwork\trained-models\GTSRB\ECOC\VGG16-32class_surrogate'
    if not os.path.exists(RESULTSAVEPATH):
        os.makedirs(RESULTSAVEPATH)
    TOP32GTSRB_CATEGORIES = np.load(r'D:\LableCodingNetwork\database\GTSRB\gysrb_top32category_label.npy')
    HADAMARD_MATRIX = np.load(r'D:\LableCodingNetwork\ecoc_hadamard_gtsrb\hadamard32.npy')


    # dataset prepartion
    train_val_img_paths, train_val_bit_model_labels = Hadamard_labels(TRAIN_IMG_FOLDER, TOP32GTSRB_CATEGORIES,
                                                                      HADAMARD_MATRIX, BIT_MODEL_INDEX)
    assert len(train_val_img_paths) == len(train_val_bit_model_labels)
    l = len(train_val_img_paths)
    np.random.seed(1)
    np.random.shuffle(train_val_img_paths)
    np.random.seed(1)
    np.random.shuffle(train_val_bit_model_labels)
    train_img_paths, val_img_paths = train_val_img_paths[:int(l * 0.8)], train_val_img_paths[int(l * 0.8):]
    train_bit_model_labels, val_bit_model_labels = \
        train_val_bit_model_labels[:int(l * 0.8)], train_val_bit_model_labels[int(l * 0.8):]

    test_img_paths, test_bit_model_labels = Hadamard_labels(TEST_IMG_FOLDER, TOP32GTSRB_CATEGORIES, HADAMARD_MATRIX,
                                                            BIT_MODEL_INDEX)
    # data augmentation setting
    trainAugSetting = aug.Compose([aug.HorizontalFlip(),
                                   # aug.ShiftScaleRotate(shift_limit=0.05,rotate_limit=5,p=0.25),
                                   # aug.MedianBlur(blur_limit=5,p=0.25),
                                   # aug.GaussianBlur(p=0.25),
                                   # aug.GaussNoise(p=0.25),
                                   # aug.RandomBrightnessContrast(p=0.25),
                                   # aug.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.25),
                                   # aug.RGBShift(r_shift_limit=5,g_shift_limit=5,b_shift_limit=5,p=0.25),
                                   ])  # ctrl + b to parameter details

    train_sequence = DataSequence(train_img_paths,
                                  to_categorical(train_bit_model_labels),  # does it need to be one-hot?
                                  BATCH_SIZE,
                                  augmentations=trainAugSetting)
    val_sequence = DataSequence(val_img_paths,
                                to_categorical(val_bit_model_labels),
                                BATCH_SIZE)
    test_sequence = DataSequence(test_img_paths,
                                 to_categorical(test_bit_model_labels),
                                 BATCH_SIZE)

    # get model and training setting
    model = get_32class_surrogate_model()
    optimizer = Adam(lr=LEARNING_RATE, decay=ADAM_DECAY)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None,
                               restore_best_weights=True)
    tensorbord = TensorBoard(RESULTSAVEPATH)
    history = model.fit_generator(train_sequence, epochs=EPOCHS, callbacks=[early_stop, tensorbord],
                                  validation_data=val_sequence)
    visualize_training_history(history, RESULTSAVEPATH)

    # careful, I only need bottom layer weights afterwards, find a way to save for further use
    # the difference between bit model and surrogate model is only the last output layer, thus I should replace it before save weights
    # for time concern save the whole model first then debugging for weights
    # model.save_weights(os.path.join(RESULTSAVEPATH, 'final_trained_weights.hdf5'))
    model.save(os.path.join(RESULTSAVEPATH, 'final_trained_model.hdf5'))

    # evaluation
    evaluation = model.evaluate_generator(test_sequence)
    with open(os.path.join(RESULTSAVEPATH, 'evaluation_results.txt'), 'w') as f:
        f.write('Reservation evaluation result on reserved test images: {}: {}\n'
                .format(model.metrics_names, evaluation))
    print('Reservation evaluation result on reserved test images: {}: {}\n'
          .format(model.metrics_names, evaluation))