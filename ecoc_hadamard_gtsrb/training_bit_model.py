""" training 1 bit ensemble models for ecoc system on gtsrb"""
import numpy as np
from utils.train_support import DataSequence, visualize_training_history
import albumentations as aug
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
import os
from ecoc_hadamard_gtsrb.models import get_1bit_ensemble_vgg16
from utils.readTrafficSigns import Hadamard_labels

if __name__ == '__main__':
    # paras needs change every time
    BIT_MODEL_INDEX = 11  # starts from 1
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    BATCH_SIZE = 32
    NUM_FROZEN_LAYER = 6
    ADAM_DECAY = 0.8
    TRAIN_IMG_FOLDER = r'D:\LableCodingNetwork\database\GTSRB\Final_Training\Images'
    TEST_IMG_FOLDER = r'D:\LableCodingNetwork\database\GTSRB\Final_Test\Images'
    RESULTSAVEPATH = r'D:\LableCodingNetwork\trained-models\GTSRB\ECOC\Hadamard32_surrogate_weights_freeze{}_bit_{}'.format(NUM_FROZEN_LAYER, BIT_MODEL_INDEX)
    if not os.path.exists(RESULTSAVEPATH):
        os.makedirs(RESULTSAVEPATH)
    HADAMARD_MATRIX = np.load(r'D:\LableCodingNetwork\ecoc_hadamard_gtsrb\hadamard32.npy')
    TOP32GTSRB_CATEGORIES = np.load(r'D:\LableCodingNetwork\database\GTSRB\gysrb_top32category_label.npy')


    # dataset prepartion
    train_val_img_paths, train_val_bit_model_labels = Hadamard_labels(TRAIN_IMG_FOLDER, TOP32GTSRB_CATEGORIES,
                                                                      HADAMARD_MATRIX, BIT_MODEL_INDEX)

    assert len(train_val_img_paths) == len(train_val_bit_model_labels)
    l = len(train_val_img_paths)
    np.random.seed(1)
    np.random.shuffle(train_val_img_paths)
    np.random.seed(1)
    np.random.shuffle(train_val_bit_model_labels)
    train_img_paths, val_img_paths = train_val_img_paths[:int(l*0.8)], train_val_img_paths[int(l*0.8):]
    train_bit_model_labels, val_bit_model_labels = \
        train_val_bit_model_labels[:int(l*0.8)], train_val_bit_model_labels[int(l*0.8):]

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
                                  train_bit_model_labels,  # does it need to be one-hot?
                                  BATCH_SIZE,
                                  augmentations=trainAugSetting)
    val_sequence = DataSequence(val_img_paths,
                                val_bit_model_labels,
                                BATCH_SIZE)
    test_sequence = DataSequence(test_img_paths,
                                 test_bit_model_labels,
                                 BATCH_SIZE)

    # get model and training setting
    model = get_1bit_ensemble_vgg16(r'D:\LableCodingNetwork\trained-models\GTSRB\surrogate_VGG16_Dec23\output_replaced_with_dense_for_bit_model.hdf5')
    for idx, layer in enumerate(model.layers):
        if idx <=NUM_FROZEN_LAYER:
            layer.trainable = False
        print('number {} layer, named {} trainable={}'.format(idx, layer.name, layer.trainable))
    optimizer = Adam(lr=LEARNING_RATE, decay=ADAM_DECAY)
    model.compile(loss='hinge',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto', baseline=None,
                               restore_best_weights=True)
    # check_point = ModelCheckpoint(os.path.join(RESULTSAVEPATH, 'weights.epoch{epoch:02d}-val_loss{val_loss:.2f}.hdf5'),
    #                               save_best_only=True)
    tensorbord = TensorBoard(RESULTSAVEPATH)

    # training
    history = model.fit_generator(train_sequence, epochs=EPOCHS, callbacks=[early_stop,  tensorbord], # check_point,
                                  validation_data=val_sequence)
    visualize_training_history(history, RESULTSAVEPATH)
    model.save_weights(os.path.join(RESULTSAVEPATH, 'final_trained_weights.hdf5'))

    # evaluation
    evaluation = model.evaluate_generator(test_sequence)
    with open(os.path.join(RESULTSAVEPATH, 'evaluation_results.txt'), 'w') as f:
        f.write('Reservation evaluation result on reserved test images: {}: {}\n'
                .format(model.metrics_names, evaluation))
    print('Reservation evaluation result on reserved test images: {}: {}\n'
          .format(model.metrics_names, evaluation))