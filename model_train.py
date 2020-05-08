import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout , Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from third_model.vgg_face import VGGFace

TRAIN_DIRECTORY = r'images'
MODEL_PATH = r'data/model/Model_VGGFace.h5'
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    directory=TRAIN_DIRECTORY,
    target_size=TARGET_SIZE,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory=TRAIN_DIRECTORY,
    target_size=TARGET_SIZE,
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    subset='validation'
)

dictionary = train_generator.class_indices
dictionary = dict (zip(dictionary.values(),dictionary.keys()))
NB_CLASSES = len(dictionary)
np.save('labels.npy', dictionary)


def baseline_model_vgg():
    base_model = VGGFace(model='vgg16' , include_top = False , input_shape =INPUT_SHAPE , pooling='avg')
    last_layer = base_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(100 , activation = 'relu')(x)
    x = Dropout(0.01)(x)
    out = Dense(NB_CLASSES, activation='softmax', name='classifier')(x)
    model = Model(base_model.input, out)

    model.compile(loss = 'categorical_crossentropy' , metrics = ['acc'] , optimizer = Adam(0.00001))
    model.summary()

    return model

if __name__ == "__main__":
    model = baseline_model_vgg()
    plot_model(model, to_file='data/model/model_plot.png', show_shapes=True, show_layer_names=True)
    checkpointer = ModelCheckpoint(filepath=MODEL_PATH, verbose=1, save_best_only=True)
    history = model.fit_generator(
	      train_generator, validation_data=validation_generator,
	      steps_per_epoch = train_generator.samples/train_generator.batch_size ,
	      epochs=20, callbacks=[checkpointer],validation_steps=100,
	      verbose=1, workers=10)
