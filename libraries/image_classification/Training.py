import os, shutil
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.engine.topology import Input
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Sequential, load_model
from libraries.image_classification.utils import PrintHelper, DataGeneratorUtility, SaveModelToMongo, Constants, \
    DatabaseFunctions, train_data_dir
from libraries.image_classification.constants import ModelMongoConstants as modelConstants

# TODO: change epochs
epochs = 1
batch_size = 20
cph = PrintHelper()


class Training:
    def __str__(self):

        return "Training"

    def __init__(self, model_name, model_hash, number_of_classes, number_of_samples):
        """

        :param model_name:
        :param model_hash:
        :param number_of_classes:
        :param number_of_samples:
        """
        Constants.directory_validation(
                os.path.dirname(os.path.realpath(__file__)) + Constants.saving_model_specific_file(model_name))
        Constants.directory_validation(
                os.path.dirname(os.path.realpath(__file__)) + Constants.saving_model_weights(model_name))

        cph.info_print('Using' + model_name)
        self._classes_hash_string = model_hash
        self._img_height = Constants.get_image_dimensions(model_name)
        self._img_width = self._img_height

        self._save_weights = os.path.dirname(
                os.path.realpath(__file__)) + Constants.weights_directory_path + 'Weights#' + model_hash + '.h5'
        self._save_model_path = os.path.dirname(
                os.path.realpath(__file__)) + Constants.model_file_directory_path + 'Model_file#' + model_hash + '.h5'
        self._save_weights_link = Constants.weights_directory_path + 'Weights#' + model_hash + '.h5'
        self._save_model_link = Constants.model_file_directory_path + 'Model_file#' + model_hash + '.h5'
        self._train_data_dir = os.path.dirname(os.path.realpath(__file__)) + train_data_dir + model_hash
        self._base_train_dir = os.path.dirname(os.path.realpath(__file__)) + train_data_dir
        self._nb_train_samples = number_of_samples * number_of_classes
        self._number_of_classes = number_of_classes
        self._model_name = model_name

    def start_model_training(self):

        if self._model_name == 'VGG16' or self._model_name == 'vgg16':
            model_json_format = self._vgg16_training()
        elif self._model_name == 'VGG19' or self._model_name == 'vgg19':
            model_json_format = self._vgg19_training()
        elif self._model_name == 'ResNet50' or self._model_name == 'resNet50' \
                or self._model_name == 'Resnet50' or self._model_name == 'resnet50':
            model_json_format = self._resnet50_training()
        elif self._model_name == 'Xception' or self._model_name == 'xception':
            model_json_format = self._xception_training()
        elif self._model_name == 'InceptionV3' or self._model_name == 'inceptionV3' \
                or self._model_name == 'inceptionv3' or self._model_name == 'Inceptionv3':
            model_json_format = self._inceptionv3_training()
        else:
            model_json_format = self._custom_model_training()

        shutil.rmtree(self._train_data_dir)
        SaveModelToMongo().training_completed(classes_hash_string=self._classes_hash_string,
                                              save_model_path=self._save_model_link,
                                              save_weights=self._save_weights_link, json_string_model=model_json_format)

        return True

    def _vgg16_training(self):
        # set class activation function and loss type
        activation_function = 'sigmoid'
        loss_function = 'binary_crossentropy'
        if self._number_of_classes > 2:
            loss_function = 'categorical_crossentropy'
            if self._number_of_classes > 10:
                activation_function = 'softmax'

        # create the base pre-trained model
        input_tensor = Input(shape=(self._img_width, self._img_height, 3))
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(self._number_of_classes, activation=activation_function))

        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional layers
        for layer in model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])
        cph.info_print('1st Compilation Done')
        # prepare data augmentation configuration

        SaveModelToMongo().training_started(self._classes_hash_string)

        train_generator = DataGeneratorUtility.training_data_generator(self._img_height, self._img_width,
                                                                       batch_size=batch_size,
                                                                       training_data_dir=self._train_data_dir)

        # train the model on the new data for a few epochs
        cph.info_print('Will Start fit_generator')
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)

        # we chose to train the top 2  blocks, i.e. we will freeze
        # the first 7 layers and unfreeze the rest:
        for layer in model.layers[:7]:
            layer.trainable = False
        for layer in model.layers[7:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)
        model.save_weights(self._save_weights)
        model.save(self._save_model_path)

        # Json Model
        json_string_model = model.to_json()

        return json_string_model

    def _vgg19_training(self):
        # set class activation function and loss type
        activation_function = 'sigmoid'
        loss_function = 'binary_crossentropy'
        if self._number_of_classes > 2:
            loss_function = 'categorical_crossentropy'
            if self._number_of_classes > 10:
                activation_function = 'softmax'

        # create the base pre-trained model
        input_tensor = Input(shape=(self._img_width, self._img_height, 3))
        base_model = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))

        top_model.add(Dense(self._number_of_classes, activation=activation_function))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])
        cph.info_print('1st Compilation Done')
        # prepare data augmentation configuration

        SaveModelToMongo().training_started(self._classes_hash_string)

        train_generator = DataGeneratorUtility.training_data_generator(self._img_height, self._img_width,
                                                                       batch_size=batch_size,
                                                                       training_data_dir=self._train_data_dir)

        # train the model on the new data for a few epochs
        cph.info_print('Will Start fit_generator')
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        # print(i, layer.name)

        # we chose to train the top 2 vgg blocks, i.e. we will freeze
        # the first 7 layers and unfreeze the rest:
        for layer in model.layers[:7]:
            layer.trainable = False
        for layer in model.layers[7:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)
        model.save_weights(self._save_weights)
        model.save(self._save_model_path)

        # Json Model
        json_string_model = model.to_json()
        return json_string_model

    def _resnet50_training(self):
        activation_function = 'sigmoid'
        loss_function = 'binary_crossentropy'
        if self._number_of_classes > 2:
            loss_function = 'categorical_crossentropy'
            if self._number_of_classes > 10:
                activation_function = 'softmax'

        # create the base pre-trained model
        input_tensor = Input(shape=(self._img_width, self._img_height, 3))
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))

        top_model.add(Dense(self._number_of_classes, activation=activation_function))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])
        cph.info_print('1st Compilation Done')

        # prepare data augmentation configuration

        SaveModelToMongo().training_started(self._classes_hash_string)

        train_generator = DataGeneratorUtility.training_data_generator(self._img_height, self._img_width,
                                                                       batch_size=batch_size,
                                                                       training_data_dir=self._train_data_dir)

        # train the model on the new data for a few epochs
        cph.info_print('Will Start fit_generator')
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        # print(i, layer.name)

        # we chose to train the top 2 vgg blocks, i.e. we will freeze
        # the first 7 layers and unfreeze the rest:
        for layer in model.layers[:11]:
            layer.trainable = False
        for layer in model.layers[11:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)
        model.save_weights(self._save_weights)
        model.save(self._save_model_path)

        # Json Model
        json_string_model = model.to_json()
        return json_string_model

    def _xception_training(self):
        # set class activation function and loss type
        activation_function = 'sigmoid'
        loss_function = 'binary_crossentropy'
        if self._number_of_classes > 2:
            loss_function = 'categorical_crossentropy'
            if self._number_of_classes > 10:
                activation_function = 'softmax'

        # create the base pre-trained model
        input_tensor = Input(shape=(self._img_width, self._img_height, 3))
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))

        top_model.add(Dense(self._number_of_classes, activation=activation_function))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])
        cph.info_print('1st Compilation Done')
        # prepare data augmentation configuration
        SaveModelToMongo().training_started(self._classes_hash_string)

        train_generator = DataGeneratorUtility.training_data_generator(self._img_height, self._img_width,
                                                                       batch_size=batch_size,
                                                                       training_data_dir=self._train_data_dir)

        # train the model on the new data for a few epochs
        cph.info_print('Will Start fit_generator')
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        # print(i, layer.name)

        # we chose to train the top 2 vgg blocks, i.e. we will freeze
        # the first 7 layers and unfreeze the rest:
        for layer in model.layers[:15]:
            layer.trainable = False
        for layer in model.layers[15:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)
        model.save_weights(self._save_weights)
        model.save(self._save_model_path)

        # Json Model
        json_string_model = model.to_json()
        return json_string_model

    def _inceptionv3_training(self):
        # set class activation function and loss type
        activation_function = 'sigmoid'
        loss_function = 'binary_crossentropy'
        if self._number_of_classes > 2:
            loss_function = 'categorical_crossentropy'
            if self._number_of_classes > 10:
                activation_function = 'softmax'

        # create the base pre-trained model
        input_tensor = Input(shape=(self._img_width, self._img_height, 3))
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))

        top_model.add(Dense(self._number_of_classes, activation=activation_function))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])
        cph.info_print('1st Compilation Done')
        # prepare data augmentation configuration
        SaveModelToMongo().training_started(self._classes_hash_string)

        train_generator = DataGeneratorUtility.training_data_generator(self._img_height, self._img_width,
                                                                       batch_size=batch_size,
                                                                       training_data_dir=self._train_data_dir)

        # train the model on the new data for a few epochs
        cph.info_print('Will Start fit_generator')
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        # print(i, layer.name))

        # we chose to train the top 2 vgg blocks, i.e. we will freeze
        # the first 7 layers and unfreeze the rest:
        for layer in model.layers[:17]:
            layer.trainable = False
        for layer in model.layers[17:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)
        model.save_weights(self._save_weights)
        model.save(self._save_model_path)

        # Json Model
        json_string_model = model.to_json()
        return json_string_model

    def _custom_model_training(self):

        # https://groups.google.com/forum/#!topic/keras-users/rK32gsHCY9k
        # keep training on a saved model will not reset the previous training

        # load the model from the models directory, the model name sent here is actually the hash name + name
        model_path = os.path.dirname(
                os.path.realpath(
                        __file__)) + Constants.model_file_directory_path + 'Model_file#' + self._model_name + '.h5'
        model = load_model(model_path)

        # set the loss fucntion based on the number of classes
        loss_function = 'binary_crossentropy'
        if self._number_of_classes > 2:
            loss_function = 'categorical_crossentropy'

        SaveModelToMongo().training_started(self._classes_hash_string)

        train_generator = DataGeneratorUtility.training_data_generator(self._img_height, self._img_width,
                                                                       batch_size=batch_size,
                                                                       training_data_dir=self._train_data_dir)

        # train the model on the new data
        model.compile(optimizer='rmsprop', loss=loss_function, metrics=['accuracy'])
        cph.info_print('Compilation Done')

        cph.info_print('Will Start fit_generator')
        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit_generator(train_generator, steps_per_epoch=self._nb_train_samples, epochs=epochs)
        model.save_weights(self._save_weights)
        model.save(self._save_model_path)

        # Json Model
        json_string_model = model.to_json()

        return json_string_model
