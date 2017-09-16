from libraries.image_classification.utils import Constants, PrintHelper
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
import numpy as np
import os, shutil

from keras.preprocessing.image import ImageDataGenerator


class FeatureExtraction:
	def __str__(self):
		return 'Feature Extraction'

	def __init__(self, model_name, images, image_count, model_hash=None, base_model=None):
		"""

		:param model_name: name/hash of the model to predict/extract-features from
		:param images: path for the images downlaoded
		:param model_hash: hash saved in the database: required for loading weights and files
		:param base_model: required only when a custom generated model is being evaluated, check what is the base model of the model being used
		"""
		self._images_path = images
		self._model_name = model_name
		size = Constants.get_image_dimensions(model_name)
		self._target_size = (size, size)
		self._image_count = image_count

		self._model_path = None
		self._model_weights = None

		# only required for custom models, in which case base_model should be supplied
		if base_model is not None and model_hash is not None:
			try:
				self._model_path = os.path.dirname(os.path.realpath(__file__)) + Constants.saving_model_specific_file(
						base_model) + base_model + '----model_file' + model_hash + '.h5'
				self._model_weights = os.path.dirname(os.path.realpath(__file__)) + Constants.saving_model_weights(
						base_model) + base_model + '----weights' + model_hash + '.h5'
			except Exception as e:
				PrintHelper.failure_print("model file creation", e)

	def extract_features(self):
		"""

		:return: A list of features
		"""
		try:
			model = self._load_model()
			features = self._get_features(model)
			return features
		except Exception as e:
			PrintHelper.failure_print("extract features.....", e)

	def get_predictions(self):
		"""

		:return: a list of predictions
		"""
		features = self.extract_features()
		predictions = None
		try:
			predictions = decode_predictions(features, top=3)
		except ValueError as e:
			PrintHelper.failure_print("get predictions.....", e)

		shutil.rmtree(self._images_path)
		return predictions

	def _get_features(self, model):
		"""

		:param model: a Keras model instance
		:return: A list of features
		"""
		features = None
		if self._model_weights:

			image_path = self._images_path + 'cats4da3c9757cb337c692b9517b9f82abf7f21c681d8b1ce111f7ce546f30bdcf95.jpeg'
			PrintHelper.info_print("in if   :  ", image_path)
			img = image.load_img(image_path, target_size=self._target_size)
			x = image.img_to_array(img)
			x = x.transpose(1, 0, 2)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = model.predict(x)

		else:
			vaidation_datagen = ImageDataGenerator()

			validation_generator = vaidation_datagen.flow_from_directory(
					self._images_path,
					target_size=self._target_size,
					batch_size=20,
					class_mode='binary',
					shuffle=False)
			try:
				features = model.predict_generator(validation_generator, steps=self._image_count)
			except ValueError as e:
				PrintHelper.failure_print("_get features.....", e)

		return features

	def _load_model(self):
		model = None
		try:
			if self._model_name == 'VGG16' or self._model_name == 'vgg16':
				model = VGG16(weights='imagenet', include_top=True)
			elif self._model_name == 'VGG19' or self._model_name == 'vgg19':
				model = VGG19(weights='imagenet', include_top=True)
			elif self._model_name == 'ResNet50' or self._model_name == 'resNet50' \
					or self._model_name == 'Resnet50' or self._model_name == 'resnet50':
				model = ResNet50(weights='imagenet', include_top=True)
			elif self._model_name == 'Xception' or self._model_name == 'xception':
				model = Xception(weights='imagenet', include_top=True)
			elif self._model_name == 'InceptionV3' or self._model_name == 'inceptionV3' \
					or self._model_name == 'inceptionv3' or self._model_name == 'Inceptionv3':
				model = InceptionV3(weights='imagenet', include_top=True)
			else:
				model = load_model(self._model_path)
				model.load_weights(self._model_weights)

		except Exception as e:
			PrintHelper.failure_print("loading models.....", e)

		return model
