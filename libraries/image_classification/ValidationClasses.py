import os, shutil
from keras.models import load_model
from libraries.image_classification.utils import PrintHelper, DataGeneratorUtility, Constants, SaveModelToMongo
from libraries.image_classification.utils import test_data_dir


class ModelValidation:
	def __init__(self):
		pass

	def __str__(self):
		"""
		Prints the name of the class (the return string) when 'self' is used in a print function
		:return:
		"""
		return "ModelValidation"

	@staticmethod
	def start_validation(model_name, class_hash):
		"""

		:return:
		"""

		img_width = Constants.get_image_dimensions(model_name)
		img_height = img_width

		cph = PrintHelper()
		validation_data_dir = os.path.dirname(os.path.realpath(__file__)) + test_data_dir + class_hash
		base_validation_dir = os.path.dirname(os.path.realpath(__file__)) + test_data_dir
		save_weights = os.path.dirname(
			os.path.realpath(__file__)) + Constants.weights_directory_path + 'Weights#' + class_hash + '.h5'
		save_model_path = os.path.dirname(
				os.path.realpath(__file__)) + Constants.model_file_directory_path + 'Model_file#' + class_hash + '.h5'

		batch_size = 5
		number_of_files = ModelValidation.number_of_images(validation_data_dir)

		cph.info_print('Loading Model')

		model = load_model(save_model_path)
		model.load_weights(save_weights)

		models_result = SaveModelToMongo.validation_started(class_hash)

		PrintHelper.info_print(' Model Loaded')
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

		PrintHelper.info_print(' Model Compiled')
		# prepare data augmentation configuration

		PrintHelper.info_print('creating image data generator')

		validation_generator = DataGeneratorUtility.validation_data_generator(img_height, img_width,
		                                                                      batch_size=batch_size,
		                                                                      validation_data_dir=validation_data_dir)

		PrintHelper.info_print('Starting Evaluate Generator')
		loss, accuracy = model.evaluate_generator(validation_generator,number_of_files)

		PrintHelper.info_print('Loss: ', loss, ' Accuracy: ', accuracy)

		shutil.rmtree(validation_data_dir)
		SaveModelToMongo.validation_completed(class_hash=class_hash,
		                                      stats=cph.return_string('Accuracy: ', round(accuracy, 4), ' Loss: ',
		                                                              round(loss, 4)),
		                                      models_result=models_result)


	@staticmethod
	def number_of_images(path):
		directories = os.listdir(path)
		count = 0
		for folder in directories:
			if os.path.isdir(path + "/" + folder):
				folder_files = os.listdir(path + "/" + folder)
				for files in folder_files:
					count = count + 1

		return count

