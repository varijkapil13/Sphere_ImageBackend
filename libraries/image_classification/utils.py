import os
from keras.preprocessing.image import ImageDataGenerator
from pymongo import MongoClient
from bson import ObjectId
from bson.json_util import dumps
from flask import jsonify
from libraries.image_classification.constants import ModelMongoConstants as modelConstants

'''
Data Directories
'''

main_image_data_dir = '/image-downloads/main_image_dataset/'
train_data_dir = '/image-downloads/training-downloads/images/'
test_data_dir = '/image-downloads/testing-downloads/images/'
prediction_images_dir = '/libraries/image_classification/image-downloads/prediction-downloads/'
# Set "dev" for development or "pro" for production
ENVIRONMENT = "dev"

class CheckEnvironment:
    def __init__(self, environment):
        self._environment = environment

    def get_celery_environment(self):
        if self._environment == "dev":
            return "localhost"
        elif self._environment == "pro":
            return "redis"

    def get_mysql_environment(self):
        if self._environment == "dev":
            return "localhost"
        elif self._environment == "pro":
            return "mysql"

    def get_mongo_environment(self):
        if self._environment == "dev":
            return "localhost"
        elif self._environment == "pro":
            return "mongo"


class Response:
    def __init__(self):
        pass

    @staticmethod
    def create_success_response(data, message):
        return jsonify(
                {
                    Constants.status : Constants.true,
                    Constants.data   : data,
                    Constants.message: message
                }), 200

    @staticmethod
    def create_failure_response(message):
        return jsonify(
                {
                    Constants.status       : Constants.false,
                    Constants.data         : 'Error',
                    Constants.error_message: message
                }), 200


class Constants:
    """
    Constants for the project
    """

    def __init__(self):
        pass

    '''
    JSON response constants

    '''
    status = 'status'
    true = 'true'
    false = 'false'
    error_message = 'error'
    system_error_message = 'system_message'
    data = 'data'
    message = 'message'
    exception_occured = 'An Exception Occured \n'

    '''
    MongoDB constants
    '''
    set_environment = CheckEnvironment(ENVIRONMENT)
    mongoDB = {
        'url' : set_environment.get_mongo_environment(),
        'port': 27017,
    }

    '''
    Unicode Color Codes
    '''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    '''
    Directory paths for saving models and weights for Models
    '''

    weights_directory_path = '/model-data/weights/'
    model_file_directory_path = '/model-data/models/'

    @staticmethod
    def saving_model_weights(model_name):
        if model_name == 'VGG16':
            return '/model-data/weights/VGG16/'
        elif model_name == 'VGG19':
            return '/model-data/weights/VGG19/'
        elif model_name == 'ResNet50':
            return '/model-data/weights/ResNet50/'
        elif model_name == 'InceptionV3':
            return '/model-data/weights/InceptionV3/'
        elif model_name == 'Xception':
            return '/model-data/weights/Xception/'
        else:
            return '/model-data/weights/CustomModelWeights/'

    @staticmethod
    def saving_model_specific_file(model_name):
        if model_name == 'VGG16':
            return '/model-data/models/VGG16/'
        elif model_name == 'VGG19':
            return '/model-data/models/VGG19/'
        elif model_name == 'ResNet50':
            return '/model-data/models/ResNet50/'
        elif model_name == 'InceptionV3':
            return '/model-data/models/InceptionV3/'
        elif model_name == 'Xception':
            return '/model-data/models/Xception/'
        else:
            return '/model-data/models/CustomModels/'

    @staticmethod
    def get_image_dimensions(model_name):
        if model_name == 'VGG16' or model_name == 'ResNet50' or model_name == 'VGG19':
            return 224
        elif model_name == 'InceptionV3' or model_name == 'Xception':
            return 299
        else:
            # TODO: Will only support predictions on base models. So remove checking for dimentions for other models
            try:
                find_model = DatabaseFunctions().find_item(modelConstants.database,
                                                           modelConstants.collection,
                                                           modelConstants.mm_model_hash, model_name)
                model = find_model[modelConstants.mm_model_base_name]
                return Constants.get_image_dimensions(model)
            except Exception as e:
                PrintHelper.failure_print("extracting predictions.....", e)
                return 224

    @staticmethod
    def directory_validation(dir_path):
        """
        Check if the directory is present, if not create this directory
        :param dir_path:
        :return:
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def get_number_of_files(directory_name, subdirectory_name=None):
        if subdirectory_name is not None:

            directory = os.path.dirname(os.path.realpath(__file__)) + directory_name + '/'
        else:
            directory = os.path.dirname(os.path.realpath(__file__)) + directory_name + str(subdirectory_name) + '/'

        number_of_files = 0
        for root, dirs, files in os.walk(directory):
            number_of_files = len(files)

        return number_of_files


class ImageModel:
    def __init__(self):
        pass

    def __str__(self):
        return "ImageModel"

    thumbnail = ""
    content = ""
    format = ""


class DatabaseJSONStructure:
    def __init__(self):
        pass

    @staticmethod
    def insert_new_model(search_terms, hash_string, model_name, model_save_name):
        """

        :param search_terms:
        :param hash_string:
        :param model_name:
        :return:
        """

        insert_item = {
            modelConstants.mm_model_hash             : hash_string,
            modelConstants.mm_model_search_terms     : search_terms,
            modelConstants.mm_model_json             : '',
            modelConstants.mm_model_training_acc_loss: '',
            modelConstants.mm_model_h5_file          : '',
            modelConstants.mm_model_weights_file     : '',
            modelConstants.mm_model_base_name        : model_name,
            modelConstants.mm_model_saved_name       : model_save_name,
            modelConstants.mm_model_validation_status: 'trainingPending'
        }
        return insert_item


class DatabaseFunctions:
    def __init__(self):
        self.mongo_client = MongoClient(Constants.mongoDB['url'], Constants.mongoDB['port'])

    def find_all_items(self, database_name, collection_name):
        """

        :param database_name:
        :param collection_name:
        :return:
        """

        try:
            db = self.mongo_client[database_name]
            collection = db[collection_name]
            documents = []
            for models in collection.find():
                documents.append(dumps(models))

            return documents

        except Exception as e:
            PrintHelper().failure_print(e)
            return None

    def update_items(self, database_name, collection_name, object_id, data):
        """

        :param database_name:
        :param collection_name:
        :param object_id:
        :param data:
        :return:
        """

        try:
            db = self.mongo_client[database_name]
            collection = db[collection_name]
            collection.find_one_and_update({modelConstants.mm_id: ObjectId(object_id)}, {'$set': data})
            return True

        except Exception as e:
            PrintHelper().failure_print(e)
            return False

    def find_item(self, database_name, collection_name, object_key, object_value):
        """

        :param database_name:
        :param collection_name:
        :param object_key:
        :param object_value:
        :return:
        """
        try:
            db = self.mongo_client[database_name]
            collection = db[collection_name]

            return collection.find_one({object_key: object_value})

        except Exception as e:
            PrintHelper().failure_print(e)
            return None

    def insert_item(self, database_name, collection_name, items_to_insert):
        """

        :param database_name:
        :param collection_name:
        :param items_to_insert:
        :return:
        """
        try:
            db = self.mongo_client[database_name]
            collection = db[collection_name]
            collection.insert_one(items_to_insert)

        except Exception as e:

            PrintHelper().failure_print(e)

            return False

    def delete_database(self, database):
        try:
            self.mongo_client.drop_database(database)

            return True
        except Exception as e:
            PrintHelper().failure_print(e)
            return False


class DataGeneratorUtility:
    def __init__(self):
        pass

    @staticmethod
    def training_data_generator(img_height, img_width, batch_size, training_data_dir):
        """

        :param img_height:
        :param img_width:
        :param batch_size:
        :param training_data_dir:
        :return:
        """
        train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
                training_data_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False)

        return train_generator

    @staticmethod
    def validation_data_generator(img_height, img_width, batch_size, validation_data_dir):
        """

        :param img_height:
        :param img_width:
        :param batch_size:
        :param validation_data_dir:
        :return:
        """
        vaidation_datagen = ImageDataGenerator(
                rescale=1. / 255)

        validation_generator = vaidation_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='binary',
                shuffle=False)
        return validation_generator


class PrintHelper:
    def __init__(self):
        pass

    @staticmethod
    def info_print(*args):
        """

        :param args:
        :return:
        """
        print_string = ''
        for arg in args:
            if isinstance(arg, str):
                print_string = print_string + arg
            else:
                print_string = print_string + str(arg)
        return
        print(Constants.OKGREEN, '[INFO: ', print_string, '] ', Constants.ENDC)

    @staticmethod
    def warning_print(*args):
        """

        :param args:
        :return:
        """
        print_string = ''
        for arg in args:
            if isinstance(arg, str):
                print_string = print_string + arg
            else:
                print_string = print_string + str(arg)
        return
        print(Constants.WARNING, '[WARNING: ', print_string, '] ', Constants.ENDC)

    @staticmethod
    def failure_print(*args):
        """

        :param args:
        :return:
        """
        print_string = ''
        for arg in args:
            if isinstance(arg, str):
                print_string = print_string + arg
            else:
                print_string = print_string + str(arg)
        return
        print(Constants.FAIL, '[FAILURE: ', print_string, '] ', Constants.ENDC)

    @staticmethod
    def return_string(*args):
        print_string = ''
        for arg in args:
            if isinstance(arg, str):
                print_string = print_string + arg
            else:
                print_string = print_string + str(arg)
        return print_string


class SaveModelToMongo:
    def __init__(self):
        pass

    @staticmethod
    def training_started(classes_hash_string):
        try:
            models_result = DatabaseFunctions().find_item(modelConstants.database, modelConstants.collection,
                                                          modelConstants.mm_model_hash,
                                                          classes_hash_string)
            data = {
                modelConstants.mm_model_validation_status: 'training',
            }

            DatabaseFunctions().update_items(modelConstants.database, modelConstants.collection, models_result['_id'],
                                             data)
        except Exception as e:
            PrintHelper().failure_print(e)
            pass

    @staticmethod
    def validation_started(class_hash):
        try:
            models_result = DatabaseFunctions().find_item(modelConstants.database, modelConstants.collection,
                                                          modelConstants.mm_model_hash, class_hash)

            data = {modelConstants.mm_model_validation_status: 'testing'}

            DatabaseFunctions().update_items(modelConstants.database, modelConstants.collection, models_result['_id'],
                                             data)
            return models_result

        except Exception as e:
            PrintHelper().failure_print(e)
            pass

    @staticmethod
    def training_completed(classes_hash_string, save_model_path, save_weights, json_string_model):
        try:
            models_result = DatabaseFunctions().find_item(modelConstants.database, modelConstants.collection,
                                                          modelConstants.mm_model_hash,
                                                          classes_hash_string)

            data = {
                modelConstants.mm_model_weights_file     : save_weights,
                modelConstants.mm_model_h5_file          : save_model_path,
                modelConstants.mm_model_json             : json_string_model,
                modelConstants.mm_model_validation_status: 'trained',
            }

            DatabaseFunctions().update_items(modelConstants.database, modelConstants.collection, models_result['_id'],
                                             data)
        except Exception as e:
            PrintHelper().failure_print(e)
            pass

    @staticmethod
    def validation_completed(class_hash, models_result, stats):

        try:
            model_id = '_id'

            if model_id in models_result:
                data = {
                    modelConstants.mm_model_training_acc_loss: stats,
                    modelConstants.mm_model_validation_status: 'tested'
                }

                DatabaseFunctions().update_items(modelConstants.database, modelConstants.collection,
                                                 models_result['_id'], data)
            else:
                models_result = DatabaseFunctions().find_item(modelConstants.database, modelConstants.collection,
                                                              modelConstants.mm_model_hash,
                                                              class_hash)
                data = {
                    modelConstants.mm_model_training_acc_loss: stats,
                    modelConstants.mm_model_validation_status: 'tested'
                }

                DatabaseFunctions().update_items(modelConstants.database, modelConstants.collection,
                                                 models_result['_id'], data)

        except Exception as e:
            PrintHelper().failure_print(e)
            pass
