#!flask/bin/python3
import hashlib
import os
import sys
import tarfile
from celery import Celery
from flask import Flask, jsonify, request, abort, make_response, send_file
from flask_cors import CORS
from flask_restful import Api
from libraries.image_classification.FeatureExtraction import FeatureExtraction
from libraries.image_classification.ImageDownloader import ImageDownloader
from libraries.image_classification.Training import Training
from libraries.image_classification.ValidationClasses import ModelValidation
from libraries.image_classification.constants import GoldStandardConstants as gsc, \
    ModelMongoConstants as modelConstants, ImageDatasetConstants as imageDatasetConstants
from libraries.image_classification.utils import Constants, PrintHelper, \
    DatabaseJSONStructure, DatabaseFunctions, Response, prediction_images_dir, CheckEnvironment

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
gold_standards_dir = os.path.join(ROOT_DIR, "backend", "data", "resources", "added_gold_standards")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
env = os.environ

# Set "dev" for development or "pro" for production
ENVIRONMENT = "dev"

set_environment = CheckEnvironment(ENVIRONMENT)

CELERY_BROKER_URL = env.get('CELERY_BROKER_URL', 'redis://' + set_environment.get_celery_environment() + ':6379/0'),
CELERY_RESULT_BACKEND = env.get('CELERY_RESULT_BACKEND',
                                'redis://' + set_environment.get_celery_environment() + ':6379/0')

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CELERY_BROKER_URL'] = CELERY_BROKER_URL
app.config['CELERY_RESULT_BACKEND'] = CELERY_RESULT_BACKEND

celery = Celery(app.name,
                broker=CELERY_BROKER_URL,
                backend=CELERY_RESULT_BACKEND)
celery.conf.update(app.config)

cors = CORS(app)
api = Api(app)

"""
Image Classification
"""


@app.route('/sphere/api/v1/image/recieveImages', methods=['POST'])
def download_dataset_images():
    if not request.json or 'images' not in request.json:
        abort(400)
    try:
        image_list = request.json['images']
        background_image_downloader.delay(image_list)
        return Response.create_success_response("Images Downloaded",
                                                "Images are being downloaded. Once downloaded, "
                                                "they will be available for training")

    except Exception as e:
        PrintHelper().failure_print(e)
        return Response.create_failure_response('Could not download images with Error:  ' + str(e))


@app.route('/sphere/api/v1/image/startTraining', methods=['POST'])
def image_transfer_and_training():
    """

    :return:
    """

    if not request.json or 'datasets' not in request.json or 'model' not in request.json:
        abort(400)

    try:

        dataset_items = request.json['datasets']
        dataset_string = ','.join(str(element) for element in dataset_items)
        selected_model_name = request.json['model']
        save_name_for_model = request.json['name']

        same_name = DatabaseFunctions().find_item(modelConstants.database, modelConstants.collection,
                                                  modelConstants.mm_model_saved_name, save_name_for_model)
        if same_name is not None:
            return Response().create_failure_response(
                    "A model with the same name is already present. Please use a different name")

        if selected_model_name is not "VGG16" and selected_model_name is not "VGG19" and selected_model_name is not "ResNet50" and selected_model_name is not "Xception" and selected_model_name is not "Inceptionv3":
            models_terms = DatabaseFunctions().find_item(modelConstants.database, modelConstants.collection,
                                                         modelConstants.mm_model_saved_name, selected_model_name)
            if models_terms is not None:
                present_search_terms = models_terms[modelConstants.mm_model_search_terms]
                dataset_string = dataset_string + "," + present_search_terms

        encoded_keys = (dataset_string + selected_model_name).encode('utf-8')
        classes_hash_key = hashlib.sha256(encoded_keys)
        hash_digest = classes_hash_key.hexdigest()
        hash_digest = save_name_for_model + "###" + hash_digest
        models_result = DatabaseFunctions().find_item(
                modelConstants.database,
                modelConstants.collection,
                modelConstants.mm_model_hash, hash_digest)

        if models_result is None:
            PrintHelper().info_print('Model was not found, continue with the processing')

            DatabaseFunctions().insert_item(
                    database_name=modelConstants.database,
                    collection_name=modelConstants.collection,
                    items_to_insert=DatabaseJSONStructure.insert_new_model(dataset_string, hash_digest,
                                                                           selected_model_name,
                                                                           model_save_name=save_name_for_model))

            status, number_of_samples, number_of_classes = ImageDownloader().transfer_datasets_and_start_training(
                    dataset_items, hash_digest)
            if status:
                start_background_training.delay(selected_model_name, hash_digest, number_of_classes, number_of_samples)

                return Response().create_success_response('',
                                                          'Training Started on selected Model')
            else:
                return Response().create_failure_response(
                        'An already trained or currently in training model is already '
                        'present in the database. Please check "View and Validate" tab to '
                        'view and test this model')
        else:
            PrintHelper().warning_print('Model already present')
            return Response().create_failure_response(
                    'An already trained or currently in training model is already '
                    'present in the database. Please check "View and Validate" tab to'
                    ' view and test this model')

    except Exception as e:
        PrintHelper().failure_print(e)
        return Response.create_failure_response(
                'Exception Occured: image_transfer_and_training' + str(e))


@app.route('/sphere/api/v1/image/getDatasets', methods=['GET'])
def get_image_dataset_list():
    """

    :return:
    """
    try:
        image_dataset = DatabaseFunctions().find_all_items(imageDatasetConstants.database,
                                                           imageDatasetConstants.collection)
        return Response.create_success_response(image_dataset, 'Dataset fetched successfully')
    except Exception as e:
        PrintHelper().failure_print(e)
        return Response.create_failure_response('Database Error')


@app.route('/sphere/api/v1/image/getModels', methods=['GET'])
def get_saved_models():
    """

    :return:
    """
    if request.method != 'GET':
        abort(403)

    try:
        saved_models = DatabaseFunctions().find_all_items(database_name=modelConstants.database,
                                                          collection_name=modelConstants.collection)

        if saved_models is None:
            PrintHelper().failure_print('No models found in the database')

            return Response.create_failure_response('No saved models found')

        else:
            return Response.create_success_response(saved_models, 'Fetched models successfully')
    except Exception as e:

        PrintHelper().failure_print('sphere_WebAPI', e)
        return Response.create_failure_response('Database error')


@app.route('/sphere/api/v1/image/startValidation', methods=['POST'])
def start_validation():
    """

    :return:
    """
    if request.method != 'POST':
        abort(403)

    if not request.json or 'model' not in request.json or 'hash' not in request.json:
        abort(400)
    try:

        model_name = request.json['model']
        hash_key = request.json['hash']
        start_background_validation.delay(model_name, hash_key)
        return Response.create_success_response('Validation Started', 'Validation Started')

    except Exception as e:
        PrintHelper().failure_print(e)
        return Response.create_failure_response('Could not start validation process')


''' Prediction API'''


@app.route('/sphere/api/v1/image/prediction', methods=['POST'])
def start_predictions():
    try:
        form = request.form

        model_name = ""

        # Needed only with custom model training
        base_model = None
        model_hash = None
        ######

        for key, value in form.items():
            if key == 'model_hash':
                model_hash = value
            if key == 'model_name':
                model_name = value
            if key == 'model_base_name':
                base_model = value

        image_count = 0
        image_paths = ''
        save_files = request.files.getlist("file")
        for upload in request.files.getlist("file"):
            filename = upload.filename.rsplit("/")[0]
            PrintHelper.info_print("Uploaded: ", filename)
            destination_main = os.path.dirname(
                    os.path.realpath(__file__)) + prediction_images_dir + '/'
            destination = destination_main + model_name + '/'
            Constants.directory_validation(destination)
            upload.save(destination + filename)
            image_paths = destination_main
            image_count = + 1

        feature_extraction = FeatureExtraction(model_name, image_paths, image_count)
        predictions = feature_extraction.get_predictions()

        # converting the complex tuple structre into javascript friendly json object
        count = 0
        prediction_array = []
        for item in predictions:
            result_outer = []
            for items in item:
                result_inner = {
                    "class"      : items[0],
                    "description": items[1],
                    "probability": str(items[2])
                }
                result_outer.append(result_inner)

            image_file = save_files[count]
            count = count + 1

            image_file_name = image_file.filename.rsplit("/")[0]

            final_prediction_result = {
                "filename"   : image_file_name,
                "predictions": result_outer
            }
            prediction_array.append(final_prediction_result)

        return Response.create_success_response(prediction_array, "success")

    except Exception as e:
        PrintHelper.failure_print("An error occurered:  " + str(e))
        return Response.create_failure_response("An error occurred: " + str(e))


@app.route('/sphere/api/v1/image/datasetUpload', methods=['POST'])
def image_dataset_upload():
    try:
        form = request.form

        class_name = ""

        for key, value in form.items():
            if key == 'class_name':
                class_name = value

        image_file_list = request.files.getlist("file")

        ImageDownloader().recieve_image_dataset_from_user(image_file_list, class_name)

        return Response.create_success_response("Images Downloaded successfully", "success")

    except Exception as e:
        PrintHelper.failure_print("An error occurered:  " + str(e))
        return Response.create_failure_response("An error occurred: " + str(e))


@app.route('/sphere/api/v1/image/fileDownload', methods=['POST'])
def download_files():
    filetype = request.json['type']
    path = request.json['path']
    if filetype == 'gold':
        return send_file(os.path.dirname(os.path.realpath(__file__)) + path)
    else:
        return send_file(os.path.dirname(os.path.realpath(__file__)) + "/libraries/image_classification" + path)


"""

Celery tasks

"""


@celery.task
def background_image_downloader(image_list):
    """

    :param image_list:
    :return:
    """
    ImageDownloader().download_main_image_dataset(image_list)


@celery.task
def start_background_training(selected_model_name, hash_digest, number_of_classes,
                              number_of_samples):
    training_object = Training(selected_model_name, hash_digest, number_of_classes,
                               number_of_samples)
    training_object.start_model_training()


@celery.task
def start_background_validation(model_name, hash_key):
    """

    :param model_name:
    :param hash_key:
    :return:
    """
    ModelValidation().start_validation(model_name, hash_key)


"""
Error Handlers

"""


@app.errorhandler(404)
def not_found(error):
    """

    :return:
    """
    PrintHelper().failure_print('404 Error')
    return make_response(
            jsonify({Constants.status: Constants.false, Constants.error_message: 'Not found'}), 404)


@app.errorhandler(400)
def invalid_data(error):
    """


    :return:
    """
    PrintHelper().failure_print('400 Error')
    return make_response(
            jsonify({Constants.status: Constants.false, Constants.error_message: 'Invalid data'}), 400)


@app.errorhandler(403)
def invalid_request(error):
    """

    :return:
    """
    PrintHelper().failure_print('403 Error')
    return make_response(
            jsonify({Constants.status: Constants.false, Constants.error_message: 'Invalid request'}),
            403)


@app.route('/sphere/api/v1/dbreset/<db_name>', methods=['GET'])
def reset_db(db_name):
    database_name = 'nothing'
    if db_name == "gold":
        database_name = gsc.database
    elif db_name == "image":
        database_name = imageDatasetConstants.database
    elif db_name == "model":
        database_name = modelConstants.database

    if database_name is not 'nothing':

        DatabaseFunctions().delete_database(database_name)
        return Response.create_success_response("success", "success")
    else:
        return Response.create_failure_response("error")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
