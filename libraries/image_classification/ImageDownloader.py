import os
import shutil
import random
import requests
import hashlib
from libraries.image_classification.utils import Constants, PrintHelper, ImageModel, main_image_data_dir, \
    train_data_dir, test_data_dir, DatabaseFunctions
from libraries.image_classification.constants import ImageDatasetConstants as imageDatasetConstants

cph = PrintHelper()


class ImageDownloader:
    def __init__(self):
        pass

    def __str__(self):
        return "ImageDownloader"

    imageUrls = []

    directory_name = ''
    selected_model_name = ''

    @staticmethod
    def _request_downloader(url, path):
        """
        
        :param url: 
        :param path: 
        :return: 
        """
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                try:
                    with open(path, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                        return True
                except Exception as e:
                    cph.failure_print(e)
                    return False
            else:
                return False
        except Exception as e:
            cph.failure_print(e)
            return False

    def download_main_image_dataset(self, image_list):
        successfully_downloaded = False

        for key, value in image_list.items():

            dataset = DatabaseFunctions().find_item(imageDatasetConstants.database, imageDatasetConstants.collection,
                                                    imageDatasetConstants.id_dataset_name, key)
            if dataset is None:

                imageurls = []
                for image in value:
                    imagemodel = ImageModel()
                    imagemodel.thumbnail = image['thumbnailUrl']
                    imagemodel.content = image['contentUrl']
                    imagemodel.format = image['format']
                    imageurls.append(imagemodel)

                self.imageUrls = imageurls
                successfully_downloaded = self._download_image_dataset(key)
                if successfully_downloaded:
                    number_of_images = Constants.get_number_of_files(main_image_data_dir, key)
                    mongo_entry = {
                        'name' : key,
                        'count': number_of_images
                    }
                    DatabaseFunctions().insert_item(imageDatasetConstants.database, imageDatasetConstants.collection,
                                                    items_to_insert=mongo_entry)
            else:
                PrintHelper.warning_print("Image dataset already present")
                continue

    def _download_image_dataset(self, image_class_name):
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__)) + main_image_data_dir + image_class_name + '/'
            Constants.directory_validation(dir_path)
            if len(self.imageUrls) > 0:

                for imagedata in self.imageUrls:

                    random_hash = hashlib.sha256(str(random.randint(0, 1000000000000000000)).encode('utf-8') +
                                                 str(random.randint(0, 1000000000000000000)).encode('utf-8'))

                    path = dir_path + image_class_name + str(random_hash.hexdigest()) + '.' + imagedata.format
                    url = imagedata.content
                    download_success = self._request_downloader(url, path)

                    if download_success:
                        cph.info_print(self, ': Downloaded an image for ', image_class_name)
                    else:
                        cph.failure_print(self, ': Error downloading image to ' + path)
                return True
            else:
                cph.failure_print('No images in the list')
                return False
        except Exception as e:
            cph.failure_print("_download_image_dataset", e)
            return False

    def transfer_datasets_and_start_training(self, dataset_list, model_hash_key):
        try:
            model_dir_path = os.path.dirname(os.path.realpath(__file__)) + train_data_dir + model_hash_key + '/'
            Constants.directory_validation(model_dir_path)
            number_of_samples = 0
            for dataset_name in dataset_list:
                # copy the files from the main directory to training directory. required for ImageDataGenerator utility of Keras
                source_dir_path = os.path.dirname(os.path.realpath(__file__)) + main_image_data_dir + dataset_name + '/'
                destination_dir_path = model_dir_path + dataset_name + '/'
                shutil.copytree(source_dir_path, destination_dir_path)

                training_directory_path = os.path.dirname(
                        os.path.realpath(__file__)) + train_data_dir + model_hash_key + '/' + dataset_name

                src_files = os.listdir(training_directory_path)

                # transfer 30% of the images to training folder
                count = 0
                for file_name in src_files:
                    percentage = 100 * (len(src_files) - count) / len(src_files)
                    if percentage <= 70:
                        break

                    full_file_name = os.path.join(training_directory_path, file_name)
                    testing_directory = os.path.dirname(
                            os.path.realpath(__file__)) + test_data_dir + model_hash_key + '/' + dataset_name + '/'
                    Constants.directory_validation(testing_directory)
                    if os.path.isfile(full_file_name):
                        shutil.copy(full_file_name, testing_directory + file_name)
                        os.remove(full_file_name)
                    else:
                        pass
                    count = count + 1

                # calculate the number of samples in training directory
                temp_samples = Constants.get_number_of_files(train_data_dir + model_hash_key + '/', dataset_name)
                if number_of_samples > 0 and number_of_samples < temp_samples:
                    pass
                else:
                    number_of_samples = temp_samples

            files = os.listdir(os.path.dirname(os.path.realpath(__file__)) + train_data_dir + model_hash_key + '/')
            number_of_classes = len(files)
            return True, number_of_samples, number_of_classes
        except Exception as e:
            cph.failure_print("transfer_datasets_and_start_training  ", e)
            return False, 0, 0

    def recieve_image_dataset_from_user(self, fileList, class_name):

        destination = os.path.dirname(os.path.realpath(__file__)) + main_image_data_dir + class_name + "/"

        for upload in fileList:
            filename = upload.filename.rsplit("/")[0]
            PrintHelper.info_print("Uploaded: ", filename)
            Constants.directory_validation(destination)
            upload.save(destination + filename)

        number_of_images = Constants.get_number_of_files(main_image_data_dir, class_name)
        mongo_entry = {
            'name' : class_name,
            'count': number_of_images
        }
        DatabaseFunctions().insert_item(imageDatasetConstants.database, imageDatasetConstants.collection,
                                        items_to_insert=mongo_entry)


class TestImageDownloader:
    """
    Test method for this class
    """

    def __init__(self):
        pass

    image_downloader = ImageDownloader()

    def test_download_images(self):
        """
        
        :return: 
        """
