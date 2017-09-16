class GoldStandardConstants:
	database = "gold_standards_database"
	collection = 'gold_standards_collection'
	gs_name = 'name'
	gs_original_file_name = 'original_file_name'
	gs_created_file_name = 'created_file_name'
	gs_language = 'language'
	gs_description = 'description'


class ModelMongoConstants:
	database = 'saved_models_database'
	collection = 'saved_models_collection'
	mm_model_hash = 'model_hash'
	mm_model_json = 'model_json'
	mm_model_search_terms = 'model_search_terms'
	mm_id = '_id'
	mm_model_h5_file = 'model_file_link'
	mm_model_weights_file = 'model_weights_file_link'
	mm_model_validation_status = 'model_validation_status'
	mm_model_base_name = 'model_base_name'
	mm_model_training_acc_loss = 'model_training_acc_loss'
	mm_model_saved_name = 'model_saved_name'


class ImageDatasetConstants:
	database = 'saved_images_database'
	collection = 'saved_images_collection'
	id_dataset_name = 'name'
	id_image_count_in_dataset = 'count'


class LexiconsConstants:
	database = 'saved_lexicons_database'
	collection = 'saved_lexicons_collection'
	lc_name = 'name'
	lc_file_name = 'file_name'
	lc_language = 'language'
	lc_user_keys = 'user_keys'
	lc_lexicons = 'lexicons'
