import os
from database import get_image_path

def test_get_image_path(self):
        database_path = '/Volumes/T7 Shield 1/Uni/4. Semester/Big Data Engineering/image_database.db'
        table_name = 'image_database_T7_1'   
        image_id = 1

        # Test the get_image_path function
        result = get_image_path(database_path, table_name, image_id)
        expected_result = "/Volumes/T7 Shield 1/Downloads/ILSVRC/Data/CLS-LOC/all_images/n03982430_20160.JPEG"
        assert result == expected_result


