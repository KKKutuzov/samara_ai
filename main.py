import pandas as pd
import argparse
import glob
import os
import tensorflow
import sys
import time
import random
import warnings

import humanfriendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageFont, ImageDraw
import statistics
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
from tqdm import tqdm
from CameraTraps.ct_utils import truncate_float
from model import Classifier
from utils import get_inference_transform

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

print('TensorFlow version:', tf.__version__)
print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())

path_to_weights = None  # soon
transform = get_inference_transform()
model = Classifier(path_to_weights)
device = 'cpu'
def open_image(input_file):
    """
    Opens an image in binary format using PIL.Image and convert to RGB mode. This operation is lazy; image will
    not be actually loaded until the first operation that needs to load it (for example, resizing), so file opening
    errors can show up later.
    Args:
        input_file: an image in binary format read from the POST request's body or
            path to an image file (anything that PIL can open)
    Returns:
        an PIL image object in RGB mode
    """

    image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError('Input image {} uses unsupported mode {}'.format(input_file, image.mode))
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    return image

def load_image(input_file):
    """
    Loads the image at input_file as a PIL Image into memory; Image.open() used in open_image() is lazy and
    errors will occur downstream if not explicitly loaded
    Args:
        input_file: an image in binary format read from the POST request's body or
            path to an image file (anything that PIL can open)
    Returns:
        an PIL image object in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image

class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions    '
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings
class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    the MegaDetector (TF). The inference batch size is set to 1; code needs to be modified
    to support larger batch sizes, including resizing appropriately.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # An enumeration of failure reasons
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '4': 'vehicle'  # will be available in megadetector v4
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads the model at model_path and start a tf.Session with this graph. The necessary
        input and output tensor handles are obtained also."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(np_array):
        """ Two effects: convert the numpy floats to Python floats, and also change the coordinates from
        [y1, x1, y2, x2] to [x1, y1, width_box, height_box] (in relative coordinates still).
        Args:
            np_array: array of predicted bounding box coordinates from the TF detector
        Returns: array of predicted bounding box coordinates as Python floats and in [x1, y1, width_box, height_box]
        """
        # change from [y1, x1, y2, x2] to [x1, y1, width_box, height_box]
        width_box = np_array[3] - np_array[1]
        height_box = np_array[2] - np_array[0]

        new = [np_array[1], np_array[0], width_box, height_box]  # cannot be a numpy array; needs to be a list

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.
        Args:
            model_path: .pb file of the model.
        Returns: the loaded graph.
        """
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id,
                                      detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.
        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the `file` field of the output object
            detection_threshold: confidence above which to include the detection proposal
        Returns:
        A dict with the following fields, see https://github.com/microsoft/CameraTraps/tree/siyu/inference_refactor/api/batch_processing#batch-processing-api-output-format
            - image_id (always present)
            - max_detection_conf
            - detections, which is a list of detection objects containing `category`, `conf` and `bbox`
            - failure
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result
def load_and_run_detector(model_file, image_file_names, output_dir,
                          render_confidence_threshold=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD):
    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    # load and run detector on target images, and visualize the results
    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    detection_results = []
    time_load = []
    time_infer = []
    detection_categories = []

    # since we'll be writing a bunch of files to the same folder, rename
    # as necessary to avoid collisions
    output_file_names = {}

    for im_file in tqdm(image_file_names):
        try:
            start_time = time.time()

            image = load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)
            # print(time_load)
        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            result = tf_detector.generate_detections_one_image(image, im_file)
            
            #print("Detection result is:", result)
            
            detection_results.append(result)
            
            if result["detections"] == []:
                detection_categories.append(0)
            else:
                    
                detection_categories.append(result["detections"][0]["category"])

            elapsed = time.time() - start_time
            time_infer.append(elapsed)
        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(im_file, e))
            # the error code and message is written by generate_detections_one_image,
            # which is wrapped in a big try catch
            continue

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available'
        std_dev_time_infer = 'not available'
    print('On average, for each image,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))
    
    return detection_results

model_file = "./md_v4.1.0.pb"


def predict(image_path):
  imgs = list(map(lambda x: image_path + '/' + x, list(filter(lambda x: x!= '.ipynb_checkpoints',os.listdir('/content/test')))))
  labels = []
  for img_path in imgs:
    img =  imread(img_path)
    torch_img = transform(img).to(device).unsqueeze(0)
    clss = int(model(torch_img).argmax(-1) + 1)
    labels.append(clss)
  pd.DataFrame({'id': imgs,
     'class': labels,
    }).to_csv('labels.csv', index=False)

if __name__ == '__main__':
    print('\n\n\nВведите название дириектории с фотографиями:\n')
    predict(input())