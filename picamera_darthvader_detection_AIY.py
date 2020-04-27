"""
Darth Vader detector

Code for testing custom AIY Vision compiled models. Example of usage:
$ python3 picamera_advanced_example_AIY.py --model_path $(pwd)/compiled.binaryproto --input_width 256 --input_height 256

"""
import io
import os
import sys
import time
import numpy as np

from PIL import Image


from aiy.vision.inference import ImageInference, ModelDescriptor
from aiy.vision.models import utils

from cognifly_AIY_utils import process_output_tensor, draw_boxes_raw


def tensors_info(tensors):
    return ', '.join('%s [%d elements]' % (name, len(tensor.data))
        for name, tensor in tensors.items())


def crop_center(image):
    width, height = image.size
    size = min(width, height)
    x, y = (width - size) / 2, (height - size) / 2
    return image.crop((x, y, x + size, y + size)), (x, y)

class ImgCap(io.IOBase):
    '''
    Capturing Image from a Raspicam (V2.1)
    '''
    def __init__(self, model, frameWidth=240, frameHeight=240, DEBUG = False):
        # Init the stuff we are inheriting from
        super().__init__()

        self.inference = ImageInference(model)

        self.ANCHORS = np.genfromtxt("/opt/aiy/models/mobilenet_ssd_256res_0.125_person_cat_dog_anchors.txt")

        self.DEBUG = DEBUG

        # Set video frame parameters
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight

        self.prev_time = time.time()

        self.output = None

    def writable(self):
        '''
        To be a nice file, you must have this method
        '''
        return True

    def write(self, b):
        '''
        Here is where the image data is received and made available at self.output
        '''

        try:
            # b is the numpy array of the image, 3 bytes of color depth
            self.output = np.reshape(np.frombuffer(b, dtype=np.uint8), (self.frameHeight, self.frameWidth, 3))

            image_center, offset = crop_center(Image.fromarray(self.output))
            result = self.inference.run(image_center)

            # The weird shapes used for concat and concat_1 are just copying the output tensors
            # shapes when using the model directly from tensorflow.
            # => #classes: number of classes during training (the number used in the config file)
            concat = np.asarray(result.tensors['concat'].data).reshape((1, 1278, 1, 4)) #(1, 1278*#classes, 1, 4)
            concat_1 = np.asarray(result.tensors['concat_1'].data).reshape((1, 1278, 2)) #(1, 1278, #classes + 1)

            detection_boxes, detection_scores, detection_classes = process_output_tensor(concat, 
                                                                                         concat_1, 
                                                                                         self.ANCHORS, 
                                                                                         classes=[1], 
                                                                                         IoU_thres=0.5, 
                                                                                         raw_boxes=False, 
                                                                                         score_threshold=0.3)
            if self.DEBUG:
                print(f"Boxes: {detection_boxes}")
                print(f"Scores: {detection_scores}")
                print("ImgCap - Inference done!")
                print("ImgCap - Image.shape {}".format(self.output.shape))
                print("ImgCap - Running at {:2.2f} Hz".format(1/(time.time()-self.prev_time)))

            self.prev_time = time.time()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_obj, exc_tb.tb_lineno)
            print("ImgCap error: {}".format(e))

        finally:
            return len(b)


if __name__ == "__main__":

    import picamera


    DEBUG = True

    # See https://picamera.readthedocs.io/en/release-1.10/api_camera.html
    # for details about the parameters:
    frameWidth = 256 
    frameHeight = 256
    frameRate = 20
    contrast = 40
    rotation = 180
        
    # Set the picamera parametertaob
    camera = picamera.PiCamera()
    camera.resolution = (frameWidth, frameHeight)
    camera.framerate = frameRate
    camera.contrast = contrast

    model = ModelDescriptor(
        name="DarthVaderDetector",
        input_shape=(1, 256, 256, 3),
        input_normalizer=(128.0, 128.0),
        compute_graph=utils.load_compute_graph(os.path.join(os.getcwd(), "darthvader.binaryproto")))

    # Start the video process
    with ImgCap(model, frameWidth, frameHeight, DEBUG) as img:
        camera.start_recording(img, format='rgb', splitter_port = 1)
        try:
            while True:
                camera.wait_recording(timeout=0) # using timeout=0, default, it'll return immediately  
                # if img.output is not None:
                    # print(img.output[0,0,0])

        except KeyboardInterrupt:
            pass
        finally:
            camera.stop_recording(splitter_port = 1)
            # camera.stop_recording(splitter_port = 2)
