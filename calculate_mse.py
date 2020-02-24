import cv2
import os
import numpy as np
import sys, traceback
sys.path.append('../../python')
from openpose import pyopenpose as op

import tensorflow as tf

MODEL_PATH = 'resources/data/generated_model'
BATCH_SIZE = 4096
EPOCHS = 300
CHECKPOINT_PATH='resources/trained/'

def generate_open_pose_params():
    global params
    params = dict()
    params["logging_level"] = 3
    params["model_pose"] = "BODY_25"
    params["model_folder"] = "../../../models/"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["keypoint_scale"] = 3
    params["process_real_time"] = True
    params["disable_blending"] = False


dir_path = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture('./resources/video/video_104.avi')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

files = len([name for name in os.listdir(CHECKPOINT_PATH) if os.path.isfile(os.path.join(CHECKPOINT_PATH, name))])
filename = (CHECKPOINT_PATH + "trained_" + str(files)) + ".h5"
with tf.compat.v1.Session(config=config) as sess:
    model = tf.keras.models.load_model(filename)

    generate_open_pose_params()
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    model.summary()

    means = []
    result = []
    index = 0
    while True:
        try:
            datum = op.Datum()
            _, frame = cap.read()

            if frame is None:
                break

            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            # Display Image
            points_of_interest = np.vstack((datum.poseKeypoints[0][0:10],
                                            datum.poseKeypoints[0][12],
                                            datum.poseKeypoints[0][15:19]))

            transformed_input = points_of_interest[:, :2].flatten().reshape(1, points_of_interest.shape[0] * 2)
            transformed_input = np.expand_dims(transformed_input, axis=0)
            result = model.predict(transformed_input)
            mse = np.mean(np.power(transformed_input[0]-result[0],2),axis=1)
            means.append(mse)
        ##    print(mse)
            index += 1
            if(index%1000 == 0):
                print(index)

        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            sys.exit(-1)

    print("max MSE:",np.max(means))
    print("min MSE:",np.min(means))
    print("98quant MSE:",np.quantile(means,0.98))
    print("95quant MSE:",np.quantile(means,0.95))
    print("90quant MSE:",np.quantile(means,0.90))
    cap.release()
    cv2.destroyAllWindows()

