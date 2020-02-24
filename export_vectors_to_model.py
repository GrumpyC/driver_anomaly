import cv2
import os
import numpy as np
import sys, traceback
sys.path.append('../../python')
from openpose import pyopenpose as op


def generate_open_pose_params():
    global params
    params = dict()
    params["logging_level"] = 3
    params["model_pose"] = "BODY_25"
    params["model_folder"] = "../../../models/"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["keypoint_scale"] = 3
    params["scale_number"] = 1
    params["process_real_time"] = True
    params["disable_blending"] = False


dir_path = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture('./resources/video/video_104.avi')

generate_open_pose_params()
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

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
        points_of_interest = np.vstack((datum.poseKeypoints[0][0:10], datum.poseKeypoints[0][12], datum.poseKeypoints[0][15:19]))
        result.append(points_of_interest)

        #TODO Uncomment this part if you need to test
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        #cv2.imshow("OpenPose 1.5.1", datum.cvOutputData)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

        index += 1
        if(index%1000 == 0):
            print(index)

    except Exception as e:
        print(e)
        traceback.print_exc(file=sys.stdout)
        sys.exit(-1)

cap.release()
cv2.destroyAllWindows()

np.save('./resources/model/video_104.npy', np.array(result))