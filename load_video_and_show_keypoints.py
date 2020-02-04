import sys
import cv2
import os

def generate_open_pose_params():
    global params
    params = dict()
    params["logging_level"] = 3
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["process_real_time"] = True
    params["disable_blending"] = False

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../../python')
from openpose import pyopenpose as op

cap = cv2.VideoCapture('./resources/video/video_104.avi')

generate_open_pose_params()

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
result = []
while(True):
    try:
        datum = op.Datum()
        _, frame = cap.read()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Display Image
        cv2.imshow("OpenPose 1.5.1", datum.cvOutputData)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        sys.exit(-1)

cap.release()
cv2.destroyAllWindows()

