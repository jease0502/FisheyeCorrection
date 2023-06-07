import cv2
import numpy as np
import glob
import time
import os
import datetime
import copy

class FishEyeCalibration(object):
    def __init__(self):
        self.point = (6, 9)
        self.objp = np.zeros((1, self.point[0]*self.point[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:self.point[0], 0:self.point[1]].T.reshape(-1, 2)
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
        self.objpoints = []
        self.imgpoints = []
        self._img_shape = None

    def collect_images(self, num_images , camera_id = 0):
        #cap = cv2.VideoCapture(0) 
        cap = cv2.VideoCapture(camera_id, cv2.CAP_GSTREAMER)
        checkerboard = self.point
        count = 0
        video_frame = []

        while count < num_images:
            t1 = time.time()
            ret, frame = cap.read()
            #print("1 time" , time.time()-t1)
            self._img_shape = frame.shape[:2]
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
            ret, corners = cv2.findChessboardCorners(frame_gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH
                                                     + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            #print("2 time" , time.time()-t1)
            if ret :
                video_frame.append(copy.deepcopy(frame))
                self.objpoints.append(self.objp)
                cv2.cornerSubPix(frame_gray,corners,(3,3),(-1,-1),self.subpix_criteria)
                self.imgpoints.append(corners)
                count += 1
                

                # print("Images collected: ", count)
            cv2.putText(frame, "Images collected: {} / 500".format(count), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            #print(frame.shape)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.gray = frame_gray
        cap.release()
        cv2.destroyAllWindows()
        loc_dt = datetime.datetime.today() 
        loc_dt_format = loc_dt.strftime("%Y_%m_%d_%H_%M_%S")
        ca = camera_id.split("sensor_id=")[1].split("\n")[0]
        os.makedirs(os.path.join("calibration_images",ca + "_" + loc_dt_format), exist_ok=True)
	print(os.path.join("calibration_images",ca + "_" + loc_dt_format)
        for i, element in enumerate(video_frame):
            cv2.imwrite(os.path.join(os.path.join("calibration_images",ca + "_" + loc_dt_format) , str(i)) + ".jpg", element)

    def get_K_and_D(self):
        N_OK = len(self.objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = cv2.fisheye.calibrate(
        self.objpoints,
        self.imgpoints,
        self.gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        self.calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
        # DIM = self._img_shape[::-1]
        # print("Found " + str(N_OK) + " valid images for calibration")
        # print("DIM=" + str(self._img_shape[::-1]))
        # print("K=np.array(" + str(K.tolist()) + ")")
        # print("D=np.array(" + str(D.tolist()) + ")")
        # return DIM, K, D
       
if __name__ == '__main__':
    calibration = FishEyeCalibration()
    calibration.collect_images(num_images=500 , camera_id='''
            nvarguscamerasrc sensor_id=0
            ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)60/1
            ! nvvidconv
            ! video/x-raw, width=(int)360, height=(int)240, format=(string)BGRx
            ! videoconvert
            ! appsink
	''')
    # DIM, K, D = calibration.get_K_and_D()
    # np.save('DIM.npy', DIM)
    # np.save('K.npy', K)
    # np.save('D.npy', D)
    calibration2 = FishEyeCalibration()
    calibration2.collect_images(num_images=500 , camera_id='''
            nvarguscamerasrc sensor_id=1
            ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)60/1
            ! nvvidconv
            ! video/x-raw, width=(int)360, height=(int)240, format=(string)BGRx
            ! videoconvert
            ! appsink
	''')
    # DIM, K, D = calibration.get_K_and_D()
    # np.save('2DIM.npy', DIM)
    # np.save('2K.npy', K)
    # np.save('2D.npy', D)
