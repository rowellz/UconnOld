import numpy as np
import cv2 as cv2
import glob

def calibreate_cameras(img_folder_left, img_folder_right):

    imagesLeft = sorted(glob.glob(img_folder_left + '*.jpg'))
    imagesRight = sorted(glob.glob(img_folder_right + '*.jpg'))

    cv_image = cv2.imread(imagesLeft[0])
    shape = cv_image.shape
    width = shape[1]
    height = shape[0]

    print("calibrating for dimensions of width: ", width, ' height: ', height)

    chessboardSize = (9,6)
    frameSize = (width,height)   #opencv works best with small(ish) images.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    objp = objp * 26 #define the real life size of checkerboard squares (26mm)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    imgpointsR = [] # 2d points in image plane.

    for imgLeft, imgRight in zip(imagesLeft, imagesRight):

        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            print(imgLeft, ' and ', imgRight, 'are callibration pairs.')

            objpoints.append(objp)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv2.imshow('img left', imgL)
            cv2.waitKey(0)
            cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv2.imshow('img right', imgR)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    ############## CALIBRATION #######################################################

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, channelsL = imgL.shape
    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, channelsR = imgR.shape
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

    ########## Stereo Vision Calibration #############################################

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same 

    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

    print("Saving parameters!")

    cv_file = cv2.FileStorage('./StereoMaps/stereoMap.xml', cv2.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
    cv_file.write('q',Q) #The Q matrix is the necessary data structure for 3d Reconstruction

    cv_file.release()


calibreate_cameras('./images/stereoLeft/RESIZED/', './images/stereoRight/RESIZED/')