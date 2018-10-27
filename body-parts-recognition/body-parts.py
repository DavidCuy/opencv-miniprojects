import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", dest='output', required=True, help="path to output video file")
parser.add_argument("-f", "--fps", dest='fps', type=int, default=20, help="FPS of output video")
parser.add_argument("-c", "--codec", dest='codec', type=str, default="MJPG", help="codec of output video")
parser.add_argument("-v", "--videocam", dest='videocam', type=int, default=0, help="number of video camara to use")
args_input = parser.parse_args()

protoFile = "body-parts-recognition/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "body-parts-recognition/pose/mpi/pose_iter_146000.caffemodel"
BODY_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
points_len = 15
BODY_PARTS = {
    "HEAD": 0,
    "NECK": 1,
    "R_SHOULDER": 2,
    "R_ELBOW": 3,
    "R_WRIST": 4,
    "L_SHOULDER": 5,
    "L_ELBOW": 6,
    "L_WRIST": 7,
    "R_HIP": 8,
    "R_KNEE": 9,
    "R_ANKLE": 10,
    "L_HIP": 11,
    "L_KNEE": 12,
    "L_ANKLE": 13,
    "CHEST": 14
}

def main():
    cap = cv2.VideoCapture(args_input.videocam)
    hasFrame, frame = cap.read()

    fouorcc = cv2.VideoWriter_fourcc(*args_input.codec)
    vid_writer = cv2.VideoWriter(args_input.output, fouorcc, args_input.fps, (frame.shape[1],frame.shape[0]))

    inWidth = int(frame.shape[1] / 2)
    inHeight = int(frame.shape[0] / 2)
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(points_len):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold : 
                #cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                #cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else :
                points.append(None)

        if points[BODY_PARTS['HEAD']] and points[BODY_PARTS['NECK']]:
            head_text = "HEAD"
            head_text_size = cv2.getTextSize(head_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
            x_middle = int( (points[BODY_PARTS['HEAD']][0] + points[BODY_PARTS['NECK']][0]) / 2 )
            x_middle = int( x_middle - (head_text_size[0] / 2) )
            y_middle = int( (points[BODY_PARTS['HEAD']][1] + points[BODY_PARTS['NECK']][1]) / 2 )
            cv2.putText(frame,head_text, (x_middle, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        
        if points[BODY_PARTS['R_ELBOW']]:
            arm_text = "ARM"
            arm_text_size = cv2.getTextSize(arm_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
            x_middle = int( points[BODY_PARTS['R_ELBOW']][0] - (arm_text_size[0] / 2) )
            y_middle = points[BODY_PARTS['R_ELBOW']][1]
            cv2.putText(frame, arm_text, (x_middle, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if points[BODY_PARTS['L_ELBOW']]:
            arm_text = "ARM"
            arm_text_size = cv2.getTextSize(arm_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
            x_middle = int( points[BODY_PARTS['L_ELBOW']][0] - (arm_text_size[0] / 2) )
            y_middle = points[BODY_PARTS['L_ELBOW']][1]
            cv2.putText(frame, arm_text, (x_middle, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        
        if points[BODY_PARTS['CHEST']]:
            body_text = "BODY"
            body_text_size = cv2.getTextSize(body_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
            x_middle = int( points[BODY_PARTS['CHEST']][0] - (body_text_size[0] / 2) )
            y_middle = points[BODY_PARTS['CHEST']][1]
            cv2.putText(frame, body_text, (x_middle, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if points[BODY_PARTS['L_KNEE']]:
            leg_text = "LEG"
            leg_text_size = cv2.getTextSize(leg_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
            x_middle = int( points[BODY_PARTS['L_KNEE']][0] - (leg_text_size[0] / 2) )
            y_middle = points[BODY_PARTS['L_KNEE']][1]
            cv2.putText(frame, leg_text, (x_middle, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if points[BODY_PARTS['R_KNEE']] and points[BODY_PARTS['R_ANKLE']]:
            leg_text = "LEG"
            leg_text_size = cv2.getTextSize(leg_text, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)[0]
            x_middle = int( (points[BODY_PARTS['R_KNEE']][0] + points[BODY_PARTS['R_ANKLE']][0]) / 2 )
            x_middle = int( x_middle - (leg_text_size[0] / 2) )
            y_middle = int( (points[BODY_PARTS['R_KNEE']][1] + points[BODY_PARTS['R_ANKLE']][1]) / 2 )
            cv2.putText(frame,leg_text, (x_middle, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        vid_writer.write(frame)
        cv2.imshow('Body Parts', frame)

    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
