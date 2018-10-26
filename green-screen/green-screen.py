import numpy as np
import cv2
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", dest='output', required=True, help="path to output video file")
parser.add_argument("-f", "--fps", dest='fps', type=int, default=20, help="FPS of output video")
parser.add_argument("-c", "--codec", dest='codec', type=str, default="MJPG", help="codec of output video")
parser.add_argument("-v", "--videocam", dest='videocam', type=int, default=0, help="number of video camara to use")
parser.add_argument("-l", "--learn", dest='learn', type=int, default=5, help="learn time of background subtractor (in seconds)")
args_input = parser.parse_args()

def main():
    cap = cv2.VideoCapture(args_input.videocam)
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    fouorcc = cv2.VideoWriter_fourcc(*args_input.codec)

    hasFrame, frame = cap.read()
    vid_writer = cv2.VideoWriter(args_input.output, fouorcc, args_input.fps, (frame.shape[1],frame.shape[0]))

    calibrateFlag = False
    initTime = time.time()
    calibratingText = "Learning background..."
    calibratingTextsize = cv2.getTextSize(calibratingText, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

    while True:
        _, frame = cap.read()

        if calibrateFlag == False:
            bgSubtractor.apply(frame, learningRate=0.5)
            calibratingTextX = int((frame.shape[1] - calibratingTextsize[0]) / 2)
            calibratingTextY = int((frame.shape[0] + calibratingTextsize[1]) / 2)
            cv2.putText(frame, calibratingText, (calibratingTextX, calibratingTextY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            timeStr = "Time elapsed: {:.1f} s".format(time.time() - initTime)
            timeTextsize = cv2.getTextSize(timeStr, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            timeTextX = int(frame.shape[1] - timeTextsize[0])
            timeTextY = int(frame.shape[0] - timeTextsize[1])
            cv2.putText(frame, timeStr, (timeTextX, timeTextY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            fgMask = bgSubtractor.apply(frame, learningRate=0)
            img_fg = cv2.bitwise_and(frame, frame, mask = fgMask)
            cv2.imshow('BGSUB', img_fg)
            vid_writer.write(img_fg)


        cv2.imshow('Video', frame)

        calibrateFlag = True if (time.time() - initTime > args_input.learn) else False

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()