# Body pose recognition
For this example I based in the next entry of blog ```https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/``` where it draws a skeleton throug a human body. I modified the example to detect and recognized body parts. Maybe it's not the best way to recognized it but, the code does the job.

### Comments
* The caffemodel are very large files, so here's the link to download them
    * [COCO MODEL](http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel)
    * [MPI MODEL]( http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_146000.caffemodel) ```I used this in the example```
* If you want to know the relation of the models, points recognized and body parts, the file ```points_body_equivalents.md``` will be useful

### Usage
```shell
usage: body-parts.py [-h] -o OUTPUT [-f FPS] [-c CODEC] [-v VIDEOCAM]

optional arguments:
  -h, --help                        show this help message and exit
  -o OUTPUT, --output OUTPUT        path to output video file
  -f FPS, --fps FPS                 FPS of output video
  -c CODEC, --codec CODEC           codec of output video
  -v VIDEOCAM, --videocam VIDEOCAM  number of video camara to use
```

* ```-o OUTPUT, --output OUTPUT``` is required
* ```-f FPS, --fps FPS``` default value 20
* ```-c CODEC, --codec CODEC``` default value MJPEG
* ```-v VIDEOCAM, --videocam VIDEOCAM``` default value 0