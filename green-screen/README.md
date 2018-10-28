# Green screen
Example of use the green screen effect without a green screen. Only with a background learning, but a good illumination would helps much.

### Usage
```shell
usage: green-screen.py [-h] -o OUTPUT [-f FPS] [-c CODEC] [-v VIDEOCAM] [-l LEARN]

optional arguments:
  -h, --help                        show this help message and exit
  -o OUTPUT, --output OUTPUT        path to output video file
  -f FPS, --fps FPS                 FPS of output video
  -c CODEC, --codec CODEC           codec of output video
  -v VIDEOCAM, --videocam VIDEOCAM  number of video camara to use
  -l LEARN, --learn LEARN           learn time of background subtractor (in seconds)
```

* ```-o OUTPUT, --output OUTPUT``` is required
* ```-f FPS, --fps FPS``` default value 20
* ```-c CODEC, --codec CODEC``` default value MJPEG
* ```-v VIDEOCAM, --videocam VIDEOCAM``` default value 0
* ```-l LEARN, --learn LEARN``` default value 5