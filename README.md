# Cloning Driving Behaviour Using Deep Learning

Based on Behaviour Cloning project from Term 1 of 
Udacity Self-Driving Car Engineer Nanodegree [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Overview
---
In this project we build a simple deep convolutional neural network (CNN)
to drive a car. It will learn from labeled data which is produced by someone
driving a car in the simulator. So it learns by cloning driving behaviour. 
This approach to programming self-driving cars is called end-to-end learning
because it takes raw sensor inputs (image pixels) 
and produces the resulting control commands
without breaking the intermediate steps down into logical steps like
detecting lanes, objects, distances etc.
[This article by NVIDIA](http://arxiv.org/pdf/1604.07316v1.pdf) details how 
it was done successfully on a real car using 
[NVIDIA Drive PX](https://en.wikipedia.org/wiki/Drive_PX-series)
and about 72 hours of driving in various conditions. They called their system DAVE 2 and you
can see [video of its performance here](https://drive.google.com/file/d/0B9raQzOpizn1TkRIa241ZnBEcjQ/view)
Notice around minute 8 what image they send to the CNN and the feature maps
it produces.

Here we repeat NVIDIA success to show that anyone can program self-driving cars
using open source tools!
We use the NVIDIA CNN architecture, code it up in Keras and then
use python to process telemetery events from the simulator and send the controls back.
The network takes image that 
a car sees in the simulator and produces the steering control.

The Simulator
---
Udacity produced an open-source driving simulator based on Unity 3D engine.
You can grab it [here (download Version 2 or above)](https://github.com/udacity/self-driving-car-sim) 

After you start the simulator choose the desired graphics quality and resolution.
Because later we will resize the images to 80x160 pixels, do not choose to high a resolution,
as the saved images will take up more space and lenghten the preprocessing but will not
improve the quiality of the resulting network.

Press the record button and choose the folder where the simulator creates
`IMG` folder and `driving-log.csv` files.
Then steer the car around the track for data collection. 
Here are a few points to consider:
* steering with the keyboard (left/right buttons) is quite sensitive, so you may want to
limit key press durations
* you can control steering via mouse instead of keyboard. 
This creates better angles for training. The angle is based on the mouse distance. 
To steer hold the left mouse button and move left or right. 
To reset the angle to 0 simply lift your finger off the left mouse button.
* You can toggle record by pressing `R` key (or press `record` button on the screen)
* When recording is finished, all the captured images are saved to the disk at the same time
You will see save status and play back of the captured data.
* You can takeover in autonomous mode. While `W` or `S` are held down you can control the 
car the same way you would in training mode. 
This can be helpful for debugging. As soon as `W` or `S` are let go simulator goes back
into autonomous mode.
* Pressing the `spacebar` in training mode toggles cruise control on and off
(effectively presses `W` for you).









The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.





### Dependencies

We use python 3.5 and [conda environment/package manager](http://conda.pydata.org/docs)

Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) or full Anaconda on your computer
2. Create a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html)
3. Each time you wish to work, activate your `conda` environment

---

## Installation

**Download** the version of `miniconda` that matches your system. Make sure you download the version for Python 3.5.

**NOTE**: There have been reports of issues creating an environment using miniconda `v4.3.13`. If it gives you issues try versions `4.3.11` or `4.2.12` from [here](https://repo.continuum.io/miniconda/).

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

**Create** your the `pydata-sdc` conda environment. 

```
conda env create -f environment.yml python=3.5
```

**Verify** that the carnd-term1 environment was created in your environments:

```sh
conda info --envs
```

**Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```sh
conda clean -tp
```

### Uninstalling 

To uninstall the environment:

```sh
conda env remove -n pydata-sdc
```

---

## Using Anaconda

Now that you have created an environment, in order to use it, you will need to activate the environment. This must be done **each** time you begin a new working session i.e. open a new terminal window. 

**Activate** the `pydata-sdc` environment:

### OS X and Linux
```sh
$ source activate pydata-sdc
```
### Windows
Depending on shell either:
```sh
$ source activate pydata-sdc
```
or

```sh
$ activate pydata-sdc
```

Install tensorflow.

```sh
$ pip install tensorflow-gpu==1.0.0
```

or, if you dont have an NVIDIA GPU and CUDA drivers: 

```sh
$ pip install tensorflow==1.0.0
```




## Resources

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
