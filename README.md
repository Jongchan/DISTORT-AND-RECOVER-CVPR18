# DISTORT-AND-RECOVER-CVPR18
Code for "Distort-and-Recover: Color Enhancement with Deep Reinforcement Learning", CVPR18

## Overview

- You can pull&use the docker image from pjc0309/default_setup:latest to run the code. (TensorFlow 0.11.0rc, and other packages such as numpy/scipy)
- Before training,prepare MIT5K train/test images in separate folders (train/raw/, train/target/, test/raw/, test/target/). And edit the path in main.py accordingly.
- ```run_train.sh``` starts training.
- Use ```parse_test.py``` to parse the test results. (edit the paths accordingly)
- The training speed (iterations per second) should be between 20~40 it/sec. (When trained on i5-6600 and GTX 1080)

## Data

- MIT5K Train/Val(RANDOM250) images. Resized to maximum side 500px, JPEG format. (including RANDOM250 list) [LINK](https://www.dropbox.com/sh/web5of2dswd55b3/AABs5xY3V1CXEzfGWzBw9OUQa?dl=0)
