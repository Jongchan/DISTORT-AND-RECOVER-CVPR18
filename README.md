# DISTORT-AND-RECOVER-CVPR18
Code for "Distort-and-Recover: Color Enhancement with Deep Reinforcement Learning", CVPR18

## Overview

- You can pull&use the docker image from pjc0309/default_setup:latest to run the code. (TensorFlow 0.11.0rc, and other packages such as numpy/scipy)
- Before training,prepare MIT5K train/test images in separate folders (train/raw/, train/target/, test/raw/, test/target/). And edit the path in main.py accordingly.
- ```run_train.sh``` starts training.
- Use ```parse_test.py``` to parse the test results. (edit the paths accordingly)
- The training speed (iterations per second) should be between 20~40 it/sec. (When trained on i5-6600 and GTX 1080)

## Data

- Training data for expert C, resized to maximum side 500px, JPEG format. (including RANDOM250 list) [link](https://www.dropbox.com/s/0getrmsn1bktdop/C.zip?dl=0)
