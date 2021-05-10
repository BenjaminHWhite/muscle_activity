# Classify muscle activity from Ca images of flies 

# Training 

The folder training_data contains 5 fly images of same X-Y dimension in h5 format. The intensity of the images indicate muscle activity. Each slice of the image is a time series acquired at 2Hz. Each image has an associated csv file which contains 9 types movements for each frame, AntComp, BotSwing, Brace, Crunch, Dgap, Lift, Pgap, Rest, Swing. Some frames have two types of undefined motions, denoted by either NaN or Twitch. The motions for each frame have been manually identified by an expert rater.

The training code **train.py** uses the training images and the ground truth motions for each frame to create a neural network model. The Inception-v3 [[1]](#1) network has been used to train the model. However, to fit the model in available GPU memory based on the large image size, the default Inception-v3 network in Keras  [[2]](#2) has been modified. The final Dense layer has been removed and replaced with a GlobalAveragePooling2D layer. In addition, a dropout layer was also added to introduce stochasticity into the model to avoid overfitting. The number of filters in each layer has also been reduced to keep the total number of parameters low in order to avoid overfitting.


## Requirements
```
Python 3.6 or later is required
tensorflow_gpu>=1.10.0
Keras>=2.2.5
```


## Training usage
```
python train.py -h
```
There are several arguments to the training script.

```
  --atlases ATLAS [ATLAS ...]
                        (Required) Atlas images to train. All atlases must have same
                        width and height, but might have differet number of
                        frames, as noted in the corresponding csv file.
                        Preferably use HDF5 .h5 files, because they will be
                        read multiple times to save runtime memory.
  --csv CSV [CSV ...]   (Required) CSV files with five columns, with frame numbers and
                        movement type (e.g. Switch, Rest etc).
  --nframe NFRAMES      (Required) Total number of frames to consider while training,
                        must be odd.
  --outdir OUTDIR       (Required) Output directory where the trained models are written.
  --discard DISCARD [DISCARD ...]
                        Discard this movement from training. Default is NaN.
                        E.g. --discard NaN Twitch. It is case sensitive.
  --basefilter BASEFILTER
                        Base filter for Inception v3 module. Default is 16,
                        which results in ~22million parameters.
  --gpuids GPU          Specifc GPUs to use for training, separated by comma.
                        E.g., --gpuids 3,4,5
  --initmodel INITMODEL
                        Existing pre-trained model. If provided, the weights
                        from the pre-trained model will be used to initiate
                        the training.
```                      

The following parameters have been used to train the network separately on each of the 5 training images. 
```
#!/bin/bash
ATLASIMG=training_data/X_sameT.h5    # X = 114, 115, 821, 917, 925
ATLASCSV=training_data/XLabels.csv   # X = 114, 115, 821, 917, 925
python train.py --atlases $ATLASIMG --csv $ATLASCSV --nframe 25 --outdir ./trained_models/ \
                     --gpuids 0,1,2,3 --discard NaN Twitch --basefilter 4 
```

To estimate motion at a particular frame, its previous 12 and next 12 frames (hence **--nframe 25**) are used. The code can train in parallel using multiple GPUs via **--gpuids** argument. The  NaN and Twitch from the prediction list since we do not consider them as a motion, rather a state. To reduce overfitting, **--basefilter 4** option is used to keep the number of trainable parameters low (~1.3 million). Five trained models are located in the folder **trained_models**.


# Prediction on new image
Two example images are provided. The test images must have same x-y dimension as the atlas images. However, number of slices (frames) can be different. 

To predict the motions in each frame of an image, the following command is used,
```
#!/bin/bash
IMAGE=test_data/X.h5       # Input test image
OUTPUT=test_data/prediction/X_prediction.csv  # Output csv file
MODEL=trained_model/Y.h5   # Trained models obtained from train.py
python test.py --image  ${IMAGE} --csv ${OUTPUT} --nframe 25 --model ${MODEL} --discard NaN Twitch  --gpuid 0
```

Using each of the 5 training images, the frame-wise predicted motions are located in **test_data/prediction/** folder.

# Training Ensembles

The training can be done using all training images, where a single model is generated and then used to predict motions on a new image. However, the number of training images is typically small. Also the movement of muscles between frames are highly correlated. Therefore, the network is prone to easy overfitting. To increase the generalizability of the model, we used an ensembling strategy instead of using all atlases at once. Five different models are generated using 5 atlases, then they are aggregated using **aggregate_csvs.m**.

## References
<a id="1">[1]</a> 
C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 2818-2826, doi: 10.1109/CVPR.2016.308.

<a id="2">[2]</a> 
https://keras.io/api/applications/inceptionv3/
