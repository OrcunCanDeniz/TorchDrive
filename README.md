# TorchDrive
###### Work in progress. 

This repository aims to implement [NVIDIA's Behavioral Cloning Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) using PyTorch. 

As the source of the data [Udacity Self Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) is utilized since it seamlessly lets gathering the type of data we need. Term 1 Version 2 is used for development of this repository, so it is recommended. 

#### Data Collection

 1. Launch the simulator and proceed to "Training Mode".
 2. Hit "R" key to start recording your drive data and chose the directory which the data is going to be saved to. It is recommended to drive minimum 10 mins.
 3. When you're done with your data collection step hit "R" key again. And wait for data capture to end.
 
 Above steps going to create a csv file and a folder full of "dashcam" images. 

#### Training

Run training script as below;  

    python3 Train.py -d <path_to_data_csv>
				     -t <training_set_fraction>
				     -e <epochs>
				     -b <batch_size>
				     -l <learning_rate>

 
 Thorought out your training, validation will also be performed and the model with the best validation loss will be saved as "driver_best.pt" .

#### Drive !

As you finished your training, you now can see your model drive.

 1. Launch the simulator in "Autonomous Mode".
 2. Then run `python3 drive_torch <path_to_model>`


###### This repo is created by referring this [work](https://github.com/naokishibuya/car-behavioral-cloning).


