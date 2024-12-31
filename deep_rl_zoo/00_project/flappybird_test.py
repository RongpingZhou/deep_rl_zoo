# Running with docker container with pytorch 2.5.1 and necessary cuda and cudnn

# docker run --gpus all -u $(id -u):$(id -g) -ti --rm -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw  -v $(realpath ~/mygit/rl/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_pytorch2-5-1:2.5.1

# docker run -u $(id -u):$(id -g) -ti --rm -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v $(realpath ~/mygit/rl/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_pytorch2-5-1:2.5.1

import time
import numpy as np
import sys
import statistics
sys.path.append("domain/")

import pygame
# import wrapped_flappy_bird as game
# import flappy_bird
from domain.flappy_bird import FlappyBirdEnv

import skimage
from skimage import transform, color, exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os

IMAGE_CHANNELS = 4

def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (85,84), mode = 'constant')
	image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	# image = image.reshape(1, image.shape[0], image.shape[1], 1)
	image = image.reshape(1, 1, image.shape[0], image.shape[1])
	return image

class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        # 4 input image channel, 16 output channels, 8x8 square convolution, 4x4 strides,
        # kernel
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 16, 8, stride=4)
        # 16 input image channel, 32 output channels, 4x4 square convolution, 2x2 strides,
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.flatten = nn.Flatten()
        # an affine operation: y = W * x + b
        # out features number is 256
        self.fc1 = nn.LazyLinear(256)
        self.actor = nn.Linear(256, 1)
        self.critic = nn.Linear(256, 1)

    def forward(self, input):
        # Convolution layer C1: 4 input image channel, 16 output channels,
        # IMAGE_ROWS = 85
        # IMAGE_COLS = 84
        # IMAGE_CHANNELS = 4
        # 8x8 square convolution, it uses RELU activation function, and 4x4 strides
        # outputs a Tensor with size (N, 16, 20, 20), where N is the size of the batch
        h0 = F.relu(self.conv1(input))
        # Convolution layer C3: 16 input channels, 32 output channels,
        # 4x4 square convolution, it uses RELU activation function, and 2x2 strides
        # outputs a (N, 32, 9, 9) Tensor
        h1 = F.relu(self.conv2(h0))
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        h2 = self.flatten(h1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        h3 = F.relu(self.fc1(h2))
        a1 = self.actor(h3)
        c1 = self.critic(h3)
        output1 = F.sigmoid(a1)
        return output1, c1

model = ActorCritic()
# model.load_state_dict(torch.load('champions/model_updates90000_best.pth', weights_only=True))
model.load_state_dict(torch.load('saved_models/model_updates90000.pth', weights_only=True))
model.eval()

env = FlappyBirdEnv(FPS = 300, render_mode = "human")

with torch.no_grad():

    currentScore = 0
    topScore = 0
    scores = np.array([])
    # a_t = [1,0]
    a_t = 0
    FIRST_FRAME = True

    terminal = False
    r_t = 0

    EPISODE = 0

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
                # run = False
                print("escape")
                env.close() 
                quit()

        if FIRST_FRAME:
            # x_t = game_state.getCurrentFrame()
            x_t = env.reset()
            x_t = preprocess(x_t)
            s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=1)
            FIRST_FRAME = False
            # print("FirstFrame")
        else:        
            # x_t, r_t, terminal = game_state.frame_step(a_t)
            x_t, r_t, terminal = env.step(a_t)
            x_t = preprocess(x_t)
            # s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)
            s_t = np.append(x_t, s_t[:, :3, :, :], axis=1)

        s_t_tensor = torch.FloatTensor(s_t)
        y, _ = model(s_t_tensor)
        # no = np.random.random()	
        # no = np.random.rand()
        # a_t = [0,1] if no < y[0] else [1,0]    #stochastic policy
        # a_t = [0,1] if 0.5 < y[0] else [1,0]   #deterministic policy
        a_t = 1 if 0.5 < y[0] else 0   #deterministic policy

        if(r_t == 1):
            currentScore += 1
            # print("topScore type is: ", type(topScore))
            # print("currentScore type is: ", type(currentScore))
            topScore = max(topScore, currentScore)
            # print("Current Score: " + str(currentScore) + " Top Score: " + str(topScore))
        if terminal == True:
            FIRST_FRAME = True
            terminal = False
            scores = np.append(scores, currentScore)
            currentScore = 0
            episodes = scores.shape[0]
            min_score = scores.min()
            max_score = scores.max()
            median = np.median(scores)
            average = np.mean(scores)           
            if episodes % 2 == 1:
                print("Total episodes are: ", episodes)
                print("min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " average: " + str(average))
            # if episodes > 100:
            if episodes > 2:
                break   

            # f = open("rewards.txt","a")
            # f.write("EPISODE: " + str(EPISODE) + ", currentScore: " + str(currentScore) + ", min: " + str(currentScore) + "\n")
            # f.close()
    
    # docker container is using UTC timezone, not Sydney timezone
    current_datetime = datetime.now()
    file_path = 'flappybird-data-'+current_datetime.strftime('%Y-%m-%d-%H-%M-%S')+'.npz'
    if os.path.exists(file_path):
        print("file exists, choose another file name")
    else:
        # Save the array and date/time information to a file
        np.savez(file_path, array=scores, datetime=str(current_datetime))

        # Load the existing data from the .npz file
        loaded_data = np.load(file_path)
        
        # Retrieve the existing array and datetime
        existing_array = loaded_data['array']
        existing_datetime = loaded_data['datetime']

        # print("Existing array:", type(existing_array))
        print("Existing date and time:", existing_datetime)
        print("Existing array:", existing_array)
        min_score = existing_array.min()
        max_score = existing_array.max()
        median = np.median(existing_array)
        
        average = np.mean(existing_array)
        
        print("min: " + str(min_score) + " max: " + str(max_score) + " median: " + str(median) + " average: " + str(average))

env.close()
time.sleep(2)

quit()
