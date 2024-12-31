# Running with docker container with pytorch 2.5.1 and necessary cuda and cudnn

# docker run --gpus all -u $(id -u):$(id -g) -ti --rm -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw  -v $(realpath ~/mygit/rl/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_pytorch2-5-1:2.5.1

# docker run -u $(id -u):$(id -g) -ti --rm -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /dev/snd:/dev/snd:rw -v $(realpath ~/mygit/rl/):/rl/ -e DISPLAY=unix$DISPLAY -p 8888:8888 --privileged ubuntu2204_cuda12-4-1_cudnn9-1-0-70-1_pytorch2-5-1:2.5.1

import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pygame
import sys
sys.path.append("domain/")
from domain.flappy_bird import FlappyBirdEnv

import skimage
from skimage import transform, color, exposure

import math

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# game_state = game.GameState(30000)
env = FlappyBirdEnv(FPS = 300, render_mode = "human")

GAMMA = 0.99                #discount value
IMAGE_ROWS = 85
IMAGE_COLS = 84
IMAGE_CHANNELS = 4
LEARNING_RATE = 7e-4
t_max = 5  
const = 1e-5
T = 0
a_t = 0

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
        h1 = F.relu(self.conv2(h0))
        h2 = self.flatten(h1)
        h3 = F.relu(self.fc1(h2))
        a1 = self.actor(h3)
        c1 = self.critic(h3)
        output1 = F.sigmoid(a1)
        return output1, c1

model = ActorCritic()
print(model)
# create your optimizer
optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, eps=0.1)

#loss function for policy output
def logloss(y_true, y_pred):     #policy loss
	return -torch.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const)

#loss function for critic output
def sumofsquares(y_true, y_pred):        #critic loss
	return torch.square(y_pred - y_true)

#function to preprocess an image before giving as input to the neural network
def preprocess(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image, (IMAGE_ROWS, IMAGE_COLS), mode = 'constant')	
	image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
	image = image.reshape(1, 1, image.shape[0], image.shape[1])
	return image

def runprocess(s_t):
	global T
	global a_t
	global model
	global sess
	global graph
	
	t = 0
	t_start = t
	terminal = False
	r_t = 0
	r_store = []
	state_store = np.zeros((0, IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS))
	output_store = []
	critic_store = []

	while t-t_start < t_max and terminal == False:
		s_t_tensor = torch.FloatTensor(s_t)

		t += 1
		T += 1
		out, _ = model(s_t_tensor)
		no = np.random.rand()
		# a_t = [0,1] if no < out else [1,0]  #stochastic action
		a_t = 1 if no < out else 0  #stochastic action

		# x_t, r_t, terminal = game_state.frame_step(a_t)
		x_t, r_t, terminal = env.step(a_t)

		x_t = preprocess(x_t)
		
		_, critic_reward = model(s_t_tensor)

		# y = 0 if a_t[0] == 1 else 1
		y = 0 if a_t == 0 else 1

		r_store = np.append(r_store, r_t)
		
		state_store = np.append(state_store, s_t, axis = 0)
		output_store = np.append(output_store, y)
		critic_store = np.append(critic_store, critic_reward.detach().numpy())
		
		s_t = np.append(x_t, s_t[:, :3, :, :], axis=1)

	if terminal == False:
		r_store[len(r_store)-1] = critic_store[len(r_store)-1]
	else:
		r_store[len(r_store)-1] = -1
		s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=1)
	
	for i in range(2,len(r_store)+1):
		r_store[len(r_store)-i] = r_store[len(r_store)-i] + GAMMA*r_store[len(r_store)-i + 1]

	return s_t, state_store, output_store, r_store, critic_store

#function to decrease the learning rate after every epoch. In this manner, the learning rate reaches 0, by 20,000 epochs
def step_decay(epoch):
	decay = 3.2e-8
	lrate = LEARNING_RATE - epoch*decay
	lrate = max(lrate, 0)
	return lrate

def main():

    episode_r = []
    episode_state = np.zeros((0, IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS))
    episode_output = []
    episode_critic = []
    EPISODE = 0

    image = env.reset()

    image = preprocess(image)
    state = np.concatenate((image, image, image, image), axis=1)
    # state = np.concatenate((image, image, image, image), axis=3)
    # print("state shape is ", state.shape)

    while True:	

        next_state = state
        print("state shape is ", state.shape)
        # next_state, state_store, output_store, r_store, critic_store = runprocess(next_state)
        next_state, state_store, output_store, r_store, critic_store = runprocess(state)
        print("next_state shape is ", next_state.shape)
        # print("r_store is ", r_store)
        next_state = next_state.reshape(next_state.shape[1], next_state.shape[2], next_state.shape[3])
        # print("next_state shape is ", next_state.shape)

        episode_r = np.append(episode_r, r_store)
        episode_output = np.append(episode_output, output_store)
        # print("episode_output: ", episode_output)
        # print("episode_state shape is ", episode_state.shape)
        episode_state = np.append(episode_state, state_store, axis = 0)

        episode_critic = np.append(episode_critic, critic_store)

        state = next_state
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])
        
        # print("episode_r: ", episode_r)
        # print("episode_critic: ", episode_critic)
        # advantage calculation for each action taken

        # R+gamma*v(S) episode_r -V(S')
        advantage = episode_r - episode_critic
        print("advantage: ", advantage)
        # print("backpropagating")
            
        # weights = {'o_P':advantage, 'o_V':np.ones(len(advantage))}   
        # backpropagation
        # history = model.fit(episode_state, [episode_output, episode_r], epochs = EPISODE + 1, batch_size = len(episode_output), callbacks = callbacks_list, sample_weight = weights, initial_epoch = EPISODE)

        # in your training loop:
        state_minibatch = torch.FloatTensor(episode_state)

        action, value = model(state_minibatch)
        # print("action type: ", action.type())
        # print("value type: ", value.type())
        # print("advantage type: ", type(advantage))
        # print("episode_output type: ", type(episode_output))
        # print("episode_r type: ", type(episode_r))
        target_action = torch.FloatTensor(episode_output)
        advantage_tensor = torch.FloatTensor(advantage)
        
        # print("advantage * episode_output: ", advantage * episode_output)
        # target_action = torch.FloatTensor(episode_output)
        target_value = torch.FloatTensor(episode_r)
        np_ones_tensor = torch.FloatTensor(np.ones(len(advantage)))

        actor_loss = torch.mean(advantage_tensor * logloss(target_action, action))
        critic_loss = torch.mean(np_ones_tensor * sumofsquares(target_value, value))
        # print("actor_loss type: ", type(actor_loss))
        # print("critic_loss type: ", type(critic_loss))
        # print("actor_loss shape is ", actor_loss.shape)
        # print("critic_loss shape is ", critic_loss.shape)
        loss = actor_loss + 0.5 * critic_loss
        # print("loss shape is ", loss.shape)
        optimizer.zero_grad()   # zero the gradient buffers
        loss.backward()
        optimizer.step()    # Does the update

        # lrate = LearningRateScheduler(step_decay)
        # callbacks_list = [lrate]
        new_lr = step_decay(EPISODE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        episode_r = []
        episode_output = []
        # episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
        episode_state = np.zeros((0, IMAGE_CHANNELS, IMAGE_ROWS, IMAGE_COLS))
        episode_critic = []
        # states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))
        
        # f = open("rewards.txt","a")
        # f.write("Update: " + str(EPISODE) + ", Reward_mean: " + str(e_mean) + ", Loss: " + str(history.history['loss']) + "\n")
        # f.close()
        print(f"EPISODE: {EPISODE:>5d}")
        if EPISODE % 50 == 0:
            # print(f"EPISODE: {EPISODE:>5d}")
            print(f"loss: {loss:>7f} actor_loss: {actor_loss:>7f} critic_loss: {critic_loss:>7f}")
        if EPISODE % 5000 == 0:
            # model.save("saved_models/model_updates" +	str(EPISODE))
            torch.save(model.state_dict(), "saved_models/model_updates" + str(EPISODE) + ".pth")
            
        EPISODE += 1

        if EPISODE > 100000:
            break

        # if stop_event.is_set():
        #     print("stop event")
        #     sensor_ps.terminate()
        #     time.sleep(0.5)
        #     if not sensor_ps.is_alive():
        #         sensor_ps.join(1.0)
        #         print("[MAIN]: joined process successfully!")
        #     clear_pipe(parent_conn)
        #     print("clear parent pipe")
        #     break

    env.close()
    time.sleep(2)
    print("quit pygame")
    # # Release the capture device and close all windows
    # cap.release()
    # cv2.destroyAllWindows()
    print("before exit")
    exit(0)

if __name__ == '__main__':
    main()