import numpy as np
import time
import sys
import random

from domain.make_env import make_env
from .ind import *
import gymnasium as gym
# import gym

import torch

class Task():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """
    # Network properties
    # print("game is: ", game)
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers      
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]

    # Environment
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    # print(f"inside Task game.env_name is {game.env_name}")

    if not paramOnly:
      self.env = make_env(game.env_name)
    #   self.env = gym.make(game.env_name, render_mode='human')
    #   self.env = gym.make(game.env_name)

    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))
    print("task.py: __init__(): end: paramOnly: ", paramOnly) 
  
  def testInd(self, wVec, aVec, view=False,seed=-1):
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    # self.env.action_space.seed(42)
    if seed >= 0:
      random.seed(seed)
      np.random.seed(seed)
    #   state, _ = self.env.reset(seed=seed)
      self.env.seed(seed)
    # else:
    #   state, _ = self.env.reset()
    state = self.env.reset()
    # print("task.py(): Task: testInd(): after env.reset(): type of state: ", type(state))
    # print("task.py(): Task: testInd(): after env.reset(): type of self.nInput: ", type(self.nInput))
    # print("task.py(): Task: testInd(): after env.reset(): type of self.nOutput: ", type(self.nOutput))
    # print("task.py(): Task: testInd(): after env.reset(): wVec type:", np.shape(wVec), " wVec: ")
    # print(wVec)
    # print("task.py(): Task: testInd(): after env.reset(): aVec shape:", np.shape(aVec), " aVec: ")
    # print(aVec)

    self.env.t = 0

    # annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
    wVec_tensor = torch.from_numpy(wVec)
    aVec_tensor = torch.from_numpy(aVec)
    state_tensor = torch.from_numpy(state)
    nInput_tensor = torch.tensor(self.nInput)
    nOutput_tensor = torch.tensor(self.nOutput)
    annOut = act(wVec_tensor, aVec_tensor, nInput_tensor, nOutput_tensor, state_tensor)
    action = selectAct(annOut,self.actSelect)    
    # print(self.env.step(action))
    state, reward, done, info = self.env.step(action)
    # state, reward, done, _, info = self.env.step(action)
    if self.maxEpisodeLength == 0:
      return reward
    else:
      totalReward = reward
    
    # print("task.py(): Task: testInd(): before for tStep in range(self.maxEpisodeLength): ")
    for tStep in range(self.maxEpisodeLength): 
    #   annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
    #   wVec_tensor = torch.from_numpy(wVec)
    #   aVec_tensor = torch.from_numpy(aVec)
      state_tensor = torch.from_numpy(state)
    #   nInput_tensor = torch.tensor(self.nInput)
    #   nOutput_tensor = torch.tensor(self.nOutput)
      annOut = act(wVec_tensor, aVec_tensor, nInput_tensor, nOutput_tensor, state_tensor) 
      action = selectAct(annOut,self.actSelect) 
      state, reward, done, info = self.env.step(action)
    #   state, reward, done, _, info = self.env.step(action)
      totalReward += reward  
      if view:
        #time.sleep(0.01)
        # print("task.py(): Task: testInd(): view: ", view)
        if self.needsClosed:
        #   print("task.py(): Task: testInd(): if self.needsClosed: ", self.needsClosed)
          self.env.render(close=done)
        else:
        #   print("task.py(): Task: testInd(): if self.needsClosed: else: ")
          self.env.render()
      if done:
        break

    return totalReward

# -- 'Weight Agnostic Network' evaluation -------------------------------- -- #
  def setWeights(self, wVec, wVal):
    """Set single shared weight of network
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      wVal    - (float)    - value to assign to all weights
  
    Returns:
      wMat    - (np_array) - weight matrix with single shared weight
                [N X N]
    """
    # Create connection matrix
    wVec[np.isnan(wVec)] = 0
    dim = int(np.sqrt(np.shape(wVec)[0]))    
    cMat = np.reshape(wVec,(dim,dim))
    cMat[cMat!=0] = 1.0

    # Assign value to all weights
    wMat = np.copy(cMat) * wVal 
    return wMat


  def getDistFitness(self, wVec, aVec, hyp, \
                    seed=-1,nRep=False,nVals=6,view=False,returnVals=False):
    """Get fitness of a single individual with distribution of weights
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
      hyp     - (dict)     - hyperparameters
        ['alg_wDist']        - weight distribution  [standard;fixed;linspace]
        ['alg_absWCap']      - absolute value of highest weight for linspace
  
    Optional:
      seed    - (int)      - starting random seed for trials
      nReps   - (int)      - number of trials to get average fitness
      nVals   - (int)      - number of weight values to test

  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = hyp['alg_nReps']

    # Set weight values to test WANN with
    if (hyp['alg_wDist'] == "standard") and nVals==6: # Double, constant, and half signal 
      wVals = np.array((-2,-1.0,-0.5,0.5,1.0,2))
    else:
      wVals = np.linspace(-self.absWCap, self.absWCap ,nVals)


    # Get reward from 'reps' rollouts -- test population on same seeds
    reward = np.empty((nRep,nVals))
    # print("reward is: ", reward)
    for iRep in range(nRep):
      for iVal in range(nVals):
        wMat = self.setWeights(wVec,wVals[iVal])
        # np.set_printoptions(threshold=np.inf)
        # print("wMat shape: ", wMat.shape)
        # print(wMat)
        if seed == -1:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=seed,view=view)
        else:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=seed+iRep,view=view)
          
    if returnVals is True:
      return np.mean(reward,axis=0), wVals
    return np.mean(reward,axis=0)
 
