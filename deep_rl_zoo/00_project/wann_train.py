import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
np.set_printoptions(precision=2, linewidth=160) 

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from wann_src import * # WANN evolution
from domain import *   # Task environments

def master_pytorch(): 
  """Main WANN optimization script
  """
  global fileName, hyp

  print("wann_train.py: master(): inside, type of hyp ", type(hyp))
  print("wann_train.py: master(): inside, rank = ", rank)
  data = DataGatherer(fileName, hyp)
  wann = Wann(hyp)
  print("wann_train.py: master(): rank = {}, Max Gen is {}".format(rank, hyp['maxGen']))
  print()
  print("*" * 80)

  for gen in range(hyp['maxGen']):
    print()
    pop = wann.ask()            # Get newly evolved individuals from WANN
    print("wann_train.py: master(): gen: ", gen, "pop length", len(pop))
    # print(pop[0].node)
    # print(pop[0].conn)
    # reward = batchMpiEval(pop)  # Send pop to evaluate
    reward = batchEval(pop)  # Send pop to evaluate
    print("wann_train.py: master(): type of reward: ", type(reward))
    print(reward)
    wann.tell(reward)           # Send fitness to WANN

    data = gatherData(data,wann,gen,hyp)
    print(gen, '\t - \t', data.display())

  # Clean up and data gathering at end of run
  data = gatherData(data,wann,gen,hyp,savePop=True)
  data.save()
  data.savePop(wann.pop,fileName)
  stopAllWorkers()

# -- Run NE -------------------------------------------------------------- -- #
def master(): 
  """Main WANN optimization script
  """
  global fileName, hyp

  print("wann_train.py: master(): inside, type of hyp ", type(hyp))
  print("wann_train.py: master(): inside, rank = ", rank)
  data = DataGatherer(fileName, hyp)
  wann = Wann(hyp)
  print("wann_train.py: master(): rank = {}, Max Gen is {}".format(rank, hyp['maxGen']))
  print()
  print("*" * 80)

  for gen in range(hyp['maxGen']):
    print()
    pop = wann.ask()            # Get newly evolved individuals from WANN
    print("wann_train.py: master(): gen: ", gen, "pop length", len(pop))
    # print(pop[0].node)
    # print(pop[0].conn)
    # reward = batchMpiEval(pop)  # Send pop to evaluate
    reward = batchEval(pop)  # Send pop to evaluate
    print("wann_train.py: master(): type of reward: ", type(reward))
    print(reward)
    wann.tell(reward)           # Send fitness to WANN

    data = gatherData(data,wann,gen,hyp)
    print(gen, '\t - \t', data.display())

  # Clean up and data gathering at end of run
  data = gatherData(data,wann,gen,hyp,savePop=True)
  data.save()
  data.savePop(wann.pop,fileName)
  stopAllWorkers()

def gatherData(data,wann,gen,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    wann       - (Wann)          - neat algorithm container
      .pop     - (Ind)           - list of individuals in population    
      .species - (Species)       - current species
    gen        - (int)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(wann.pop, wann.species)
  print("inside gatherData and gen%hyp['save_mod'] is: ", gen%hyp['save_mod'])
#   if (gen%hyp['save_mod']) is 0:
  if (gen%hyp['save_mod']) == 0:
    #data = checkBest(data, bestReps=16)
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks    
    global fileName
    pref = 'log/' + fileName
    import pickle
    with open(pref+'_pop.obj', 'wb') as fp:
      pickle.dump(wann.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = max(hyp['bestReps'], (nWorker-1))
    rep = np.tile(data.best[-1], bestReps)
    fitVector = batchMpiEval(rep, sameSeedForEachIndividual=False)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!      
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data

def batchEval(pop, sameSeedForEachIndividual=True):
    """Sends population to workers for evaluation one batch at a time.

    Args:
    pop - [Ind] - list of individuals
        .wMat - (np_array) - weight matrix of network
                [N X N] 
        .aVec - (np_array) - activation function of each node
                [N X 1]


    Optional:
        sameSeedForEachIndividual - (bool) - use same seed for each individual?

    Return:
    reward  - (np_array) - fitness value of each individual
                [N X 1]

    Todo:
    * Asynchronous evaluation instead of batches
    """  
    global nWorker, hyp
    #   nSlave = nWorker-1
    nJobs = len(pop)
    #   nBatch= math.ceil(nJobs/nSlave) # First worker is master
    nBatch= nJobs
    print(" nJobs: ", nJobs, " nBatch: ", nBatch)
    # global hyp

    #   print("wann_train.py: slave(): inside, rank = ", rank)

    task = Task(games[hyp['task']], nReps=hyp['alg_nReps'])

    print("wann_train.py: batchEval(): after task init, rank = ", rank)

    # Set same seed for each individual
    if sameSeedForEachIndividual is False:
        seed = np.random.randint(1000, size=nJobs)
    else:
        seed = np.random.randint(1000)

    reward = np.empty((nJobs,hyp['alg_nVals']), dtype=np.float64)
    i = 0 # Index of fitness we are filling
    for iBatch in range(nBatch): # Send one batch of individuals
        # for iWork in range(nSlave): # (one to each worker if there)
        if i < nJobs:
            # print("wann_train.py: batchMpiEval(): iBatch: ", iBatch, " iWork: ", iWork, " i: ", i)
            print("wann_train.py: batchMpiEval(): iBatch: ", iBatch, " i: ", i)
            wVec   = pop[i].wMat.flatten()
            print("wann_train.py: batchEval(): wVec shape: ", np.shape(wVec))

            n_wVec = np.shape(wVec)[0]
            print("wann_train.py: batchEval(): n_wVec: ", n_wVec)

            aVec   = pop[i].aVec.flatten()
            print("wann_train.py: batchEval(): aVec shape: ", np.shape(aVec))

            n_aVec = np.shape(aVec)[0]
            print("wann_train.py: batchEval(): n_aVec: ", n_aVec)
            
            if sameSeedForEachIndividual is False:
            #   comm.send(seed.item(i), dest=(iWork)+1, tag=5)
                fseed = seed.item(i)
            else:
            #   comm.send(  seed, dest=(iWork)+1, tag=5)
                fseed = seed     

            # Evaluate any weight vectors sent this way
            # while True:
            # comm.send(n_wVec, dest=(iWork)+1, tag=1)
            # n_wVec = comm.recv(source=0,  tag=1)
            # how long is the array that's coming?
            if n_wVec > 0:
            #   wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
            #   comm.Send(  wVec, dest=(iWork)+1, tag=2)
            #   comm.Recv(wVec, source=0,  tag=2) # recieve weights

            #   comm.send(n_aVec, dest=(iWork)+1, tag=3)
            #   n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?

            #   comm.Send(  aVec, dest=(iWork)+1, tag=4)
            #   aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
            #   comm.Recv(aVec, source=0,  tag=4) # recieve it

            #   seed = comm.recv(source=0, tag=5) # random seed as int
                
            #   print("wann_train.py: slave(): before task.getDistFitness, rank = ", rank)
            #   result = task.getDistFitness(wVec,aVec,hyp,seed=seed) # process it
                result = task.getDistFitness(wVec,aVec,hyp,seed=fseed) # process it
            #   print("wann_train.py: slave(): after task.getDistFitness, rank = ", rank, " result: ", result)

            # comm.Send(result, dest=0) # send it back
            # comm.Recv(workResult, source=iWork)

                reward[i,:] = result

            if n_wVec < 0: # End signal recieved
                # print('Worker # ', rank, ' shutting down.')
                break


        else: # message size of 0 is signal to shutdown workers
            n_wVec = 0
            # comm.send(n_wVec,  dest=(iWork)+1)

        i = i+1

    return reward


# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(pop, sameSeedForEachIndividual=True):
  """Sends population to workers for evaluation one batch at a time.

  Args:
    pop - [Ind] - list of individuals
      .wMat - (np_array) - weight matrix of network
              [N X N] 
      .aVec - (np_array) - activation function of each node
              [N X 1]

  
  Optional:
      sameSeedForEachIndividual - (bool) - use same seed for each individual?

  Return:
    reward  - (np_array) - fitness value of each individual
              [N X 1]

  Todo:
    * Asynchronous evaluation instead of batches
  """  
  global nWorker, hyp
  nSlave = nWorker-1
  nJobs = len(pop)
  nBatch= math.ceil(nJobs/nSlave) # First worker is master
  print("nSlave: ", nSlave, " nJobs: ", nJobs, " nBatch: ", nBatch)

  # Set same seed for each individual
  if sameSeedForEachIndividual is False:
    seed = np.random.randint(1000, size=nJobs)
  else:
    seed = np.random.randint(1000)

  reward = np.empty( (nJobs,hyp['alg_nVals']), dtype=np.float64)
  i = 0 # Index of fitness we are filling
  for iBatch in range(nBatch): # Send one batch of individuals
    for iWork in range(nSlave): # (one to each worker if there)
      if i < nJobs:
        print("wann_train.py: batchMpiEval(): iBatch: ", iBatch, " iWork: ", iWork, " i: ", i)
        wVec   = pop[i].wMat.flatten()
        n_wVec = np.shape(wVec)[0]
        aVec   = pop[i].aVec.flatten()
        n_aVec = np.shape(aVec)[0]

        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        comm.Send(  wVec, dest=(iWork)+1, tag=2)
        comm.send(n_aVec, dest=(iWork)+1, tag=3)
        comm.Send(  aVec, dest=(iWork)+1, tag=4)
        if sameSeedForEachIndividual is False:
          comm.send(seed.item(i), dest=(iWork)+1, tag=5)
        else:
          comm.send(  seed, dest=(iWork)+1, tag=5)        

      else: # message size of 0 is signal to shutdown workers
        n_wVec = 0
        comm.send(n_wVec,  dest=(iWork)+1)
      i = i+1 
  
    # Get fitness values back for that batch
    i -= nSlave
    for iWork in range(1,nSlave+1):
      if i < nJobs:
        workResult = np.empty(hyp['alg_nVals'], dtype='d')
        comm.Recv(workResult, source=iWork)
        reward[i,:] = workResult
      i+=1
  return reward

def slave():
  """Evaluation process: evaluates networks sent from master process. 

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node 
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """  
  global hyp

  print("wann_train.py: slave(): inside, rank = ", rank)

  task = Task(games[hyp['task']], nReps=hyp['alg_nReps'])

  print("wann_train.py: slave(): after task init, rank = ", rank)

  # Evaluate any weight vectors sent this way
  while True:
    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights

      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it

      seed = comm.recv(source=0, tag=5) # random seed as int
      
    #   print("wann_train.py: slave(): before task.getDistFitness, rank = ", rank)
      result = task.getDistFitness(wVec,aVec,hyp,seed=seed) # process it
    #   print("wann_train.py: slave(): after task.getDistFitness, rank = ", rank, " result: ", result)

      comm.Send(result, dest=0) # send it back

    if n_wVec < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

def stopAllWorkers():
  """Sends signal to all workers to shutdown.
  """
  global nWorker
  nSlave = nWorker-1
  print('stopping workers')
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    print("wann_train.py: mpifork(): os.getenv(\"IN_MPI\"): ", os.getenv("IN_MPI"))

    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print("wann_trian.py: mpi_fork(), before subprocess")
    # print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    a = sys.executable
    b = sys.argv
    print("mpirun -np {}, {}  -u {}".format(n, a, b))
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    print("wann_train.py: mpifork(): after subprocess")
    return "parent"
  else:
    print("wann_train.py: mpifork(): os.getenv(\"IN_MPI\"): ", os.getenv("IN_MPI"))
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    print('wann_train.py: mpi_fork(): assigning the rank and nworkers', nWorker, rank)
    return "child"


# -- Input Parsing ------------------------------------------------------- -- #

def main(argv):
  """Handles command line input, launches optimization or evaluation script
  depending on MPI rank.
  """
  global fileName, hyp
  fileName    = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam

  print("wann_train.py: main.c: rank = ", rank, " args: ", args)

  hyp = loadHyp(pFileName=hyp_default)
  print("wann_train.py: main.c: after loadHyp, rank = ", rank)

  updateHyp(hyp,hyp_adjust)
  print("wann_train.py: main.c: after updateHyp, rank = ", rank)

  print("wann_train.py: main.c: rank = ", rank, " hyp: ", hyp)


  # Launch main thread and workers
  if (rank == 0):
    print("wann_train.py: main(): start master(): rank = ", rank)
    master()
  else:
    print("wann_train.py: main(): start slave(): rank = ", rank)
    slave()

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Evolve NEAT networks'))
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_wan.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default='p/laptop_swing.json')

  parser.add_argument('-o', '--outPrefix', type=str,\
   help='file name for result output', default='test')
  
  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=8)

  args = parser.parse_args()

  print("wann_train.py: args: ", args)
  print("wann_train.py: rank: ", rank)

  print("wann_train.py: before mpi_fork")

#   nWorker = args.num_worker

  # Use MPI if parallel
#   if "parent" == mpi_fork(args.num_worker+1): os._exit(0)

  main(args)                              
  




