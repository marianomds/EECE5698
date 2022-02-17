import numpy as np
from numpy import random
import matplotlib.pyplot as plt


STEPS = 1000
RUNS  = 100
MU1   = 5
MU2   = 10
MU3   = 4
VAR1  = 10
VAR2  = 15
VAR3  = 10
EPS   = [0, 0.1, 0.2, 0.5]
Q_B   = [[0, 0], [5, 7], [20, 20]]
EPS_B = 0.1
A_B   = 0.1
A_C   = 0.1
MIN_Y = 5.25
MAX_Y = 7.25

np.random.seed(1)

def reward_f(action):
  if action == 1:
    reward = random.normal(loc=MU1, scale=np.sqrt(VAR1))
  else: # action 2
    if random.uniform() > 0.5:
      reward = random.normal(loc=MU2, scale=np.sqrt(VAR2))
    else:
      reward = random.normal(loc=MU3, scale=np.sqrt(VAR3))
  return reward

def e_greedy(epsilon, q1, q2):
  if (random.uniform() < epsilon) | (q1 == q2): # random
    action = round(random.uniform() + 1)
  else: # greedy  
    if q1 > q2:
      action = 1
    else:
      action = 2
  reward = reward_f(action)
#  print("Epsilon: {} - Action: {} - Reward: {}".format(epsilon, action, reward))
  return (action, reward)

def gradient(pi1):
  if (random.uniform() < pi1):
    action = 1
  else:
    action = 2
  reward = reward_f(action)
  return (action, reward)


print("----- PART A -----")


# Initialize storage matrices
R  = np.zeros([RUNS, STEPS]) # rewards
AR = np.zeros([4, 4, STEPS])# average accumulated rewards
Q1 = np.zeros(RUNS) # Q1 for each run
Q2 = np.zeros(RUNS) # Q2 for each run
Q1a = np.zeros([4, 4]) # average final Q1
Q2a = np.zeros([4, 4]) # average final Q2
A = np.zeros([4, STEPS]) # learning rates (alpha)
for a in range (4):
  for k in range(STEPS):
    if   a == 0: A[a][k] = 1
    elif a == 1: A[a][k] = 0.9**(k + 1)
    elif a == 2: A[a][k] = 1/(1 + np.log(1 + (k + 1)))
    elif a == 3: A[a][k] = 1/(k + 1)

for a in range (4):
  for e in range(4):
    for i in range(RUNS):
      for k in range(STEPS):
        act, r = e_greedy(EPS[e], Q1[i], Q2[i])
        if act == 1: Q1[i] = Q1[i] + A[a][k]*(r - Q1[i])
        else:        Q2[i] = Q2[i] + A[a][k]*(r - Q2[i])
        R[i][k] = r # store each step's reward
      # convert them to accumulated rewards
      for k in reversed(range(STEPS)):
        R[i][k] = np.mean(R[i][:k+1])
    Q1a[a][e] = np.mean(Q1)
    Q2a[a][e] = np.mean(Q2)
    # average the accumulated rewards over all runs
    for k in range(STEPS):
      acc = 0
      for i in range(RUNS):
        acc = acc + R[i][k]
      AR[a][e][k] = acc/RUNS

for a in range (4):
  fig, ax1 = plt.subplots()
  for e in range(4):
    print("Alpha: {} - Epsilon: {}\t- Ave final Q1: {:.4}\t- Ave final Q2: {:.4}".format(a, EPS[e], Q1a[a][e], Q2a[a][e]))
    ax1.plot(AR[a][e][:])
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  ax2.plot(A[a][:], 'k')
  fig.legend(["E = {}".format(EPS[0]), "E = {}".format(EPS[1]), "E = {}".format(EPS[2]), "E = {}".format(EPS[3]), "Learning rate"], loc = "lower right")
  ax1.axis([0, STEPS, MIN_Y, MAX_Y])
  ax2.axis([0, STEPS, 0, 1.2])
  ax1.set_ylabel('Average Accumulated Reward')
  ax2.set_ylabel('Learning Rate')
  ax1.set_xlabel('Step K')
  plt.show(block=False)


print("----- PART B -----")


R   = np.zeros([RUNS, STEPS]) # rewards
AR  = np.zeros([3, STEPS])# average accumulated rewards
Q1a = np.zeros(3) # average final Q1
Q2a = np.zeros(3) # average final Q2

for q in range(3):
  Q1 = np.ones(RUNS)*Q_B[q][0] # Q1 for each run
  Q2 = np.ones(RUNS)*Q_B[q][1] # Q2 for each run
  for i in range(RUNS):
    for k in range(STEPS):
      act, r = e_greedy(EPS_B, Q1[i], Q2[i])
      if act == 1: Q1[i] = Q1[i] + A_B*(r - Q1[i])
      else:        Q2[i] = Q2[i] + A_B*(r - Q2[i])
      R[i][k] = r # store each step's reward
    # convert them to accumulated rewards
    for k in reversed(range(STEPS)):
      R[i][k] = np.mean(R[i][:k+1])
  Q1a[q] = np.mean(Q1)
  Q2a[q] = np.mean(Q2)
  # average the accumulated rewards over all runs
  for k in range(STEPS):
    acc = 0
    for i in range(RUNS):
      acc = acc + R[i][k]
    AR[q][k] = acc/RUNS

plt.figure()
for q in range(3):
  print("Initial Q: {}\t- Ave final Q1: {:.4}\t- Ave final Q2: {:.4}".format(Q_B[q], Q1a[q], Q2a[q]))
  plt.plot(AR[q][:])
plt.legend(["[0 0]", "[5 7]", "[20 20]"], loc = "lower right")
plt.axis([0, STEPS, MIN_Y, MAX_Y])
plt.ylabel('Average Accumulated Reward')
plt.xlabel('Step K')
plt.show(block=False)


print("----- PART C -----")


H1 = 0
H2 = 0
Ra  = np.zeros([RUNS, STEPS]) # accumulated rewards
ARc = np.zeros([STEPS]) # average accumulated rewards

for i in range(RUNS):
  R  = np.array([]) # rewards
  for k in range(STEPS):
    pi1 = np.exp(H1)/(np.exp(H1) + np.exp(H2))
  #  pi2 = np.exp(H2)/(np.exp(H1) + np.exp(H2))
    pi2 = 1 - pi1 # faster, same result
    act, r = gradient(pi1)
    R = np.append(R, r) # store each step's reward
    Ra[i][k] = np.mean(R) # accumulated rewards up to current step
    if act == 1:
      H1 = H1 + A_C*(r - Ra[i][k])*(1 - pi1)
      H2 = H2 - A_C*(r - Ra[i][k])*pi2
    else:
      H2 = H2 + A_C*(r - Ra[i][k])*(1 - pi2)
      H1 = H1 - A_C*(r - Ra[i][k])*pi1

# average the accumulated rewards over all runs
for k in range(STEPS):
  acc = 0
  for i in range(RUNS):
    acc = acc + Ra[i][k]
  ARc[k] = acc/RUNS

plt.figure()
plt.plot(AR[0][:]) # [0, 0] from part B
plt.plot(ARc[:])
plt.legend(["E-Greedy", "Gradient-Bandit"], loc = "lower right")
plt.axis([0, STEPS, MIN_Y, MAX_Y])
plt.ylabel('Average Accumulated Reward')
plt.xlabel('Step K')
plt.show(block=False)

input()
