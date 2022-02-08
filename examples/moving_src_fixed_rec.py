#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a moving source with a microphone array.
You need to have a 'source_signal.wav' audio file to use it as source signal and it will generate
the file 'filtered_signal.wav' with the stereo recording simulation.
"""
import time
import numpy as np
from scipy.io import wavfile

tic = time.time()
import gpuRIR
gpuRIR.activateMixedPrecision(False)
print("init gpuRIR : " + str(time.time()-tic)+ "sec")

tic = time.time()
#fs, source_signal = wavfile.read('source_signal.wav')
fs, source_signal = wavfile.read('source_2.wav')
print("load wav : " + str(time.time()-tic)+ "sec")
print("source signal : " + str(source_signal.shape))

tic = time.time()
room_sz = [5,5,2.5]  # Size of the room [m]

## NOTE : discrete trajectory points 
traj_pts = 300  # Number of trajectory points

pos_traj = np.tile(np.array([2.5,2.5,1.7]), (traj_pts,1))

#pos_traj[:,0] = np.linspace(0.1, 4.9, traj_pts) # Positions of the trajectory points [m]
# moving on axis-Y
pos_traj[:,1] = np.linspace(0.1, 4.9, traj_pts) # Positions of the trajectory points [m]
# moving on axis-Z
#pos_traj[:,2] = np.linspace(1.5, 1.7, traj_pts) # Positions of the trajectory points [m]

print("pos_traj : " + str(pos_traj.shape))

nb_rcv = 4 # Number of receivers
pos_rcv = np.array([
        [2.50  - 0.04 ,2.5 + 0.04, 1.0]
        ,[2.50 + 0.04 ,2.5 + 0.04, 1.0]
        ,[2.50 - 0.04 ,2.5 - 0.04, 1.0]
        ,[2.50 + 0.04 ,2.5 - 0.04, 1.0]
        
])	 # Position of the receivers [m]


#T60 = 0.6 # Time for the RIR to reach 60dB of attenuation [s]
T60 = 0.7 # Time for the RIR to reach 60dB of attenuation [s]

#rcv_traj = np.tile(np.linspace(0.1,7.5,traj_pts),(nb_rcv,1))
#print("rcv_traj : " + str(rcv_traj.shape))
#
#pos_rcv[:,:,0] = np.transpose( rcv_traj)
#pos_rcv[:,1,0] +=0.3
#print("pos_rcv : " + str(pos_rcv.shape))


# orV_rcv :
# ndarray with 2 dimensions and 3 columns or None, optional. Orientation of the receivers as vectors pointing in the same direction. Applies to each receiver. None (default) is only valid for omnidirectional patterns

#orV_rcv = np.array([[-1,0,0],[1,0,0]])
orV_rcv = None
#mic_pattern = "card" # Receiver polar pattern
mic_pattern = "omni" # Receiver omnidrectional patterns
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]

beta = gpuRIR.beta_SabineEstimation(room_sz, T60) # Reflection coefficients
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

print("coef : " + str(time.time()-tic)+ "sec")


len_RIR = int(fs*T60)
RIRs = np.zeros((traj_pts,nb_rcv,len_RIR))

tic = time.time()
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_traj, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
print("RIRs : " +str(RIRs.shape))


print("filter generated : " + str(time.time()-tic)+ "sec")

tic = time.time()
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs.astype(np.float32))
filtered_signal = filtered_signal/np.max(np.abs(filtered_signal))
print("filter applied : " + str(time.time()-tic)+ "sec" )

print(filtered_signal.shape)

wavfile.write('filtered_signal.wav', fs, filtered_signal)