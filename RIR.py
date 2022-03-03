#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gpuRIR
gpuRIR.activateMixedPrecision(False)

import time
import numpy as np
import torch
from scipy.io import wavfile


'''
    To Generate RIR filter    
'''
def gen_filter(room,beta,traj_src,traj_mic,n_img,Tmax,fs,Tdiff,orV_rcv,mic_pattern):
    RIRs = np.zeros((len(tarj_src),nb_rcv,len_RIR))
    for i in range(len(traj_mic)) : 
        RIRs[i,:,:] = gpuRIR.simulateRIR(room_sz, beta, pos_traj, pos_rcv[i], nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)[i,:,:]

'''
    To load generated RIR filter
'''
def load_filter():
    pass


'''
    path_source : 
    RT60 : float
    room : [x,y,z]
    pos_mic : array of mic array displacements
    trajs_src : trajectories of sources
    traj_mic  : trajectory of mic
    att_diff : Attenuation when start using the diffuse reverberation model  [dB]
    att_max : Attenuation at the end of the simulation [dB]
'''
def Mix(
    source,
    trajs_src,
    traj_mic,
    path_output,
    noise=None,
    RT60=0.5,
    room=[
        5.0, 
        5.0, 
        2.5
    ],
    pos_mic=[
        [0.00,0.00,0.00],
        [0.00,0.00,0.00],
        [0.00,0.00,0.00],
        [0.00,0.00,0.00]
    ],
    mic_pattern = "omni",
    att_diff = 15.0,
    att_max = 60.0
):
    orV_rcv = None

    # Reflection coefficients
    beta = gpuRIR.beta_SabineEstimation(room_sz, RT60) 
    # Time to start the diffuse reverberation model [s]
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, RT60)     
	# Time to stop the simulation [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, RT60)
    # Number of image sources in each dimension
    n_img = gpuRIR.t2n( Tdiff, room_sz )	

    len_traj = len(traj_src)
    n_source = len(sources)
    len_RIR = int(fs*RT60)

    # RIR
    RIRs = gen_filter(room,beta,traj_src,traj_mic, n_img,Tmax,fs,Tdiff,orV_rcv,mic_pattern)

    # Filter
    filtered = gpuRIR.simulateTrajectory(source, RIRs.astype(np.float32))

    return filtered


'''
To generate random output

'''
def generate(path_out,path_sources):

    # load files 

    # match audio length

    # generate trajectories

    # RIR

    # MIX with SNR ratio

    # normalize

    wavfile.write(path_out, fs, signal)

