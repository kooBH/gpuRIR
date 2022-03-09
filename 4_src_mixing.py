#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
mixing script for 
maxmium 4-source senario

'''
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

    # fixed src, fixed mic

    # fixed src, moving mic

    # moving sr, moving mic

    # moving src, moving mic
    for i in range(len(traj_mic)) : 
        RIRs[i,:,:] = gpuRIR.simulateRIR(room_sz, beta, pos_traj, pos_rcv[i], nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)[i,:,:]
    return RIRs

'''
    To load generated RIR filter
'''
def load_filter(path_filter):
    raise Exception("ERROR::load_filter() not implemented")

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

def mix(
    source,
    trajs_src,
    traj_mic,
    path_filter=None,
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

    # traj_mics
    traj_mics = np.zeros(4,len_traj,3)
    for i in range(len(pos_mic)) : 
        traj_mics[i,:,:] = traj_mic + pos_mic[i]

    # RIR
    if path_filter is None : 
        RIRs = gen_filter(room,beta,traj_src,traj_mics, n_img,Tmax,fs,Tdiff,orV_rcv,mic_pattern)
    else : 
        RIRs = load_filter(path_filter)

    # Filter
    filtered = gpuRIR.simulateTrajectory(source, RIRs.astype(np.float32))

    return filtered


def diffuse():
    pass

'''
To generate random output

'''
def generate(path_out,path_sources,n_traj = 50):

    if len(path_sources) > 4:
        raise Exception("ERORR:Maximum number of sources is 4.")

    
    raws = []

    # load files 
    for path in path_sources :
        fs, raw = wavfile.read(path)
        raws.append(raw)
    n_src = len(raws)

    meta = {}
    print(n_src)

    # TODO:match audio length
    max_length = 0
    len_signals = []
    for x in raws : 
        len_signals.append(len(raws))
        if n_src > max_length :
            max_length = n_src
    

    # generate room
    room = [5, 4, 3] # temp : fix

    # mic array 
    pos_mic=[
        [-0.04,-0.04,0.00],
        [-0.04,+0.04,0.00],
        [+0.04,-0.04,0.00],
        [+0.04,+0.04,0.00]
    ]

    # Not to have crossed trajectory,
    # divide room for 5 sectors 
    # => s0,s1,s2,s3,mic
    # 
    # y
    #  s0   |    s1 
    #    ---|---
    #    |     |
    # ---  mic  ----
    #    |     |
    #    ---|---
    #  s2   |    s3
    # 0             x
    # sector of mic can vary. 

    # tmp : fixed value
    ratio_mic = 0.3

    sec_mic = [
            [room[0]*0.5*(1-ratio_min), room[0]*0.5*(1+ratio_mic)],
            [room[1]*0.5*(1-ratio_min), room[1]*0.5*(1+ratio_mic)],
            [0.3,0.3]  # 30 cm fixed
            ]
    
    ## Soruces trajectory
    # sec = [s0,s1,s2,s3]
    sec = [
            [
                [0, sec_mic[0]],
                [sec_mic[1],room[1]],
                [0.3, room[2]]
            ],
            [
                [sec_mic[1], room[0]],
                [sec_mic[1],room[1]],
                [0.3, room[2]]
            ],
            [
                [0, sec_mic[0]],
                [0, sec_mic[0]],
                [0.3, room[2]]
            ],
            [
                [sec_mic[1],room[0]],
                [0, sec_mic[0]],
                [0.3, room[2]]
            ]
        ]
    
    s_idx = np.arange(max_src)
    s_idx = np.random.shuffle(s_idx)
    s_idx = s_idx[:n_src]

    for i in range(n_src) :
        meta["path_s"+str(i)] = path_sources[s_idx[i]]

    ## MiC trajectory 
    ## initial point
    x_0 = np.random.uniform(low=sec_mic[0,0], high=sec_mic[0,1], size=None)
    y_0 = np.random.uniform(low=sec_mic[1,0], high=sec_mic[1,1], size=None)
    z_0 = np.random.uniform(low=sec_mic[2,0], high=sec_mic[2,1], size=None)

    ## destination
    x_1 = np.random.uniform(low=sec_mic[0,0], high=sec_mic[0,1], size=None)
    y_1 = np.random.uniform(low=sec_mic[1,0], high=sec_mic[1,1], size=None)
    z_1 = np.random.uniform(low=sec_mic[2,0], high=sec_mic[2,1], size=None)

    traj_m = np.zeros(n_traj,3)
    traj_m[:,0] = np.linespace(x_0,x_1,n_traj)
    traj_m[:,1] = np.linespace(y_0,y_1,n_traj)
    traj_m[:,2] = np.linespace(z_0,z_1,n_traj)

    signals = []

    traj_s = np.zeros(n_src,n_traj,3)
    for i in range(n_src) : 
        # generate trajectories
        # tmp : fix

        ## initial point
        x_0 = np.random.uniform(low=sec[s_idx[i],0,0], high=sec[s_idx[i],0,1], size=None)
        y_0 = np.random.uniform(low=sec[s_idx[i],1,0], high=sec[s_idx[i],1,1], size=None)
        z_0 = np.random.uniform(low=sec[s_idx[i],2,0], high=sec[s_idx[i],2,1], size=None)

        ## destination
        x_1 = np.random.uniform(low=sec[s_idx[i],0,0], high=sec[s_idx[i],0,1], size=None)
        y_1 = np.random.uniform(low=sec[s_idx[i],1,0], high=sec[s_idx[i],1,1], size=None)
        z_1 = np.random.uniform(low=sec[s_idx[i],2,0], high=sec[s_idx[i],2,1], size=None)

        traj[i,:,0] = np.linespace(x_0,x_1,n_traj)
        traj[i,:,1] = np.linespace(y_0,y_1,n_traj)
        traj[i,:,2] = np.linespace(z_0,z_1,n_traj)

        # RIR mixing
        RT60 = np.random.uniform(low=0.5, high=0.7, size=None)
        signal = mix(raws[s_idx[i]],traj,traj_mic,room=room,pos_mic=pos_mic,RT60=RT60)

        # normalization before mixing
        signal = signal/np.max(np.abs(signal))
        signals.append(signal)

    # merge with SNR ratio
    ## Cacluate Enerygy  
    energies = []

    for i in range(n_src) : 
        sum_energy = np.sum(np.power(signals[i],2),axis=1)
        energies.append(sum_energy/len_signals[s_idx[i]])
    
    # SNR
    for i in range(2, n_src) : 
        ## gen SNR
        SNR = np.random.uniform(low=0, high=10, size=None)

        ## adjust enerygy by SNR
        # set first signal as SNR base and get ratio
        normal = np.sqrt(energies[0])/np.sqrt(energies[i])
        weight = normal / np.sqrt(np.power(10,SNR/10))

        signals[i] = signals[i]*weight
    signal = np.sum(signals,axis=0)

    # Calculate Angle of Direct Path
    ## TODO

    ## Save angles in label
    for i in range(n_src) : 
        angles = np.zeros(n_frame)
        meta["angle_"+str(i)] = 

    # save
#    wavfile.write(path_out, fs, signal)



# Test script
if __name__=="__main__" : 
    path_sources = [
        "./sample/male_1.wav",
        "./sample/female_1.wav",
        ]
    path_out = [
        "./sample/out.wav"
    ]
    generate(path_out,path_sources)