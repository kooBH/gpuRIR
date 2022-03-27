#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################
# Simulation data with direction 3D-angle label
# with maximum 4 moving sources
# with moving capture device
# using gpuRIR(https://github.com/DavidDiazGuerra/gpuRIR)
# 
# kooBH / nine4409@sogang.ac.kr
##############################################

import pdb

import gpuRIR
gpuRIR.activateMixedPrecision(False)

import time
import numpy as np
import torch
from scipy.io import wavfile
import json

# Not to have crossed trajectory,
# divide room for 5 sectors 
# => s0,s1,s2,s3,mic
# 
# * to avoid corssing case, 
#   mic secton will be rhombus shaped
# y
#  s0   |    s1 
#    ---|---
#    |⟋   ⟍|
# ---  mic  ----
#    |⟍   ⟋|
#    ---|---
#  s2   |    s3
# 0             x
# 
# -----------------------
#  each sector for source
#  is divided into 3 parts
#  ex) sector for s0
#         |
#  part 2 |  part 3  
#   ------ ------⟋     
#  part 1  |   ⟋
#          |  ⟋  mic
#   ------ ⟋
#
# =>  into parts
# y
#
#  10 9  8  7
#  11 m  m  6
#  0  m  m  5  
#  1  2  3  4 
# 0           x
#
def gen_traj(room,n_src=4,n_traj=50):
    ratio_mic = np.random.uniform(low=0.3,high=0.5)
    
    # adjust to avoid wall attachment
    eta = 0.5

    # [start,end][x,y,z]
    sec_m = np.zeros((2,3))
    sec_m[:,0] = [room[0]*0.5*(1-ratio_mic), room[0]*0.5*(1+ratio_mic)]
    sec_m[:,1] = [room[1]*0.5*(1-ratio_mic), room[1]*0.5*(1+ratio_mic)]
    sec_m[:,2] = [0.3,0.3]  # 30 cm fixed

    w = room[0] - eta
    h = room[1] - eta

    mw = w*ratio_mic
    mh = h*ratio_mic

    ## parts[12-part][start,end][x,y,z]
    parts = np.zeros((12,2,3))
    parts += eta

    # x start
    tmp = 0.5*w - 0.5*mw
    parts[2,0,0] = tmp
    parts[9,0,0] = tmp

    tmp = 0.5*w
    parts[3,0,0] = tmp
    parts[8,0,0] = tmp

    tmp = 0.5*w + 0.5*mw
    parts[4:8,0,0] = tmp

    # x end
    parts[1:4,1,0]=parts[2:5,0,0]
    parts[10:7:-1,1,0]=parts[9:6:-1,0,0]

    parts[4:8,1,0] = w
    parts[10:,1,0] = 0.5*w - 0.5*mw
    parts[0,1,0] = 0.5*w - 0.5*mw

    # y start
    tmp =  0.5*h - 0.5*mh
    parts[5,0,1] = tmp
    parts[0,0,1] = tmp

    tmp = 0.5*h
    parts[6,0,1] = tmp
    parts[11,0,1] = tmp

    tmp =  0.5*h + 0.5*mh
    parts[7:11,0,1] = tmp

    # y end
    parts[1:4,1,1] = 0.5*h - 0.5*mh
    parts[4:7,1,1] = parts[4:7,0,1]
    parts[7:11,1,1] = h

    tmp = 0.5*h
    parts[5,1,1] =  tmp
    parts[0,1,1] = tmp

    tmp =  0.5*h + 0.5*mh
    parts[6,1,1] = tmp
    parts[11,1,1] = tmp

    # z
    parts[:,0,2] = 0.3
    parts[:,1,2] = room[2] - eta

    ## MiC trajectory  : rhombus shaped
    pts_m = np.zeros((2,3))
    pts_m[:,2] = np.random.uniform(low=sec_m[0,2],high=sec_m[1,2],size=2)
    # start
    pts_m[0,0] = np.random.uniform(low=0.5*w-0.5*mw,high= 0.5*w+0.5*mw)

    if pts_m[0,0] - 0.5*w > 0.5*mw : 
        tmp = 2*mh*pts_m[0,0]/mw
    else :
        tmp =  2*mh*(1-pts_m[0,0]/mw)
    pts_m[0,1] = np.random.uniform(low=0.5*h-0.5*tmp,high=0.5*h+0.5*tmp)
    #end
    pts_m[1,0] = np.random.uniform(low=0.5*w-0.5*mw,high= 0.5*w+0.5*mw)
    if pts_m[1,0] -0.5*w > 0.5*mw : 
        tmp = 2*mh*pts_m[1,0]/mw
    else :
        tmp =  2*mh*(1-pts_m[1,0]/mw)
    pts_m[1,1] = np.random.uniform(low=0.5*h-0.5*tmp,high=0.5*h+0.5*tmp)

    # gen traj
    traj_m = np.zeros((n_traj,3))
    traj_m[:,0] = np.linspace(pts_m[0,0],pts_m[1,0],n_traj)
    traj_m[:,1] = np.linspace(pts_m[0,1],pts_m[1,1],n_traj)
    traj_m[:,2] = np.linspace(pts_m[0,2],pts_m[1,2],n_traj)

    ## Soruces trajectory
    traj_s = np.zeros((n_src,n_traj,3))
    pts_s = np.zeros((n_src,2,3))
    # traj per src==section
    #print("room : {}".format(room))
    #print("pts_m {}".format(pts_m))
    for i in range(n_src) :
        ps = np.random.choice(range(3*i,3*i+4),2)

        pts_s = np.random.uniform(low=parts[ps[0],0,:],high=parts[ps[1],1,:],size=(2,3))
        #print("pts_s[{}] {}".format(i,pts_s))
        traj_s[i,:,:]=np.linspace(pts_s[0],pts_s[1],n_traj)    

    return traj_m, traj_s

#    To Generate RIR filter    
def gen_filter(room,traj_src,traj_mic,beta,n_img,RT60, Tmax,Tdiff,orV_rcv,mic_pattern,n_rec=4,fs=16000):
    n_traj_pts =  traj_mic.shape[1]
    len_filter =  int(fs * RT60)

    n_src = len(traj_src)

    RIR = np.zeros((n_traj_pts,n_rec,len_filter))
    #print("traj_src : "+str(traj_src.shape))
    #print("traj_mic : "+str(traj_mic.shape))
    #print('RIRs : ' + str(RIRs.shape))

    # fixed src, fixed mic

    # fixed src, moving mic

    # moving sr, moving mic

    # moving src, moving mic

    for i in range(n_traj_pts) : 
        # [traj,src,rec,filter]
        # TODO : need to fix fitler length dismatch problem
        # RT60 * fs != filter length

        RIR[i,:,:len_filter] = gpuRIR.simulateRIR(room, beta, traj_src, traj_mic[:,i,:], n_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)[i,:,:len_filter]

        #print("traj_src : "+str(traj_src.shape))
        #print("traj_mic : "+str(traj_mic.shape))
        #print("RIR : "+str(RIR.shape))

    return RIR


#    To load generated RIR filter
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
    sources,
    SIRs,
    trajs_src,
    trajs_mic,
    path_filter=None,
    noise=None,
    RT60=0.5,
    room=[
        5.0, 
        5.0, 
        2.5
    ],
    mic_pattern = "omni",
    att_diff = 15.0,
    att_max = 60.0,
    fs = 16000
):
    if len(SIRs) != len(sources) : 
        print(sources.shape)
        raise Exception("len(SIRs) {} != len(sources) {}".format(len(SIRs),len(sources)))
    orV_rcv = None

    # Reflection coefficients
    beta = gpuRIR.beta_SabineEstimation(room, RT60) 
    # Time to start the diffuse reverberation model [s]
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, RT60)     
	# Time to stop the simulation [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, RT60)
    # Number of image sources in each dimension
    n_img = gpuRIR.t2n( Tdiff, room )	

    len_traj = trajs_src.shape[1]
    n_src = len(sources)
    n_mic = trajs_mic.shape[0]
    len_RIR = int(fs*RT60)

    #for i in range(n_src) : 
     #   wavfile.write(path_out+"/ori_"+str(i)+".wav", fs, sources[i])

    #print("mix::trajs_src : " + str(trajs_src.shape))
    #print("mix::trajs_mic : " + str(trajs_mic.shape))

    RIRs = np.zeros((n_src,len_traj,n_mic,len_RIR))

    # RIR
    if path_filter is None : 
        for i in range(n_src) : 
            RIRs[i,:,:,:] = gen_filter(room,trajs_src[i],trajs_mic, beta=beta,n_img=n_img,RT60=RT60,Tmax=Tmax  ,Tdiff=Tdiff,orV_rcv=orV_rcv,mic_pattern=mic_pattern,fs=fs)
    else : 
        RIRs = load_filter(path_filter)

    #print("sources : "+str(np.array(sources).shape))
    #print("RIRs : " + str(RIRs.shape))

    # Filtering per source
    signals = []
    for i in range(n_src) : 
        # (wav length ,channel)
        signals.append(gpuRIR.simulateTrajectory(sources[i], RIRs[i].astype(np.float32)))

        #signals.append(np.tile(sources[i],(4,1)))

        # normalization before mixing 
        signals[i] = signals[i]/np.max(np.abs(signals[i])) 

    # merge with SIR ratio 
    ## Cacluate Enerygy  
    energies = []

    for i in range(n_src) : 
        sum_energy = np.sum(np.power(signals[i][:,0],2))
        len_signal = len(signals[i][:,0])
        energies.append(sum_energy/len_signal)

    ## NOTE signal with lowest SIR will be base 
    idx_low_SIR = np.argmin(SIRs)
    
    # SIR
    for i in range(n_src) : 
        ## adjust enerygy by SIR
        if i == idx_low_SIR :
            SIRs[i]=0
            continue
        # set first signal as SIR base and get ratio
        normal = np.sqrt(energies[idx_low_SIR])/np.sqrt(energies[i])
        weight = normal / np.sqrt(np.power(10,SIRs[i]/10))

        signals[i] = signals[i]*weight
    signal = np.sum(np.array(signals),axis=0)

    ## Normalization
    signal = signal/np.expand_dims(np.max(np.abs(signal),axis=0),axis=0)

    # Debug : SIR
    #for i in range(n_src) : 
    #    wavfile.write(path_out+"/tmp_"+str(i)+".wav", fs, signals[i])

    return signal,SIRs

def diffuse():
    pass

'''
To generate random output

'''
def generate(path_out,path_sources,id_file,n_traj = 50,match="min",shift=128)->None:

    if len(path_sources) > 4:
        raise Exception("ERORR:Maximum number of sources is 4.")

    raws = []
    len_max = 0
    len_min = 1e16

    # mic array 
    pos_mic=[
        [-0.04,-0.04,0.00],
        [-0.04,+0.04,0.00],
        [+0.04,-0.04,0.00],
        [+0.04,+0.04,0.00]
    ]
    pos_mic = np.expand_dims(pos_mic,1)
    n_rec = len(pos_mic)

    # load files 
    for path in path_sources :
        fs, raw = wavfile.read(path)
        raws.append(raw)
        if len_max < len(raw) : 
            len_max = len(raw)
        if len_min > len(raw) : 
            len_min = len(raw)
    n_src = len(raws)

    meta = {}
    meta["n_src"]=n_src
    #print(n_src)
    
    ## Matching length of sources
    # comapct
    if  match == "min":
        for i in range(n_src) :
            idx_start = int(len(raws[i])/2 - len_min/2)
            raws[i] = raws[i][idx_start:idx_start+len_min]
        len_signals = len_min
    # TODO
    elif match == 'max':
        raise Exception("Unimplemented")
        # for now 1.5*ma
        len_data = 1.5*len_max
        # spread sources

        len_signals = len_data
    else :
        raise Exception("ERROR:generate(), unsupported matching method : {}".format(match))

    # Debug : raws
    #for i in range(n_src):
    #    wavfile.write(path_out+"/tmp_"+str(i)+".wav", fs, raws[i])
    n_frame = int(np.ceil(len_signals/shift))

    # generate room
    room = np.random.uniform(low=[5,5,3.0],high=[10,10,3.5])
    meta["room"]=room

    #s_idx = np.arange(n_src)
    #np.random.shuffle(s_idx)
#    s_idx = s_idx[:n_src]
    signals = []

    # trajectory allocation
    traj_m,traj_s = gen_traj(room,n_src)
    traj_mm = np.tile(traj_m,(n_rec,1,1))

    traj_mm = traj_mm + pos_mic

    #print("traj_m : "+str(traj_m.shape))
    #print("traj_s : "+str(traj_s.shape))

    meta["traj_m"] = traj_m
    meta["traj_s"] = traj_s
    #meta["s_idx"] = s_idx

    SIRs = np.random.uniform(low=0, high=10, size=n_src)
    RT60 = np.random.uniform(low=0.5, high=0.7, size=None) 
    signal,SIRs = mix(raws,SIRs,traj_s,traj_mm,room=room, RT60=RT60) 
    meta["SIRs"]=SIRs

    ## Save angles in label
    # match size of n_traj to n_frame for label
    traj_m_adj = np.zeros((n_frame,3))
    traj_adj = np.zeros((n_src,n_frame,3))    

    ratio = int(n_frame/n_traj)
    n_req_pad = n_frame - ratio*n_traj

    idx_adj = 0
    for i in range(n_traj):
        len_rep = ratio
        # padding
        if i < n_req_pad :
            len_rep +=1

        traj_m_adj[idx_adj:idx_adj+len_rep,:] = traj_m[i,:]
        traj_adj[:,idx_adj:idx_adj+len_rep:,:] = traj_s[:,i:i+1,:]
        idx_adj += len_rep

    # Calculate Angle of Direct Path
    for i in range(n_src) : 
        dist = np.sqrt(np.power(traj_m_adj[:,0]-traj_adj[i,:,0],2) + np.power(traj_m_adj[:,1]-traj_adj[i,:,1],2))
        aizmuth = np.arctan((traj_m_adj[:,1]-traj_adj[i,:,1])/(traj_m_adj[:,0]-traj_adj[i,:,0]))
        elevation = np.arctan(dist/(traj_adj[i,:,2] - traj_m_adj[:,2] ))
        meta["azimuth_"+str(i)] = np.degrees(aizmuth)
        meta["elevation_"+str(i)]= np.degrees(elevation)
     
    ## save
    wavfile.write(path_out+"/"+id_file+".wav", fs, signal)
    with open(path_out+"/"+id_file+".json", 'w') as f:
        json.dump(meta, f, indent=2,cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Test script
if __name__=="__main__" : 

    path_sources = [
        "./sample/male_1.wav",
        "./sample/female_1.wav",
        ]
    path_out = "./sample/"
    
    generate(path_out,path_sources,"test")