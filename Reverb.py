"""
    Generate RIR filter for data augmentation
"""
import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)
import numpy as np
import argparse
import os
from tqdm.auto import tqdm


roomsize_min = np.array([12,20,4])
roomsize_max = np.array([5,5,2.5])
RT60_range =np.array([0.7, 2.0]) 
att_max_range=np.array([70,120])
fs=8000

n_src = 1
n_rcv = 1
mic_pattern = "omni"
orv_rcv = None
eps = 1e-1

att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
#att_max = 60.0 # Attenuation at the end of the simulation [dB]

roomsize_diff = roomsize_max-roomsize_min
RT60_diff = RT60_range[1]-RT60_range[0]
att_max_diff = att_max_range[1]-att_max_range[0]

def gen_filter(room,RT60,pos_src,pos_rcv,att_max) :
    # Reflection coefficients
    beta = gpuRIR.beta_SabineEstimation(room, RT60) 
    # Time to start the diffuse reverberation model [s]
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, RT60)     
    # Time to stop the simulation [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, RT60)
    n_img = gpuRIR.t2n( Tdiff, room )	

    RIR = gpuRIR.simulateRIR(room, beta, pos_src, pos_rcv, n_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orv_rcv, mic_pattern=mic_pattern)

    return RIR

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_out','-n', type=int, required=True)
    parser.add_argument('--dir_out','-o', type=str, required=True)
    args = parser.parse_args()
    n_out = args.n_out

    os.makedirs(args.dir_out,exist_ok=True)

    for i in tqdm(range(n_out)) :
        rand = np.random.rand(3)
        room_size = roomsize_min + roomsize_diff*rand
        
        rand = np.random.rand(1,3)
        rand[rand<eps] = eps
        rand[rand>1-eps] = 1-eps
        pos_rcv = room_size*rand

        rand = np.random.rand(1,3)
        rand[rand<eps] = eps
        rand[rand>1-eps] = 1-eps
        pos_src = room_size*rand

        RT60 = RT60_range[0] + np.random.rand()*RT60_diff
        att_max = att_max_range[0] + np.random.rand()*att_max_diff

        RIR = gen_filter(room_size,RT60,pos_src,pos_rcv,att_max)

        path = "RIR_RT60_{}_fs_{}_room_{}_{}_{}_src_{}_{}_{}_rcv_{}_{}_{}_att_{}.npy".format(
            int(RT60*100),
            fs,
            int(room_size[0]),
            int(room_size[1]),
            int(room_size[2]),

            int(pos_src[0,0]),
            int(pos_src[0,1]),
            int(pos_src[0,2]),

            int(pos_rcv[0,0]),
            int(pos_rcv[0,1]),
            int(pos_rcv[0,2]),

            att_max
            )

        np.save(os.path.join(args.dir_out,path),RIR[0,0])



