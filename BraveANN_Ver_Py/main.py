import numpy as np
import pandas as pd
from utils import read_data, setup_seed, initialization, compute_tilted_sse_InEachCluster, get_noised_data, compute_sse
from update import rkm, Fastrkm, Balanced_rkm
from options import args_parser
import json
import time

args = args_parser()
seeds = args.seeds
sample_size = args.sample_size
dataset_names = ['sift', 'mnist', 'femnist', 'gist']
noise_frac_list = args.noise_frac
for noise_frac in noise_frac_list:
    for dataset_name in dataset_names:
        for seed in seeds:
            setup_seed(seed)
            if (dataset_name == 'sift'):
                csv_file_path = 'data\siftsmall_128_normalized.csv'
            elif (dataset_name == "mnist"):
                csv_file_path = 'data\mnist.csv'
            elif (dataset_name == "femnist"):
                csv_file_path = 'data\\fashion_mnist.csv'
            elif (dataset_name == "gist"):
                csv_file_path = 'data\\gist_vectors.csv'
            data = read_data(csv_file_path)
            data_with_noise = get_noised_data(data, noise_frac)
            t_list = args.t
            k_list = args.num_clusters
            epoch_list = args.epoch_list
            lr_list = args.lr_list

            for k in k_list:
                for t in t_list:
                    for num_epoch in epoch_list:
                        for lr in lr_list:
                            output = {}
                            time1 = time.monotonic()
                            centroids, labels = initialization(data_with_noise, k, args)
                            phi = compute_tilted_sse_InEachCluster(data_with_noise, centroids, labels, k, t)
                            if args.solver == 'rkm':
                                centroids, labels, SSE, tilted_SSE = rkm(data_with_noise, args, t, k, num_epoch, lr)
                            elif args.solver == 'fast_rkm':
                                centroids, labels, SSE, tilted_SSE = Fastrkm(data_with_noise, args, t, k, num_epoch, lr, centroids, labels, phi)
                            elif args.solver == 'balanced_rkm':
                                centroids, labels, SSE = Balanced_rkm(data_with_noise, args, t, k, num_epoch, lr, centroids, labels, phi, rho=0.001)
                            else:
                                exit('Not implemented solver')
                            time2 = time.monotonic()
                            SSE_without_noise = compute_sse(data, centroids)
                            print('Dataset:', dataset_name)
                            print("SSE in each iteration:\n", SSE)
                            if args.solver == 'rkm' or args.solver == 'fast_rkm':
                                print("tilted SSE in each iteration:\n", tilted_SSE)
                                output['tilted_SSE_iteration'] = tilted_SSE

                            print(f't={t}, k={k}, noise_frac={noise_frac}')
                            print('Running time:', time2-time1)
                            print('SSE without noise:', SSE_without_noise)
                            output['SSE_without_noise'] = SSE_without_noise
                            output['dataset'] = dataset_name
                            output['SSE_iteration'] = SSE
                            output['SSE'] = np.mean(SSE[-20:])/sample_size
                            output['tilted_SSE'] = min(tilted_SSE)
                            address = 'output/'
                            file_name = (address + dataset_name + '_t=' + str(t) + '_k=' + str(k) + '_seed=' +
                                         str(seed) + '_lr=' +str(lr)+'_epoch=' + str(num_epoch) + '_noise='+str(noise_frac) + '.json')
                            with open(file_name, "w") as dataf:
                                json.dump(output, dataf)
