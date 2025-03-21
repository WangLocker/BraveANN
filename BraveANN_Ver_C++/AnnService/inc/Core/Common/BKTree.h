// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_BKTREE_H_
#define _SPTAG_COMMON_BKTREE_H_

#define XTENSOR_USE_XSIMD

#include <stack>
#include <string>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include "inc/Core/VectorIndex.h"

#include "CommonUtils.h"
#include "QueryResultSet.h"
#include "WorkSpace.h"
#include "Dataset.h"
#include "DistanceUtils.h"

#include <xsimd/xsimd.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xadapt.hpp>


#include <chrono>


namespace SPTAG
{
    namespace COMMON
    {
        // node type for storing BKT
        struct BKTNode
        {
            SizeType centerid;
            SizeType childStart;
            SizeType childEnd;

            BKTNode(SizeType cid = -1) : centerid(cid), childStart(-1), childEnd(-1) {}
        };

        struct DistanceLabelResult {
            std::vector<float> distances_min; 
            std::vector<int> labels;  
        };

        template <typename T>
        struct KmeansArgs {
            int _K;
            int _DK;
            DimensionType _D;
            DimensionType _RD;
            int _T;
            DistCalcMethod _M;
            T* centers;
            xt::xarray_adaptor<xt::xbuffer_adaptor<T*, xt::no_ownership>> centroids;
            T* newTCenters;
            SizeType* counts;
            float* newCenters;
            SizeType* newCounts;
            int* label;
            xt::xarray_adaptor<xt::xbuffer_adaptor<int*, xt::no_ownership>> labels;
            SizeType* clusterIdx;
            float* clusterDist;
            float* weightedCounts;
            float* newWeightedCounts;
            //added
            std::function<float(const T*, const T*, DimensionType)> fComputeDistance;
            const std::shared_ptr<IQuantizer>& m_pQuantizer;

            KmeansArgs(int k, DimensionType dim, SizeType datasize, int threadnum, DistCalcMethod distMethod, const std::shared_ptr<IQuantizer>& quantizer = nullptr) : _K(k), _DK(k), _D(dim), _RD(dim), _T(threadnum), _M(distMethod), m_pQuantizer(quantizer),
            centroids(xt::xbuffer_adaptor<T*, xt::no_ownership>(nullptr, 0), std::vector<std::size_t>{0, 0}),
            labels(xt::xbuffer_adaptor<int*, xt::no_ownership>(nullptr, 0), std::vector<std::size_t>{0, 0})
            {
                if (m_pQuantizer) {
                    _RD = m_pQuantizer->ReconstructDim();
                    fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(distMethod);
                }
                else {
                    fComputeDistance = COMMON::DistanceCalcSelector<T>(distMethod);
                }

                centers = (T*)ALIGN_ALLOC(sizeof(T) * _K * _D);
                newTCenters = (T*)ALIGN_ALLOC(sizeof(T) * _K * _D);
                counts = new SizeType[_K];
                newCenters = new float[_T * _K * _RD];
                newCounts = new SizeType[_T * _K];
                label = new int[datasize];
                clusterIdx = new SizeType[_T * _K];
                clusterDist = new float[_T * _K];
                weightedCounts = new float[_K];
                newWeightedCounts = new float[_T * _K];
                //added
                std::vector<std::size_t> shapeC = {_K, _D};
                centroids=xt::xarray_adaptor<xt::xbuffer_adaptor<T*, xt::no_ownership>>(
                    xt::xbuffer_adaptor<T*, xt::no_ownership>(centers, _K * _D),
                    shapeC
                );
                
                std::vector<std::size_t> shapeL = {datasize};
                labels=xt::xarray_adaptor<xt::xbuffer_adaptor<int*, xt::no_ownership>>(
                    xt::xbuffer_adaptor<int*, xt::no_ownership>(label, datasize),
                    shapeL
                );
            }

            ~KmeansArgs() {
                ALIGN_FREE(centers);
                ALIGN_FREE(newTCenters);
                delete[] counts;
                delete[] newCenters;
                delete[] newCounts;
                delete[] label;
                delete[] clusterIdx;
                delete[] clusterDist;
                delete[] weightedCounts;
                delete[] newWeightedCounts;
            }

            inline void ClearCounts() {
                memset(newCounts, 0, sizeof(SizeType) * _T * _K);
                memset(newWeightedCounts, 0, sizeof(float) * _T * _K);
            }

            inline void ClearCenters() {
                memset(newCenters, 0, sizeof(float) * _T * _K * _RD);
            }

            inline void ClearDists(float dist) {
                for (int i = 0; i < _T * _K; i++) {
                    clusterIdx[i] = -1;
                    clusterDist[i] = dist;
                }
            }

            void Shuffle(std::vector<SizeType>& indices, SizeType first, SizeType last) {
                SizeType* pos = new SizeType[_K];
                pos[0] = first;
                for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

                for (int k = 0; k < _K; k++) {
                    if (counts[k] == 0) continue;
                    SizeType i = pos[k];
                    while (newCounts[k] > 0) {
                        SizeType swapid = pos[label[i]] + newCounts[label[i]] - 1;
                        newCounts[label[i]]--;
                        std::swap(indices[i], indices[swapid]);
                        std::swap(label[i], label[swapid]);
                    }
                    while (indices[i] != clusterIdx[k]) i++;
                    std::swap(indices[i], indices[pos[k] + counts[k] - 1]);
                }
                delete[] pos;
            }

            void Shuffle_R3_noneed_Idx(std::vector<SizeType>& indices, SizeType first, SizeType last) {
                SizeType* pos = new SizeType[_K];
                pos[0] = first;
                for (int k = 1; k < _K; k++) pos[k] = pos[k - 1] + newCounts[k - 1];

                for (int k = 0; k < _K; k++) {
                    if (counts[k] == 0) continue;
                    SizeType i = pos[k];
                    while (newCounts[k] > 0) {
                        SizeType swapid = pos[label[i]] + newCounts[label[i]] - 1;
                        newCounts[label[i]]--;
                        std::swap(indices[i], indices[swapid]);
                        std::swap(label[i], label[swapid]);
                    }
                }
                delete[] pos;
            }
        };
        template<typename T>
        void RefineLambda(KmeansArgs<T>& args, float& lambda, int size)
        {
            int maxcluster = -1;
            SizeType maxCount = 0;
            for (int k = 0; k < args._DK; k++) {
                if (args.counts[k] > maxCount && args.newCounts[k] > 0)
                {
                    maxcluster = k;
                    maxCount = args.counts[k];
                }
            }

            float avgDist = args.newWeightedCounts[maxcluster] / args.newCounts[maxcluster];
            //lambda = avgDist / 10 / args.counts[maxcluster];
            //lambda = (args.clusterDist[maxcluster] - avgDist) / args.newCounts[maxcluster];
            lambda = (args.clusterDist[maxcluster] - avgDist) / size;
            if (lambda < 0) lambda = 0;
        }

        template <typename T, typename R>
        float RefineCenters(const Dataset<T>& data, KmeansArgs<T>& args)
        {
            int maxcluster = -1;
            SizeType maxCount = 0;
            for (int k = 0; k < args._DK; k++) {
                if (args.counts[k] > maxCount && args.newCounts[k] > 0 && DistanceUtils::ComputeDistance((T*)data[args.clusterIdx[k]], args.centers + k * args._D, args._D, DistCalcMethod::L2) > 1e-6)
                {
                    maxcluster = k;
                    maxCount = args.counts[k];
                }
            }

            if (maxcluster != -1 && (args.clusterIdx[maxcluster] < 0 || args.clusterIdx[maxcluster] >= data.R()))
                SPTAGLIB_LOG(Helper::LogLevel::LL_Debug, "maxcluster:%d(%d) Error dist:%f\n", maxcluster, args.newCounts[maxcluster], args.clusterDist[maxcluster]);

            float diff = 0;
            std::vector<R> reconstructVector(args._RD, 0);
            for (int k = 0; k < args._DK; k++) {
                T* TCenter = args.newTCenters + k * args._D;
                if (args.counts[k] == 0) {
                    if (maxcluster != -1) {
                        //int nextid = Utils::rand_int(last, first);
                        //while (args.label[nextid] != maxcluster) nextid = Utils::rand_int(last, first);
                        SizeType nextid = args.clusterIdx[maxcluster];
                        std::memcpy(TCenter, data[nextid], sizeof(T)*args._D);
                    }
                    else {
                        std::memcpy(TCenter, args.centers + k * args._D, sizeof(T)*args._D);
                    }
                }
                else {
                    float* currCenters = args.newCenters + k * args._RD;
                    for (DimensionType j = 0; j < args._RD; j++) {
                        currCenters[j] /= args.counts[k];
                    }
                    
                    if (args._M == DistCalcMethod::Cosine) {
                        COMMON::Utils::Normalize(currCenters, args._RD, COMMON::Utils::GetBase<T>());
                    }
                    
                    if (args.m_pQuantizer) {
                        for (DimensionType j = 0; j < args._RD; j++) reconstructVector[j] = (R)(currCenters[j]);
                        args.m_pQuantizer->QuantizeVector(reconstructVector.data(), (uint8_t*)TCenter);
                    }
                    else {
                        for (DimensionType j = 0; j < args._D; j++) TCenter[j] = (T)(currCenters[j]);
                    }
                }
                diff += DistanceUtils::ComputeDistance(TCenter, args.centers + k * args._D, args._D, DistCalcMethod::L2);
            }
            return diff;
        }

#if defined(NEWGPU)

#include "inc/Core/Common/cuda/Kmeans.hxx"

        template <typename T, typename R>
        inline float KmeansAssign(const Dataset<T>& data,
            std::vector<SizeType>& indices,
            const SizeType first, const SizeType last, KmeansArgs<T>& args,
            const bool updateCenters, float lambda) {
            float currDist = 0;
            SizeType totalSize = last - first;

// TODO - compile-time options for MAX_DIM and metric
            computeKmeansGPU<T, float, 100>(data, indices, first, last, args._K, args._D,
                                args._DK, lambda, args.centers, args.label, args.counts, args.newCounts, args.newCenters, 
                                args.clusterIdx, args.clusterDist, args.weightedCounts, args.newWeightedCounts, 0, updateCenters);
        }                               

#else

        template <typename T, typename R>
        inline float KmeansAssign(const Dataset<T>& data,
            std::vector<SizeType>& indices,
            const SizeType first, const SizeType last, KmeansArgs<T>& args, 
            const bool updateCenters, float lambda) {
            float currDist = 0;
            SizeType subsize = (last - first - 1) / args._T + 1;

#pragma omp parallel for num_threads(args._T) shared(data, indices) reduction(+:currDist)
            for (int tid = 0; tid < args._T; tid++)
            {
                SizeType istart = first + tid * subsize;
                SizeType iend = min(first + (tid + 1) * subsize, last);
                SizeType *inewCounts = args.newCounts + tid * args._K;
                float *inewCenters = args.newCenters + tid * args._K * args._RD;
                SizeType * iclusterIdx = args.clusterIdx + tid * args._K;
                float * iclusterDist = args.clusterDist + tid * args._K;
                float * iweightedCounts = args.newWeightedCounts + tid * args._K;
                float idist = 0;
                R* reconstructVector = nullptr;
                if (args.m_pQuantizer) reconstructVector = (R*)ALIGN_ALLOC(args.m_pQuantizer->ReconstructSize());

                for (SizeType i = istart; i < iend; i++) {
                    int clusterid = 0;
                    float smallestDist = MaxDist;
                    for (int k = 0; k < args._DK; k++) {
                        float dist = args.fComputeDistance(data[indices[i]], args.centers + k*args._D, args._D) + lambda*args.counts[k];
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k; smallestDist = dist;
                        }
                    }
                    args.label[i] = clusterid;
                    inewCounts[clusterid]++;
                    iweightedCounts[clusterid] += smallestDist;
                    idist += smallestDist;
                    if (updateCenters) {
                        if (args.m_pQuantizer) {
                            args.m_pQuantizer->ReconstructVector((const uint8_t*)data[indices[i]], reconstructVector);
                        }
                        else {
                            reconstructVector = (R*)data[indices[i]];
                        }
                        float* center = inewCenters + clusterid*args._RD;
                        for (DimensionType j = 0; j < args._RD; j++) center[j] += reconstructVector[j];

                        if (smallestDist > iclusterDist[clusterid]) {
                            iclusterDist[clusterid] = smallestDist;
                            iclusterIdx[clusterid] = indices[i];
                        }
                    }
                    else {
                        if (smallestDist <= iclusterDist[clusterid]) {
                            iclusterDist[clusterid] = smallestDist;
                            iclusterIdx[clusterid] = indices[i];
                        }
                    }
                }
                if (args.m_pQuantizer) ALIGN_FREE(reconstructVector);
                currDist += idist;
            }

            for (int i = 1; i < args._T; i++) {
                for (int k = 0; k < args._DK; k++) {
                    args.newCounts[k] += args.newCounts[i * args._K + k];
                    args.newWeightedCounts[k] += args.newWeightedCounts[i * args._K + k];
                }
            }

            if (updateCenters) {
                for (int i = 1; i < args._T; i++) {
                    float* currCenter = args.newCenters + i*args._K*args._RD;
                    for (size_t j = 0; j < ((size_t)args._DK) * args._RD; j++) args.newCenters[j] += currCenter[j];

                    for (int k = 0; k < args._DK; k++) {
                        if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] > args.clusterDist[k]) {
                            args.clusterDist[k] = args.clusterDist[i*args._K + k];
                            args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                        }
                    }
                }
            }
            else {
                for (int i = 1; i < args._T; i++) {
                    for (int k = 0; k < args._DK; k++) {
                        if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] <= args.clusterDist[k]) {
                            args.clusterDist[k] = args.clusterDist[i*args._K + k];
                            args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                        }
                    }
                }
            }
            return currDist;
        }

#endif


        template <typename T, typename R>
        inline float InitCenters(const Dataset<T>& data, 
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            KmeansArgs<T>& args, int samples, int tryIters) {
            SizeType batchEnd = min(first + samples, last);
            float lambda = 0, currDist, minClusterDist = MaxDist;
            for (int numKmeans = 0; numKmeans < tryIters; numKmeans++) {
                for (int k = 0; k < args._DK; k++) {
                    SizeType randid = COMMON::Utils::rand(last, first);
                    std::memcpy(args.centers + k*args._D, data[indices[randid]], sizeof(T)*args._D);
                }
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                currDist = KmeansAssign<T, R>(data, indices, first, batchEnd, args, true, 0);
                if (currDist < minClusterDist) {
                    minClusterDist = currDist;
                    memcpy(args.newTCenters, args.centers, sizeof(T)*args._K*args._D);
                    memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                    RefineLambda(args, lambda, batchEnd - first);
                }
            }
            return lambda;
        }


        inline float logsumexp(const xt::xarray<float>& arr) {
            float max_val = xt::amax(arr)();  
            return max_val + std::log(xt::sum(xt::exp(arr - max_val))());
        }

        inline xt::xarray<float> compute_tilted_sse_InEachCluster(const xt::xarray<float>& X,
                                                                const xt::xarray<float>& centroids,
                                                                const xt::xarray<int>& labels,
                                                                int k,
                                                                float t) {
            auto s0 = std::chrono::high_resolution_clock::now();
            int n_samples = X.shape(0);
            int n_features = X.shape(1);
            xt::xarray<float> phi = xt::zeros<float>({static_cast<size_t>(k)});
    
            auto selected_centroids = xt::view(centroids, xt::keep(labels), xt::all()); // centroids[labels]
            auto s1 = std::chrono::high_resolution_clock::now();
            xt::xarray<float> distances_to_centroids = xt::sum(xt::square(X - selected_centroids), {1});
            auto e1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> t1 = e1 - s1;

            for (int j = 0; j < k; ++j) {
                auto mask = xt::equal(labels, j);
                xt::xarray<float> mask_float = xt::cast<float>(mask);
                xt::xarray<float> masked_distances = t * distances_to_centroids * mask_float;
                phi(j) = (logsumexp(masked_distances) + std::log(1.0f / n_samples)) / t;
            }
            auto e0 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> t0 = e0 - s0;
            return phi;
        }

        

        inline float compute_tilted_sse(const xt::xarray<float>& X,
                                        const xt::xarray<float>& centroids,
                                        const xt::xarray<int>& labels,
                                        int k,
                                        float t,
                                        int n_samples) {
            xt::xarray<float> phi = xt::zeros<float>({static_cast<size_t>(k)});
    
            auto selected_centroids = xt::view(centroids, xt::keep(labels), xt::all()); // centroids[labels]
            xt::xarray<float> distances_to_centroids = xt::sum(xt::square(X - selected_centroids), {1});
            

            for (int j = 0; j < k; ++j) {
                auto mask = xt::equal(labels, j);
                xt::xarray<float> mask_float = xt::cast<float>(mask);
                xt::xarray<float> masked_distances = t * distances_to_centroids * mask_float;
                phi(j) = (logsumexp(masked_distances) + std::log(1.0f / n_samples)) / t;
            }
    
            float phi_sum = xt::sum(phi)();
            return phi_sum;
        }

        template <typename T, typename C, typename L, typename Phi>
        void r3km(T&& X,
                float t,
                int k,
                int num_epoch,
                float lr,
                C&& centroids,
                L&& labels,
                Phi&& phi,
                int num_batch = 50,
                int max_iter = 1000,
                float mu = 0.5){
            
            int n_samples=X.shape(0);
            for(int i=0;i<max_iter;i++){
                auto start_1_for = std::chrono::high_resolution_clock::now();
                for(int j=0;j<num_epoch;j++){
                    //test
                    auto batch_indices = xt::random::randint<int>({num_batch}, 0, n_samples);
                    auto batch = xt::view(X, xt::keep(batch_indices), xt::all());

                    auto batch_expanded = xt::view(batch, xt::all(), xt::newaxis(), xt::all());
                    auto distances_batch_expr = xt::norm_l2(batch_expanded - centroids, {2});
                    auto distances_batch = xt::eval(distances_batch_expr); 

                    auto distances_batch_min = xt::amin(distances_batch, {1});
                    auto sorted_indices = xt::argsort(distances_batch, 1);
                    auto labels_batch = xt::view(sorted_indices, xt::all(), 0);

                    auto selected_centroids = xt::view(centroids, xt::keep(labels_batch),xt::all());  // centroids[labels_batch]
                    xt::xarray<float> gradients = 2 * (batch - selected_centroids);

                    xt::xarray<float> phi_batch = compute_tilted_sse_InEachCluster(batch, centroids, labels_batch, k, t);

                    for (int j = 0; j < k; ++j) {
                        float phi_j_exp = std::exp(t * phi(j));
                        float phi_batch_j_exp = std::exp(t * phi_batch(j));
                        phi(j) = (1.0f / t) * std::log((1 - mu) * phi_j_exp + mu * phi_batch_j_exp);
                    }

                    auto phi_labels_batch = xt::index_view(phi, labels_batch);
                    auto weights_batch_expr = xt::exp(t * (distances_batch_min - phi_labels_batch)) / num_batch;
                    auto weights_batch = xt::eval(xt::reshape_view(weights_batch_expr, {num_batch, 1}));   

                    auto repeated_weights = xt::eval(xt::repeat(weights_batch, gradients.shape(1), 1));
                    xt::xarray<float> weighted_gradients = repeated_weights * gradients;

                    xt::xarray<float> delta_centroids = xt::zeros<float>({static_cast<size_t>(k), gradients.shape(1)});

                    for (std::size_t i = 0; i < labels_batch.size(); ++i) {
                        int cluster_id = labels_batch(i);
                        xt::view(delta_centroids, cluster_id, xt::all()) += xt::view(weighted_gradients, i, xt::all());
                    }

                    centroids += lr * delta_centroids;
                }
                auto end_1_for = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds1for = end_1_for - start_1_for;
                auto start_l_up = std::chrono::high_resolution_clock::now();


                auto s_l1 = std::chrono::high_resolution_clock::now();
                auto distances_expanded = xt::view(X, xt::all(), xt::newaxis(), xt::all());
                auto e_l1 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds_l1 = e_l1-s_l1;
                


                auto s_l2 = std::chrono::high_resolution_clock::now();
                auto distances_expr = xt::norm_l2(distances_expanded - centroids, {2});
                auto e_l2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds_l2 = e_l2-s_l2;

                auto s_l3 = std::chrono::high_resolution_clock::now();
                auto distances = xt::eval(distances_expr); 
                auto e_l3 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds_l3 = e_l3-s_l3;

                auto s_l4 = std::chrono::high_resolution_clock::now();
                auto distances_min = xt::amin(distances, {1});
                auto e_l4 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds_l4 = e_l4-s_l4;

                auto s_l5 = std::chrono::high_resolution_clock::now();
                auto sorted_indices_X = xt::argsort(distances, 1);
                auto e_l5 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds_l5 = e_l5-s_l5;

                auto s_l6 = std::chrono::high_resolution_clock::now();
                labels = xt::view(sorted_indices_X, xt::all(), 0);
                auto e_l6 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds_l6 = e_l6-s_l6;



                auto end_l_up = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_secondsupl = end_l_up-start_l_up;

                float tilted_SSE = compute_tilted_sse(X,centroids,labels,k,t,n_samples); 
                std::cout<<"iteration  "<<i<<"  "<<tilted_SSE<<std::endl;
            }


        }

        template <typename T>
        inline DistanceLabelResult CalculateBatchDistancesAndLabels(
            const Dataset<T>& data,
            std::vector<SizeType>::iterator indices_begin,  
            SizeType num_batch,                              
            const KmeansArgs<T>& args) {

            DistanceLabelResult result;
            result.distances_min.resize(num_batch);
            result.labels.resize(num_batch);

            int num_threads = args._T;  
            SizeType subsize = num_batch / num_threads;      
            SizeType remainder = num_batch % num_threads;    

            
            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            
            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

#pragma omp parallel for num_threads(num_threads) shared(data, indices_begin, result)
            for (int tid = 0; tid < num_threads; tid++) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];

                
                for (SizeType i = istart; i < iend; i++) {
                    SizeType data_index = *(indices_begin + i);
                    int clusterid = -1;
                    float smallestDist = MaxDist;

                    
                    for (int k = 0; k < args._K; k++) {
                        float dist = args.fComputeDistance(data[data_index], args.centers + k * args._D, args._D);
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k;
                            smallestDist = dist;
                        }
                    }

                    
                    result.labels[i] = clusterid;
                    result.distances_min[i] = std::sqrt(smallestDist);
                }
            }
            return result;
        }

        

        template <typename T, typename E>
        void kmeans_plus_plus_init(const xt::xexpression<E>& data_expr,int max_iters, KmeansArgs<T>& args) {
            int k = args._K;
            const auto& data = data_expr.derived_cast();
            int n_samples = data.shape(0);
            int n_features = data.shape(1);

            
            args.centroids = xt::zeros<T>({k, n_features});
            args.labels = xt::zeros<int>({n_samples});  

            
            int first_center_index = xt::random::randint<int>({1}, 0, n_samples)(0);
            xt::view(args.centroids, 0, xt::all()) = xt::view(data, first_center_index, xt::all());

            
            xt::xarray<T> min_distances = xt::ones<T>({n_samples}) * std::numeric_limits<T>::max();
            int iter_count = 0;

            for (int i = 1; i < k; ++i) {
                if (iter_count >= max_iters) {
                    std::cout << "end loop " << std::endl;
                    break;
                }

                for (int j = 0; j < n_samples; ++j) {
                    T dist = xt::norm_sq(xt::view(data, j, xt::all()) - xt::view(args.centroids, i - 1, xt::all()))();
                    min_distances(j) = std::min(min_distances(j), dist);
                }

                xt::xarray<T> probabilities = min_distances / xt::sum(min_distances)();

                T cumulative_prob = 0.0;
                float r = static_cast<float>(xt::random::rand<float>({1})(0));
                for (int j = 0; j < n_samples; ++j) {
                    cumulative_prob += probabilities(j);
                    if (r < cumulative_prob) {
                        xt::view(args.centroids, i, xt::all()) = xt::view(data, j, xt::all());
                        break;
                    }
                }
                iter_count++;
            }

            for (int j = 0; j < n_samples; ++j) {
                T min_dist = std::numeric_limits<T>::max();
                int closest_center = 0;
                for (int i = 0; i < k; ++i) {
                    T dist = xt::norm_sq(xt::view(data, j, xt::all()) - xt::view(args.centroids, i, xt::all()))();
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_center = i;
                    }
                }
                args.labels(j) = closest_center; 
            }
        }

        template <typename T>
        inline float CalculateDistancesToClosestCenter(const Dataset<T>& data,
                                        const std::vector<SizeType>& indices,
                                        const SizeType first,
                                        const SizeType last,
                                        const KmeansArgs<T>& args,
                                        std::vector<float>& distToClosestCenter) {
            float totalDist = 0.0f;
            SizeType subsize = (last - first - 1) / args._T + 1;

#pragma omp parallel for num_threads(args._T) reduction(+:totalDist) shared(distToClosestCenter)
            for(int tid=0;tid<args._T;tid++){
                SizeType istart = first + tid * subsize;
                SizeType iend = min(first + (tid + 1) * subsize, last);
                float idist = 0;

                for(SizeType i=istart;i<iend;i++){
                    float smallestDist = MaxDist;
                    for (int k = 0; k < args._DK; k++) {
                        float dist = args.fComputeDistance(data[indices[i]], args.centers + k*args._D, args._D);
                        if (dist > -MaxDist && dist < smallestDist) {
                            smallestDist = dist;
                        }
                    }
                    distToClosestCenter[i - first] = smallestDist;
                    idist += smallestDist;
                }
                totalDist+=idist;
            }
            return totalDist;
        }

        inline SizeType SelectNextCenter(const std::vector<float>& distToClosestCenter,
                          const SizeType first,
                          const SizeType last,
                          float totalDist) {
            std::vector<float> cumulativeDist(distToClosestCenter.size(), 0);
            cumulativeDist[0] = distToClosestCenter[0];

            for (SizeType i = 1; i < distToClosestCenter.size(); i++) {
                cumulativeDist[i] = cumulativeDist[i - 1] + distToClosestCenter[i];
            }   

            std::mt19937 gen(0);  
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            float target = dis(gen) * totalDist;

            auto it = std::lower_bound(cumulativeDist.begin(), cumulativeDist.end(), target);
            return first + std::distance(cumulativeDist.begin(), it);
        }   

        
        template <typename T, typename R>
        inline float KmeansAssign_normalKM(const Dataset<T>& data,
                          std::vector<SizeType>& indices,
                          const SizeType first, const SizeType last, KmeansArgs<T>& args, 
                          const bool updateCenters,
                          float& lambda) {
            float currDist = 0;
            SizeType subsize = (last - first - 1) / args._T + 1;
#pragma omp parallel for num_threads(args._T) shared(data, indices) reduction(+:currDist)
            for (int tid = 0; tid < args._T; tid++) {
                SizeType istart = first + tid * subsize;
                SizeType iend = std::min(first + (tid + 1) * subsize, last);

                SizeType *inewCounts = args.newCounts + tid * args._K;
                float *inewCenters = args.newCenters + tid * args._K * args._D;
                float idist = 0;
                R* reconstructVector = nullptr;

                for (SizeType i = istart; i < iend; i++) {
                    int clusterid = 0;
                    float smallestDist = MaxDist;
                    for (int k = 0; k < args._K; k++) {
                        float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D) + lambda*args.counts[k];
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k;
                            smallestDist = dist;
                        }
                    }

                
                    args.label[i] = clusterid;
                    inewCounts[clusterid]++;
                    idist += smallestDist;

                
                    if (updateCenters) {
                        reconstructVector = (R*)data[indices[i]];
                        float* center = inewCenters + clusterid * args._D;
                        for (DimensionType j = 0; j < args._D; j++) {
                            center[j] += reconstructVector[j];
                        }
                    }
                }
                currDist += idist;
            }

            
            for (int i = 1; i < args._T; i++) {
                for (int k = 0; k < args._K; k++) {
                    args.newCounts[k] += args.newCounts[i * args._K + k];
                }
            }

            if (updateCenters) {
                
                for (int i = 1; i < args._T; i++) {
                    float* currCenter = args.newCenters + i * args._K * args._D;
                    for (size_t j = 0; j < ((size_t)args._K) * args._D; j++) {
                        args.newCenters[j] += currCenter[j];
                    }
                }

                
                for (int k = 0; k < args._K; k++) {
                    if (args.newCounts[k] > 0) {
                        float* center = args.newCenters + k * args._D;
                        for (int j = 0; j < args._D; j++) {
                            center[j] /= args.newCounts[k];
                        }
                    }
                }
            }

            std::memcpy(args.centers, args.newCenters, sizeof(float)*args._K*args._D);
            std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

            return currDist;
        }


        template <typename T, typename R>
        inline float KmeansAssign_afterF(const Dataset<T>& data,
                          std::vector<SizeType>& indices,
                          const SizeType first, const SizeType last, KmeansArgs<T>& args, 
                          const bool updateCenters,
                          float lambda) {
            float currDist = 0;
            SizeType subsize = (last - first - 1) / args._T + 1;
#pragma omp parallel for num_threads(args._T) shared(data, indices) reduction(+:currDist)
            for (int tid = 0; tid < args._T; tid++) {
                SizeType istart = first + tid * subsize;
                SizeType iend = std::min(first + (tid + 1) * subsize, last);

                SizeType *inewCounts = args.newCounts + tid * args._K;
                float *inewCenters = args.newCenters + tid * args._K * args._D;
                float idist = 0;
                R* reconstructVector = nullptr;

                for (SizeType i = istart; i < iend; i++) {
                    int clusterid = 0;
                    float smallestDist = MaxDist;
                    for (int k = 0; k < args._K; k++) {
                        float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D) + lambda*args.counts[k];
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k;
                            smallestDist = dist;
                        }
                    }

                
                    args.label[i] = clusterid;
                    inewCounts[clusterid]++;
                    idist += smallestDist;

                
                    if (updateCenters) {
                        reconstructVector = (R*)data[indices[i]];
                        float* center = inewCenters + clusterid * args._D;
                        for (DimensionType j = 0; j < args._D; j++) {
                            center[j] += reconstructVector[j];
                        }
                    }
                }
                currDist += idist;
            }

            
            for (int i = 1; i < args._T; i++) {
                for (int k = 0; k < args._K; k++) {
                    args.newCounts[k] += args.newCounts[i * args._K + k];
                }
            }


            // std::memcpy(args.centers, args.newCenters, sizeof(float)*args._K*args._D);
            std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

            return currDist;
        }


        template <typename T, typename R>
        inline float KmeansAssign_normalKM_init(const Dataset<T>& data,
                          std::vector<SizeType>& indices,
                          const SizeType first, const SizeType last, KmeansArgs<T>& args, 
                          const bool updateCenters,
                          float lambda) {
            float currDist = 0;
            SizeType subsize = (last - first - 1) / args._T + 1;
#pragma omp parallel for num_threads(args._T) shared(data, indices) reduction(+:currDist)
            for (int tid = 0; tid < args._T; tid++) {
                SizeType istart = first + tid * subsize;
                SizeType iend = std::min(first + (tid + 1) * subsize, last);

                SizeType *inewCounts = args.newCounts + tid * args._K;
                float *inewCenters = args.newCenters + tid * args._K * args._D;
                SizeType * iclusterIdx = args.clusterIdx + tid * args._K;
                float * iclusterDist = args.clusterDist + tid * args._K;
                float * iweightedCounts = args.newWeightedCounts + tid * args._K;
                float idist = 0;
                R* reconstructVector = nullptr;

                for (SizeType i = istart; i < iend; i++) {
                    int clusterid = 0;
                    float smallestDist = MaxDist;
                    for (int k = 0; k < args._K; k++) {
                        float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D) + lambda*args.counts[k];
                        if (dist > -MaxDist && dist < smallestDist) {
                            clusterid = k;
                            smallestDist = dist;
                        }
                    }

                
                    args.label[i] = clusterid;
                    inewCounts[clusterid]++;
                    idist += smallestDist;

                
                    if (updateCenters) {
                        reconstructVector = (R*)data[indices[i]];
                        float* center = inewCenters + clusterid * args._D;
                        for (DimensionType j = 0; j < args._D; j++) {
                            center[j] += reconstructVector[j];
                        }
                        if (smallestDist > iclusterDist[clusterid]) {
                            iclusterDist[clusterid] = smallestDist;
                            iclusterIdx[clusterid] = indices[i];
                        }
                    }
                }
                currDist += idist;
            }

            
            for (int i = 1; i < args._T; i++) {
                for (int k = 0; k < args._K; k++) {
                    args.newCounts[k] += args.newCounts[i * args._K + k];
                    args.newWeightedCounts[k] += args.newWeightedCounts[i * args._K + k];
                }
            }

            if (updateCenters) {
                
                for (int i = 1; i < args._T; i++) {
                    float* currCenter = args.newCenters + i * args._K * args._D;
                    for (size_t j = 0; j < ((size_t)args._K) * args._D; j++) {
                        args.newCenters[j] += currCenter[j];
                    }
                    for (int k = 0; k < args._DK; k++) {
                        if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] > args.clusterDist[k]) {
                            args.clusterDist[k] = args.clusterDist[i*args._K + k];
                            args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                        }
                    }
                }

                
                for (int k = 0; k < args._K; k++) {
                    if (args.newCounts[k] > 0) {
                        float* center = args.newCenters + k * args._D;
                        for (int j = 0; j < args._D; j++) {
                            center[j] /= args.newCounts[k];
                        }
                    }
                }
            }

            return currDist;
        }

        
        template <typename T>
        inline float KmeansPPInitCenters(const Dataset<T>& data,
                                       std::vector<SizeType>& indices,
                                       const SizeType first,
                                       const SizeType last,
                                       KmeansArgs<T>& args) {

            std::srand(0);  
            SizeType randid = COMMON::Utils::rand(last, first);
            std::memcpy(args.centers, data[indices[randid]], sizeof(T) * args._D);
            args.counts[0] = 1;
            std::vector<float> distToClosestCenter(last - first, MaxDist);

            for (int k = 1; k < args._K; k++) {
                float totalDist = CalculateDistancesToClosestCenter(data, indices, first, last, args, distToClosestCenter);
                SizeType nextCenterIdx = SelectNextCenter(distToClosestCenter, first, last, totalDist);
                std::memcpy(args.centers + k * args._D, data[indices[nextCenterIdx]], sizeof(T) * args._D);
            }

            return 0.;
        }

        template <typename T,typename R>
        inline float KMeansPlusPlus(const Dataset<T>& data, std::vector<SizeType>& indices, SizeType first, SizeType last, KmeansArgs<T>& args, int maxIters, float& lambda) {
            KmeansPPInitCenters(data, indices, first, last, args);
            float prevDist = 0;
            float currDist = 0;
            float minClusterDist = MaxDist;
            int round2stop=0;
            for (int iter = 0; iter < maxIters; iter++) {

                args.ClearCenters();
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                
                currDist = KmeansAssign_normalKM_init<T,R>(data, indices, first, last, args, true, 0);
                if (std::abs(currDist - prevDist) <1e-3) {round2stop++;}
                if (round2stop==3){break;}
                prevDist = currDist;
                if (currDist < minClusterDist) {
                    round2stop=0;
                    minClusterDist = currDist;
                    // memcpy(args.newTCenters, args.centers, sizeof(T)*args._K*args._D);
                    // memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);
                    std::memcpy(args.centers, args.newCenters, sizeof(float)*args._K*args._D);
                    std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                    RefineLambda(args, lambda, (last-first)+1);
                }
            }
            return currDist;
        }

        template <typename BatchType, typename CentroidType>
        void inline CalGrad(BatchType&& batch, CentroidType&& selected_centroids, 
                    xt::xarray<float>& gradients, SizeType num_batch, int num_threads) {
                    SizeType subsize = num_batch / num_threads;      
                    SizeType remainder = num_batch % num_threads;    
            int n_feat = gradients.shape(1);

            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

#pragma omp parallel for num_threads(num_threads) shared(batch, selected_centroids, gradients)
            for (int tid = 0; tid < num_threads; tid++) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                for (SizeType i = istart; i < iend; i++) {
                    for(int j=0;j<n_feat;j++){
                        gradients(i,j)=2 * (batch(i,j) - selected_centroids(i,j));
                    }
                }
            }
        }

        inline xt::xarray<float> parallel_mask(
            const xt::xarray<float>& distances_batch_min,
            const xt::xarray<int>& labels_batch,
            int cluster_label,
            float t,
            int thr) {

            int num_threads = thr;  
            int num_batch = distances_batch_min.size();
            SizeType subsize = num_batch / num_threads;      
            SizeType remainder = num_batch % num_threads;    

            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

            xt::xarray<float> masked_distances = xt::zeros<float>({distances_batch_min.size()});

#pragma omp parallel for num_threads(num_threads) shared(distances_batch_min, labels_batch, masked_distances)
            for (int tid = 0; tid < num_threads; ++tid) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                for (SizeType i = istart; i < iend; ++i) {
                    masked_distances(i) = (labels_batch(i) == cluster_label) 
                        ? t * std::pow(distances_batch_min(i), 2) 
                        : 0.0f;
                }
            }

            return masked_distances;
        }

        inline float parallel_logsumexp(const xt::xarray<float>& masked_distances, int num_batch, float t, int num_threads, float& max) {
            SizeType num_elements = num_batch;
            auto max_val = xt::amax(masked_distances)();
            max=max_val;

            SizeType subsize = num_elements / num_threads;   
            SizeType remainder = num_elements % num_threads;    
            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; ++tid) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; ++tid) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

            float exp_sum = 0.0;

#pragma omp parallel for num_threads(num_threads) shared(masked_distances, max_val, thread_starts, thread_task_counts) reduction(+:exp_sum)
            for (int tid = 0; tid < num_threads; ++tid) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                float local_exp_sum = 0.0;  
                for (SizeType i = istart; i < iend; ++i) {
                        local_exp_sum += std::exp(masked_distances(i) - max_val);
                }
                exp_sum += local_exp_sum;  
            }

            return exp_sum;
        }


        inline xt::xarray<float> compute_phi_batch(
            const xt::xarray<float>& distances_batch_min, 
            const xt::xarray<int>& labels_batch,
            int k,
            float t,
            int num_batch,
            int thr) {

            xt::xarray<float> phi_batch = xt::zeros<float>({static_cast<size_t>(k)});
            xt::xarray<float> masked_distances = xt::zeros<float>({static_cast<size_t>(num_batch)});

            for (int j = 0; j < k; ++j) {
                masked_distances = parallel_mask(distances_batch_min, labels_batch, j, t, thr);
                float max=0.;
                float exp_sum = parallel_logsumexp(masked_distances, num_batch, t, thr, max);
                float logsumup = std::log(exp_sum)+max;
                phi_batch(j) = (logsumup + std::log(1.0f / num_batch)) / t;
                // phi_batch(j) = (logsumexp(masked_distances) + std::log(1.0f / num_batch)) / t;
            }
            return phi_batch;
        }

        template <typename T, typename R>
        float TryRClustering(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last,
            KmeansArgs<T>& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false, IAbortOperation* abort = nullptr) {

            kmeans_plus_plus_init(data.X, 1000, args);

            xt::xarray<float> phi=compute_tilted_sse_InEachCluster(data.X,args.centroids,args.labels,args._K,-0.1);

            r3km(data.X,-0.1,args._K,10,0.5,args.centroids,args.labels,phi,100,1000);

            // return 0;
            std::exit(0); 
            return 0.0f;  

        }


        template <typename PLB>
        void inline CalWeightsBatch(PLB&& phi_labels_batch, xt::xarray<float>& distance_batch_min, 
                    SizeType num_batch, float t, int num_threads, xt::xarray<float>& weights_batch) {

            SizeType subsize = num_batch / num_threads;      
            SizeType remainder = num_batch % num_threads;    

            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

#pragma omp parallel for num_threads(num_threads) shared(phi_labels_batch, distance_batch_min, weights_batch)
            for (int tid = 0; tid < num_threads; tid++) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                for (SizeType i = istart; i < iend; i++) {
                    weights_batch(i,0)=::exp((t*(distance_batch_min(i)-phi_labels_batch(i))))/num_batch;   
                }
            }
        }

        template <typename RW>
        void inline CalWeightsGrad(RW&& repeated_weights, xt::xarray<float>& gradients, xt::xarray<float>& weights_gradients, 
                    SizeType num_batch, int num_threads) {

            SizeType subsize = num_batch / num_threads;      
            SizeType remainder = num_batch % num_threads;  
            int n_feat = gradients.shape(1);  

            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

#pragma omp parallel for num_threads(num_threads) shared(repeated_weights, gradients, weights_gradients)
            for (int tid = 0; tid < num_threads; tid++) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                for (SizeType i = istart; i < iend; i++) {
                    float temp = repeated_weights(i,0);
                    for(int m=0;m<n_feat;m++){
                        weights_gradients(i,m)=temp*gradients(i,m);
                    }  
                }
            }
        }


        void inline CalDeltaCen(xt::xarray<float>& delta_centroids, xt::xarray<int>& labels_batch, xt::xarray<float>& weights_gradients, 
                    SizeType num_batch, int num_threads) {

            SizeType subsize = num_batch / num_threads;      
            SizeType remainder = num_batch % num_threads;  
            int n_feat = delta_centroids.shape(1);

            std::vector<SizeType> thread_task_counts(num_threads);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

#pragma omp parallel for num_threads(num_threads) shared(labels_batch, weights_gradients, delta_centroids)
            for (int tid = 0; tid < num_threads; tid++) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                for (SizeType i = istart; i < iend; i++) {
                    int cluster_id = labels_batch(i);
                    for(int m = 0;m<n_feat;m++){
                        delta_centroids(cluster_id,m) = delta_centroids(cluster_id,m)+weights_gradients(i,m);
                    }
                }
            }
        }


        template <typename T,typename BI>
        void inline GetBatchData(const Dataset<T>& data, 
                               BI&& batch_indices, int d, SizeType num_batch, int num_threads, xt::xarray<float>& batch) {

            SizeType subsize = num_batch / num_threads;      
            SizeType remainder = num_batch % num_threads;  

            std::vector<SizeType> thread_task_counts(num_threads, 0);
            for (int tid = 0; tid < num_threads; tid++) {
                thread_task_counts[tid] = subsize + (tid < remainder ? 1 : 0);  
            }

            std::vector<SizeType> thread_starts(num_threads + 1, 0);
            for (int tid = 1; tid <= num_threads; tid++) {
                thread_starts[tid] = thread_starts[tid - 1] + thread_task_counts[tid - 1];
            }

#pragma omp parallel for num_threads(num_threads) shared(batch_indices, data, batch)
            for (int tid = 0; tid < num_threads; tid++) {
                SizeType istart = thread_starts[tid];
                SizeType iend = thread_starts[tid + 1];
                for (SizeType i = istart; i < iend; i++) {
                    int index = batch_indices(i);
                    const float* row_ptr = reinterpret_cast<const float*>(data.At(index));
                    for (int j = 0; j < d; ++j) {
                        batch(i, j) = row_ptr[j]; 
                    }
                }
            }
        }



        template <typename T,typename R>
        float FastRKM_Up_Center(
                const Dataset<T>& data,
                std::vector<SizeType>& indices, 
                const SizeType first, 
                const SizeType last,
                KmeansArgs<T>& args,
                xt::xarray<float>& phi,
                float t,
                int num_epoch,
                float lr,
                float mu = 0.5,
                float lambda = 0.001
        ){
            int num_batch = (last - first + 19) / 20;
            int N = (last-first);

            xt::xarray<float> batch=xt::empty<float>({num_batch,args._D});

            for(int j=0;j<num_epoch;j++){
                std::shuffle(indices.begin() + first, indices.begin() + last, rg);
                int* start_ptr = &indices[first];
                auto batch_indices = xt::adapt(
                    start_ptr, 
                    {static_cast<size_t>(num_batch)}, 
                    xt::no_ownership()
                );

                
                if(num_batch>=32){
                    GetBatchData(data,batch_indices,args._D,num_batch,args._T,batch);
                }else{
                    for (int i = 0; i < num_batch; ++i) {
                        int index = batch_indices(i); 
                        const float* row_ptr = reinterpret_cast<const float*>(data.At(index)); 
                        for (int j = 0; j < args._D; ++j) {
                            batch(i, j) = row_ptr[j]; 
                        }
                    }
                }

                // batch = xt::view(data.X, xt::keep(batch_indices), xt::all());
                xt::xarray<float> distances_batch_min;
                xt::xarray<int> labels_batch;

                if(num_batch>=32){
                    DistanceLabelResult batch_minDist_Labels = CalculateBatchDistancesAndLabels(data, indices.begin() + first, num_batch, args);
                    distances_batch_min = xt::adapt(batch_minDist_Labels.distances_min, {num_batch});
                    labels_batch = xt::adapt(batch_minDist_Labels.labels, {num_batch});
                }else{
                    auto batch_expanded = xt::view(batch, xt::all(), xt::newaxis(), xt::all());
                    auto distances_batch_expr = xt::norm_l2(batch_expanded - args.centroids, {2});
                    auto distances_batch = xt::eval(distances_batch_expr); 
                    distances_batch_min = xt::amin(distances_batch, {1});
                    auto sorted_indices = xt::argsort(distances_batch, 1);
                    labels_batch = xt::view(sorted_indices, xt::all(), 0);
                }
                
                auto selected_centroids = xt::view(args.centroids, xt::keep(labels_batch),xt::all());  // centroids[labels_batch]
                xt::xarray<float> gradients = xt::zeros<float>(batch.shape());
                if(num_batch>=32){
                    CalGrad(batch,selected_centroids,gradients,num_batch,args._T);
                }else{
                    gradients = 2 * (batch - selected_centroids);
                }
                xt::xarray<float> phi_batch = xt::zeros<float>({static_cast<size_t>(args._K)});
                if(num_batch>=32){
                    phi_batch = compute_phi_batch(distances_batch_min,labels_batch,args._K,t,num_batch,args._T);
                }else{
                    phi_batch = compute_tilted_sse_InEachCluster(batch, args.centroids, labels_batch, args._K, t);
                } 
            
                for (int j = 0; j < args._K; ++j) {
                    float phi_j_exp = std::exp(t * phi(j));
                    float phi_batch_j_exp = std::exp(t * phi_batch(j));
                    phi(j) = (1.0f / t) * std::log((1 - mu) * phi_j_exp + mu * phi_batch_j_exp);
                }

                xt::xarray<float> weights_batch = xt::zeros<float>({num_batch, 1});
                auto phi_labels_batch = xt::index_view(phi, labels_batch);
                if(num_batch>=32){
                    CalWeightsBatch(phi_labels_batch,distances_batch_min,num_batch,t,args._T,weights_batch);
                }else{
                    auto weights_batch_expr = xt::exp(t * (distances_batch_min - phi_labels_batch)) / num_batch;
                    weights_batch = xt::eval(xt::reshape_view(weights_batch_expr, {num_batch, 1}));
                }
                xt::xarray<float> weighted_gradients;
                if(num_batch>=32){
                    CalWeightsGrad(weights_batch,gradients,weighted_gradients,num_batch,args._T);
                }else{
                    auto repeated_weights = xt::eval(xt::repeat(weights_batch, gradients.shape(1), 1));
                    weighted_gradients = repeated_weights * gradients;
                }
                xt::xarray<float> delta_centroids = xt::zeros<float>({static_cast<size_t>(args._K), gradients.shape(1)});
                if(num_batch>=32){
                    CalDeltaCen(delta_centroids,labels_batch,weighted_gradients,num_batch,args._T);
                }else{
                    for (std::size_t i = 0; i < labels_batch.size(); ++i) {
                        int cluster_id = labels_batch(i);
                        xt::view(delta_centroids, cluster_id, xt::all()) += xt::view(weighted_gradients, i, xt::all());
                    }
                }
                args.centroids += lr * delta_centroids;
                            
            }
            return 0;
        }

        template <typename T, typename R>
        float TryRClustering_new(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last,
            KmeansArgs<T>& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false, IAbortOperation* abort = nullptr) {
            float lambda=0.;
            //the SSE just after the init step
            float currDist=KMeansPlusPlus<T,R>(data,indices,first,last,args,100,lambda);

            if (abort && abort->ShouldAbort()) return 0;
            xt::xarray<float> phi = xt::zeros<float>({static_cast<size_t>(args._K)});

            xt::xarray<float> distances_batch_min;
            xt::xarray<int> labels_batch;
            DistanceLabelResult batch_minDist_Labels = CalculateBatchDistancesAndLabels(data, indices.begin() + first, (last-first), args);
            distances_batch_min = xt::adapt(batch_minDist_Labels.distances_min, {(last-first)});
            labels_batch = xt::adapt(batch_minDist_Labels.labels, {(last-first)});
            phi = compute_phi_batch(distances_batch_min,labels_batch,args._K,-0.01,(last-first),args._T);

            float minClusterDist = MaxDist;
            int noImprovement = 0;
            float originalLambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / lambdaFactor / (last - first);
            int rounds2stop =0;
            float prev_ttSSE=0;
            for (int iter = 0; iter < 20; iter++) {

                args.ClearCenters();
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                FastRKM_Up_Center<T,R>(data,indices,first,last,args,phi,-0.01f,1,0.05f,min(originalLambda,lambda));

                float newDist = KmeansAssign_afterF<T,R>(data, indices, first, last, args, false, min(originalLambda,lambda));
                if (std::abs(newDist-currDist)<1e-3){
                    rounds2stop++;
                }else{
                    rounds2stop=0;
                }
                currDist=newDist;
                if(rounds2stop==3){
                    break;
                }
            }

        }

        template <typename T, typename R>
        float TryClustering(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last,
            KmeansArgs<T>& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false, IAbortOperation* abort = nullptr) {

            float adjustedLambda = InitCenters<T, R>(data, indices, first, last, args, samples, 3);
            if (abort && abort->ShouldAbort()) return 0;

            SizeType batchEnd = min(first + samples, last);
            float currDiff, currDist, minClusterDist = MaxDist;
            int noImprovement = 0;
            float originalLambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / lambdaFactor / (batchEnd - first);
            for (int iter = 0; iter < 100; iter++) {
                std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
                std::shuffle(indices.begin() + first, indices.begin() + last, rg);

                args.ClearCenters();
                args.ClearCounts();
                args.ClearDists(-MaxDist);
                currDist = KmeansAssign<T, R>(data, indices, first, batchEnd, args, true, min(adjustedLambda, originalLambda));
                std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

                if (currDist < minClusterDist) {
                    noImprovement = 0;
                    minClusterDist = currDist;
                }
                else {
                    noImprovement++;
                }

                /*
                if (debug) {
                    std::string log = "";
                    for (int k = 0; k < args._DK; k++) {
                        log += std::to_string(args.counts[k]) + " ";
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f lambda:(%f,%f) counts:%s\n", iter, currDist, originalLambda, adjustedLambda, log.c_str());
                }
                */

                currDiff = RefineCenters<T, R>(data, args);
                //if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iter, currDist, currDiff);

                if (abort && abort->ShouldAbort()) return 0;
                if (currDiff < 1e-3 || noImprovement >= 5) break;
            }

            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign<T, R>(data, indices, first, last, args, false, 0);
            for (int k = 0; k < args._DK; k++) {
                if (args.clusterIdx[k] != -1) std::memcpy(args.centers + k * args._D, data[args.clusterIdx[k]], sizeof(T) * args._D);
            }

            args.ClearCounts();
            args.ClearDists(MaxDist);
            currDist = KmeansAssign<T, R>(data, indices, first, last, args, false, 0);
            std::memcpy(args.counts, args.newCounts, sizeof(SizeType) * args._K);

            SizeType maxCount = 0, minCount = (std::numeric_limits<SizeType>::max)(), availableClusters = 0;
            float CountStd = 0.0, CountAvg = (last - first) * 1.0f / args._DK;
            for (int i = 0; i < args._DK; i++) {
                if (args.counts[i] > maxCount) maxCount = args.counts[i];
                if (args.counts[i] < minCount) minCount = args.counts[i];
                CountStd += (args.counts[i] - CountAvg) * (args.counts[i] - CountAvg);
                if (args.counts[i] > 0) availableClusters++;
            }
            CountStd = sqrt(CountStd / args._DK) / CountAvg;
            if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Lambda:min(%g,%g) Max:%d Min:%d Avg:%f Std/Avg:%f Dist:%f NonZero/Total:%d/%d\n", originalLambda, adjustedLambda, maxCount, minCount, CountAvg, CountStd, currDist, availableClusters, args._DK);

            return CountStd;
        }

        template <typename T>
        float DynamicFactorSelect(const Dataset<T> & data,
            std::vector<SizeType> & indices, const SizeType first, const SizeType last,
            KmeansArgs<T> & args, int samples = 1000) {

            float bestLambdaFactor = 100.0f, bestCountStd = (std::numeric_limits<float>::max)();
            for (float lambdaFactor = 0.001f; lambdaFactor <= 1000.0f + 1e-3; lambdaFactor *= 10) {
                float CountStd = 0.0;
                if (args.m_pQuantizer)
                {
                    switch (args.m_pQuantizer->GetReconstructType())
                    {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
CountStd = TryClustering<T, Type>(data, indices, first, last, args, samples, lambdaFactor, true); \
break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                    default: break;
                    }
                }
                else
                {
                    CountStd = TryClustering<T, T>(data, indices, first, last, args, samples, lambdaFactor, true);
                }

                if (CountStd < bestCountStd) {
                    bestLambdaFactor = lambdaFactor;
                    bestCountStd = CountStd;
                }
            }
            /*
            std::vector<float> tries(16, 0);
            for (int i = 0; i < 8; i++) {
                tries[i] = bestLambdaFactor * (i + 2) / 10;
                tries[8 + i] = bestLambdaFactor * (i + 2);
            }
            for (float lambdaFactor : tries) {
                float CountStd = TryClustering(data, indices, first, last, args, samples, lambdaFactor, true);
                if (CountStd < bestCountStd) {
                    bestLambdaFactor = lambdaFactor;
                    bestCountStd = CountStd;
                }
            }
            */
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Best Lambda Factor:%f\n", bestLambdaFactor);
            return bestLambdaFactor;
        }

        template <typename T>
        int KmeansClustering(const Dataset<T>& data,
            std::vector<SizeType>& indices, const SizeType first, const SizeType last, 
            KmeansArgs<T>& args, int samples = 1000, float lambdaFactor = 100.0f, bool debug = false, IAbortOperation* abort = nullptr) {
            
            if (args.m_pQuantizer)
            {
                switch (args.m_pQuantizer->GetReconstructType())
                {
#define DefineVectorValueType(Name, Type) \
case VectorValueType::Name: \
TryClustering<T, Type>(data, indices, first, last, args, samples, lambdaFactor, debug, abort); \
break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                default: break;
                }
            }
            else
            {
                // TryClustering<T, T>(data, indices, first, last, args, samples, lambdaFactor, debug, abort);
                TryRClustering_new<T, T>(data, indices, first, last, args, samples, lambdaFactor, debug, abort);
            }

            if (abort && abort->ShouldAbort()) return 1;

            int numClusters = 0;
            for (int i = 0; i < args._K; i++) if (args.counts[i] > 0) numClusters++;

            if (numClusters <= 1) return numClusters;

            // args.Shuffle(indices, first, last);
            args.Shuffle_R3_noneed_Idx(indices,first,last);
            return numClusters;
        }

        class BKTree
        {
        public:
            BKTree(): m_iTreeNumber(1), m_iBKTKmeansK(32), m_iBKTLeafSize(8), m_iSamples(1000), m_fBalanceFactor(-1.0f), m_bfs(0), m_lock(new std::shared_timed_mutex), m_pQuantizer(nullptr) {}
            
            BKTree(const BKTree& other): m_iTreeNumber(other.m_iTreeNumber), 
                                   m_iBKTKmeansK(other.m_iBKTKmeansK), 
                                   m_iBKTLeafSize(other.m_iBKTLeafSize),
                                   m_iSamples(other.m_iSamples),
                                   m_fBalanceFactor(other.m_fBalanceFactor),
                                   m_lock(new std::shared_timed_mutex),
                                   m_pQuantizer(other.m_pQuantizer) {}
            ~BKTree() {}

            inline const BKTNode& operator[](SizeType index) const { return m_pTreeRoots[index]; }
            inline BKTNode& operator[](SizeType index) { return m_pTreeRoots[index]; }

            inline SizeType size() const { return (SizeType)m_pTreeRoots.size(); }
            
            inline SizeType sizePerTree() const {
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                return (SizeType)m_pTreeRoots.size() - m_pTreeStart.back(); 
            }

            inline const std::unordered_map<SizeType, SizeType>& GetSampleMap() const { return m_pSampleCenterMap; }

            inline void SwapTree(BKTree& newTrees)
            {
                m_pTreeRoots.swap(newTrees.m_pTreeRoots);
                m_pTreeStart.swap(newTrees.m_pTreeStart);
                m_pSampleCenterMap.swap(newTrees.m_pSampleCenterMap);
            }

            template <typename T>
            void Rebuild(const Dataset<T>& data, DistCalcMethod distMethod, IAbortOperation* abort)
            {
                BKTree newTrees(*this);
                newTrees.BuildTrees<T>(data, distMethod, 1, nullptr, nullptr, false, abort);

                std::unique_lock<std::shared_timed_mutex> lock(*m_lock);
                m_pTreeRoots.swap(newTrees.m_pTreeRoots);
                m_pTreeStart.swap(newTrees.m_pTreeStart);
                m_pSampleCenterMap.swap(newTrees.m_pSampleCenterMap);
            }

            template <typename T>
            void BuildTrees(const Dataset<T>& data, DistCalcMethod distMethod, int numOfThreads, 
                std::vector<SizeType>* indices = nullptr, std::vector<SizeType>* reverseIndices = nullptr, 
                bool dynamicK = false, IAbortOperation* abort = nullptr)
            {
                struct  BKTStackItem {
                    SizeType index, first, last;
                    bool debug;
                    BKTStackItem(SizeType index_, SizeType first_, SizeType last_, bool debug_ = false) : index(index_), first(first_), last(last_), debug(debug_) {}
                };
                std::stack<BKTStackItem> ss;

                std::vector<SizeType> localindices;
                if (indices == nullptr) {
                    localindices.resize(data.R());
                    for (SizeType i = 0; i < localindices.size(); i++) localindices[i] = i;
                }
                else {
                    localindices.assign(indices->begin(), indices->end());
                }
                KmeansArgs<T> args(m_iBKTKmeansK, data.C(), (SizeType)localindices.size(), numOfThreads, distMethod, m_pQuantizer);

                if (m_fBalanceFactor < 0) m_fBalanceFactor = DynamicFactorSelect(data, localindices, 0, (SizeType)localindices.size(), args, m_iSamples);

                m_pSampleCenterMap.clear();
                for (char i = 0; i < m_iTreeNumber; i++)
                {
                    std::shuffle(localindices.begin(), localindices.end(), rg);

                    m_pTreeStart.push_back((SizeType)m_pTreeRoots.size());
                    m_pTreeRoots.emplace_back((SizeType)localindices.size());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start to build BKTree %d\n", i + 1);

                    ss.push(BKTStackItem(m_pTreeStart[i], 0, (SizeType)localindices.size(), true));
                    while (!ss.empty()) {
                        if (abort && abort->ShouldAbort()) return;
                        BKTStackItem item = ss.top(); ss.pop();
                        m_pTreeRoots[item.index].childStart = (SizeType)m_pTreeRoots.size();
                        if (item.last - item.first <= m_iBKTLeafSize) {
                            for (SizeType j = item.first; j < item.last; j++) {
                                SizeType cid = (reverseIndices == nullptr)? localindices[j]: reverseIndices->at(localindices[j]);
                                m_pTreeRoots.emplace_back(cid);
                            }
                        }
                        else { // clustering the data into BKTKmeansK clusters
                            if (dynamicK) {
                                args._DK = std::min<int>((item.last - item.first) / m_iBKTLeafSize + 1, m_iBKTKmeansK);
                                args._DK = std::max<int>(args._DK, 2);
                            }
                            int numClusters = KmeansClustering(data, localindices, item.first, item.last, args, m_iSamples, m_fBalanceFactor, item.debug, abort);
                            if (numClusters <= 1) {
                                SizeType end = min(item.last + 1, (SizeType)localindices.size());
                                std::sort(localindices.begin() + item.first, localindices.begin() + end);
                                m_pTreeRoots[item.index].centerid = (reverseIndices == nullptr) ? localindices[item.first] : reverseIndices->at(localindices[item.first]);
                                m_pTreeRoots[item.index].childStart = -m_pTreeRoots[item.index].childStart;
                                for (SizeType j = item.first + 1; j < end; j++) {
                                    SizeType cid = (reverseIndices == nullptr) ? localindices[j] : reverseIndices->at(localindices[j]);
                                    m_pTreeRoots.emplace_back(cid);
                                    m_pSampleCenterMap[cid] = m_pTreeRoots[item.index].centerid;
                                }
                                m_pSampleCenterMap[-1 - m_pTreeRoots[item.index].centerid] = item.index;
                            }
                            else {
                                SizeType maxCount = 0;
                                for (int k = 0; k < m_iBKTKmeansK; k++) if (args.counts[k] > maxCount) maxCount = args.counts[k];
                                for (int k = 0; k < m_iBKTKmeansK; k++) {
                                    if (args.counts[k] == 0) continue;
                                    SizeType cid = (reverseIndices == nullptr) ? localindices[item.first + args.counts[k] - 1] : reverseIndices->at(localindices[item.first + args.counts[k] - 1]);
                                    m_pTreeRoots.emplace_back(cid);
                                    if (args.counts[k] > 1){
                                        ss.push(BKTStackItem((SizeType)(m_pTreeRoots.size() - 1), item.first, item.first + args.counts[k] - 1, item.debug && (args.counts[k] == maxCount)));
                                        push_num++;
                                    } 
                                    item.first += args.counts[k];
                                }
                            }
                        }
                        m_pTreeRoots[item.index].childEnd = (SizeType)m_pTreeRoots.size();
                    }
                    m_pTreeRoots.emplace_back(-1);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d BKTree built, %zu %zu\n", i + 1, m_pTreeRoots.size() - m_pTreeStart[i], localindices.size());
                }
            }

            inline std::uint64_t BufferSize() const
            {
                return sizeof(int) + sizeof(SizeType) * m_iTreeNumber +
                    sizeof(SizeType) + sizeof(BKTNode) * m_pTreeRoots.size();
            }

            ErrorCode SaveTrees(std::shared_ptr<Helper::DiskIO> p_out) const
            {
                std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                IOBINARY(p_out, WriteBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
                IOBINARY(p_out, WriteBinary, sizeof(SizeType) * m_iTreeNumber, (char*)m_pTreeStart.data());
                SizeType treeNodeSize = (SizeType)m_pTreeRoots.size();
                IOBINARY(p_out, WriteBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
                IOBINARY(p_out, WriteBinary, sizeof(BKTNode) * treeNodeSize, (char*)m_pTreeRoots.data());
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save BKT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode SaveTrees(std::string sTreeFileName) const
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save BKT to %s\n", sTreeFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return SaveTrees(ptr);
            }

            ErrorCode LoadTrees(char* pBKTMemFile)
            {
                m_iTreeNumber = *((int*)pBKTMemFile);
                pBKTMemFile += sizeof(int);
                m_pTreeStart.resize(m_iTreeNumber);
                memcpy(m_pTreeStart.data(), pBKTMemFile, sizeof(SizeType) * m_iTreeNumber);
                pBKTMemFile += sizeof(SizeType)*m_iTreeNumber;

                SizeType treeNodeSize = *((SizeType*)pBKTMemFile);
                pBKTMemFile += sizeof(SizeType);
                m_pTreeRoots.resize(treeNodeSize);
                memcpy(m_pTreeRoots.data(), pBKTMemFile, sizeof(BKTNode) * treeNodeSize);
                if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load BKT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode LoadTrees(std::shared_ptr<Helper::DiskIO> p_input)
            {
                IOBINARY(p_input, ReadBinary, sizeof(m_iTreeNumber), (char*)&m_iTreeNumber);
                m_pTreeStart.resize(m_iTreeNumber);
                IOBINARY(p_input, ReadBinary, sizeof(SizeType) * m_iTreeNumber, (char*)m_pTreeStart.data());

                SizeType treeNodeSize;
                IOBINARY(p_input, ReadBinary, sizeof(treeNodeSize), (char*)&treeNodeSize);
                m_pTreeRoots.resize(treeNodeSize);
                IOBINARY(p_input, ReadBinary, sizeof(BKTNode) * treeNodeSize, (char*)m_pTreeRoots.data());

                if (m_pTreeRoots.size() > 0 && m_pTreeRoots.back().centerid != -1) m_pTreeRoots.emplace_back(-1);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load BKT (%d,%d) Finish!\n", m_iTreeNumber, treeNodeSize);
                return ErrorCode::Success;
            }

            ErrorCode LoadTrees(std::string sTreeFileName)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load BKT From %s\n", sTreeFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sTreeFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return LoadTrees(ptr);
            }

            template <typename T>
            void InitSearchTrees(const Dataset<T>& data, std::function<float(const T*, const T*, DimensionType)> fComputeDistance, COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space) const
            {
                for (char i = 0; i < m_iTreeNumber; i++) {
                    const BKTNode& node = m_pTreeRoots[m_pTreeStart[i]];
                    if (node.childStart < 0) {
                        p_space.m_SPTQueue.insert(NodeDistPair(m_pTreeStart[i], fComputeDistance(p_query.GetQuantizedTarget(), data[node.centerid], data.C())));
                    } else if (m_bfs) {
                        float FactorQ = 1.1f;
                        int MaxBFSNodes = 100;
                        p_space.m_currBSPTQueue.Resize(MaxBFSNodes); p_space.m_nextBSPTQueue.Resize(MaxBFSNodes);
                        Heap<NodeDistPair>* p_curr = &p_space.m_currBSPTQueue, * p_next = &p_space.m_nextBSPTQueue;
                        p_curr->Top().distance = 1e9;
                       
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            _mm_prefetch((const char*)(data[m_pTreeRoots[begin].centerid]), _MM_HINT_T0);
                        }
                        
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            float dist = fComputeDistance(p_query.GetQuantizedTarget(), data[index], data.C());
                            if (dist <= FactorQ * p_curr->Top().distance && p_curr->size() < MaxBFSNodes) {
                                p_curr->insert(NodeDistPair(begin, dist));
                            }
                            else {
                                p_space.m_SPTQueue.insert(NodeDistPair(begin, dist));
                            }
                        }

                        for (int level = 1; level <= m_bfs; level++) {
                            p_next->Top().distance = 1e9;
                            while (!p_curr->empty()) {
                                NodeDistPair tmp = p_curr->pop();
                                const BKTNode& tnode = m_pTreeRoots[tmp.node];
                                if (tnode.childStart < 0) {
                                    p_space.m_SPTQueue.insert(tmp);
                                }
                                else {
                                    for (SizeType begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                                        _mm_prefetch((const char*)(data[m_pTreeRoots[begin].centerid]), _MM_HINT_T0);
                                    }
                                    if (!p_space.CheckAndSet(tnode.centerid)) {
                                        p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, tmp.distance));
                                    }
                                    for (SizeType begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                                        SizeType index = m_pTreeRoots[begin].centerid;
                                        float dist = fComputeDistance(p_query.GetQuantizedTarget(), data[index], data.C());
                                        if (dist <= FactorQ * p_next->Top().distance && p_next->size() < MaxBFSNodes) {
                                            p_next->insert(NodeDistPair(begin, dist));
                                        }
                                        else {
                                            p_space.m_SPTQueue.insert(NodeDistPair(begin, dist));
                                        }
                                    }
                                }
                            }
                            std::swap(p_curr, p_next);
                        }

                        while (!p_curr->empty()) {
                            p_space.m_SPTQueue.insert(p_curr->pop());
                        }
                    }
                    else {
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            _mm_prefetch((const char*)(data[m_pTreeRoots[begin].centerid]), _MM_HINT_T0);
                        }
                        for (SizeType begin = node.childStart; begin < node.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(NodeDistPair(begin, fComputeDistance(p_query.GetQuantizedTarget(), data[index], data.C())));
                        }
                    }
                }
            }

            template <typename T>
            void SearchTrees(const Dataset<T>& data, std::function<float(const T*, const T*, DimensionType)> fComputeDistance, COMMON::QueryResultSet<T> &p_query,
                COMMON::WorkSpace &p_space, const int p_limits) const
            {
                while (!p_space.m_SPTQueue.empty())
                {
                    NodeDistPair bcell = p_space.m_SPTQueue.pop();
                    const BKTNode& tnode = m_pTreeRoots[bcell.node];
                    if (tnode.childStart < 0) {
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_iNumberOfCheckedLeaves++;
                            p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, bcell.distance));
                        }
                        if (p_space.m_iNumberOfCheckedLeaves >= p_limits) break;
                    }
                    else {
                        for (SizeType begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                            _mm_prefetch((const char*)(data[m_pTreeRoots[begin].centerid]), _MM_HINT_T0);
                        }
                        if (!p_space.CheckAndSet(tnode.centerid)) {
                            p_space.m_NGQueue.insert(NodeDistPair(tnode.centerid, bcell.distance));
                        }
                        for (SizeType begin = tnode.childStart; begin < tnode.childEnd; begin++) {
                            SizeType index = m_pTreeRoots[begin].centerid;
                            p_space.m_SPTQueue.insert(NodeDistPair(begin, fComputeDistance(p_query.GetQuantizedTarget(), data[index], data.C())));
                        } 
                    }
                }
            }

        private:
            std::vector<SizeType> m_pTreeStart;
            std::vector<BKTNode> m_pTreeRoots;
            std::unordered_map<SizeType, SizeType> m_pSampleCenterMap;

        public:
            std::unique_ptr<std::shared_timed_mutex> m_lock;
            int m_iTreeNumber, m_iBKTKmeansK, m_iBKTLeafSize, m_iSamples, m_bfs;
            float m_fBalanceFactor;
            std::shared_ptr<SPTAG::COMMON::IQuantizer> m_pQuantizer;
        };
    }
}
#endif
