#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <c10/macros/Macros.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__global__ void att_grid_generator_gpu_kernel(const int n, 
                                              scalar_t *mapx,
                                              scalar_t *mapy, 
                                              scalar_t *map_xi,
                                              scalar_t *map_yi, 
                                              scalar_t *index_x, 
                                              scalar_t *index_y, 
                                              const int batch_size,
                                              const int att_size,
                                              const int out_size,
                                              const float threshold,
                                              const int iters)
{
    CUDA_KERNEL_LOOP(index, n) {
        const int b = index; 
        scalar_t *mapx_ptr = mapx + b * att_size * 1;
        scalar_t *mapy_ptr = mapy + b * att_size * 1;
        scalar_t *map_xi_ptr = map_xi + b * att_size * 1;
        scalar_t *map_yi_ptr = map_yi + b * att_size * 1;
        scalar_t *index_x_ptr = index_x + b * out_size * 1;
        scalar_t *index_y_ptr = index_y + b * out_size * 1;
        scalar_t threshold_s = static_cast<scalar_t>(threshold);
        scalar_t threshold_in_use = threshold_s;
        for (int i = 0; i < att_size; i++) {
           mapx_ptr[i] = mapx_ptr[i] * out_size;
           mapy_ptr[i] = mapy_ptr[i] * out_size; 
        }
        for (int j = 0; j < iters; j++) {
            scalar_t map_max_x = 0;
            scalar_t map_max_y = 0;
            for (int k = 0; k < att_size; k++) {
                map_max_x = map_max_x > mapx_ptr[k] ? map_max_x : mapx_ptr[k];
                map_max_y = map_max_y > mapy_ptr[k] ? map_max_y : mapy_ptr[k];
            }
            map_max_x = map_max_x > map_max_y ? map_max_y : map_max_x;
            threshold_in_use = map_max_x;
            if (j == 0)
                threshold_in_use = threshold_s > map_max_x ? map_max_x : threshold_s;
            for (int k = 0; k < att_size; k++) {
                mapx_ptr[k] = mapx_ptr[k] > threshold_in_use ? threshold_in_use : mapx_ptr[k];
                mapy_ptr[k] = mapy_ptr[k] > threshold_in_use ? threshold_in_use : mapy_ptr[k];
            }

            scalar_t sum_x = 0;
            scalar_t sum_y = 0;
            for (int k = 0; k < att_size; k++) {
                sum_x += mapx_ptr[k];
                sum_y += mapy_ptr[k];
            }

            scalar_t delta_x = (out_size - sum_x) / att_size;
            scalar_t delta_y = (out_size - sum_y) / att_size;

            for (int k = 0; k < att_size; k++) {
                mapx_ptr[k] += delta_x;
                mapy_ptr[k] += delta_y;
            }

        }

        for (int i = 0; i < att_size - 1; i++) {
            map_xi_ptr[i + 1] = map_xi_ptr[i] + mapx_ptr[i + 1];
            map_yi_ptr[i + 1] = map_yi_ptr[i] + mapy_ptr[i + 1];
        }

        scalar_t step_x = map_xi_ptr[att_size - 1] / out_size;
        scalar_t step_y = map_yi_ptr[att_size - 1] / out_size;
        int i = 0; 
        int j = 1;
        scalar_t myscale = 2.0 / (att_size - 1);
        
        while (i < out_size) {
            if (map_xi_ptr[j] >= i * step_x) {
                index_y_ptr[i] = (j + (i * step_x - map_xi_ptr[j]) / (map_xi_ptr[j] - map_xi_ptr[j-1])) * myscale - 1.0;
                i++;
            }
            else
                j++;
        }

        i = 0;
        j = 1;

        while (i < out_size) {
            if (map_yi_ptr[j] >= i * step_y) {
                index_x_ptr[i] = (j + (i * step_y - map_yi_ptr[j]) / (map_yi_ptr[j] - map_yi_ptr[j-1])) * myscale - 1.0;
                i++;
            }
            else    
                j++;
        }
    }
}


void attgridgen_gpu(const at::Tensor attx, const at::Tensor atty,
    at::Tensor map_xi, at::Tensor map_yi,
    at::Tensor index_x, at::Tensor index_y,
    const int batch_size, const int att_size, const int out_size, 
    const float threshold, const int iters)
{
    int num_kernels = batch_size;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        attx.type(), "att_grid_generator_gpu", ([&] {
            scalar_t *attx_ = attx.data<scalar_t>();
            scalar_t *atty_ = atty.data<scalar_t>();
            scalar_t *map_xi_ = map_xi.data<scalar_t>();
            scalar_t *map_yi_ = map_yi.data<scalar_t>();
            scalar_t *index_x_ = index_x.data<scalar_t>();
            scalar_t *index_y_ = index_y.data<scalar_t>();

            att_grid_generator_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, attx_, atty_, map_xi_, map_yi_, index_x_, index_y_, 
                batch_size, att_size, out_size, threshold, iters);

        })
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in att_grid_generator: %s\n", cudaGetErrorString(err));
    }

}
