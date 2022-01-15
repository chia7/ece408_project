#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


// #include <cuda_runtime.h>
// #include "cuda_fp16.h"

// #include <cublas_v2.h>
// #include <mma.h>
// using namespace nvcuda;

// ./rai -p ece408_project --queue rai_amd64_ece408
// ./rai -p ece408_project --queue rai_amd64_exclusive

#define TILE_WIDTH 16
#define TILE_WIDTH_1 16
#define TILE_WIDTH_2 16

#define TILE_WIDTH_K 32
#define TILE_WIDTH_X 16
#define TILE_WIDTH_RATIO (TILE_WIDTH_K / TILE_WIDTH_X)


__constant__ float const_k[196];
// __constant__ float const_k_2[3136];


// half *device_x_unroll;
// half *device_k_unroll;



__global__ void basicConvolution(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;

    if (h < H_out && w < W_out) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}


__global__ void unroll_k(const float *k, float *k_unroll, const int B, const int C, const int H, const int W, const int K)
{
    int bx = blockIdx.x, tx = threadIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;
    
    int h = ty;
    int w = tx;
    int m = by;
    int c = bx;
    int h_unroll = m;
    int w_unroll = c * K * K + ty * K + tx;

#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define ku2d(i1, i0)        k_unroll[(i1) * (C * K * K) + i0]

    ku2d(h_unroll, w_unroll) = k4d(m, c, h, w);

#undef k4d
#undef ku2d
}


__global__ void unroll_x(const float *x, float *x_unroll, const int B, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int bx = blockIdx.x, tx = threadIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;
    
    int h = bx / W_out + ty;
    int w = bx % W_out + tx;
    int c = by;
    int h_unroll = c * K * K + ty * K + tx;
    int w_unroll = bx;

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define xu3d(i2, i1, i0)    x_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]

    for (int b = 0; b < B; b++) {
        xu3d(b, h_unroll, w_unroll) = x4d(b, c, h, w);
    }

#undef x4d
#undef xu3d
}


__global__ void unrollTiledMatrixMultiply(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float subTileK[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileX[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int xuRow = C * K * K;
    int xuCol = H_out * W_out;

    int iter = (xuRow - 1) / TILE_WIDTH + 1;

    int m = row;
    int b = blockIdx.z;
    int K2 = K * K;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0.0f;

    for (int q = 0; q < iter; q++) {

        int tile_c = q * TILE_WIDTH + tx;
        int ck = tile_c / K2;
        int k_remain = tile_c % K2;
        int hk = k_remain / K;
        int wk = k_remain % K;

        subTileK[ty][tx] = 0.0f;
        if (row < M && tile_c < xuRow) {
            subTileK[ty][tx] = k4d(m, ck, hk, wk);
        }

        int tile_r = q * TILE_WIDTH + ty;
        int cx = tile_r / K2;
        k_remain = tile_r % K2;
        int hx = col / W_out + k_remain / K;
        int wx = col % W_out + k_remain % K;

        subTileX[ty][tx] = 0.0f;
        if (tile_r < xuRow && col < xuCol) {
            subTileX[ty][tx] = x4d(b, cx, hx, wx);
        }

        __syncthreads();

        if (row < M && col < xuCol) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                acc += subTileK[ty][i] * subTileX[i][tx];
                // acc += subTileK[i][ty] * subTileX[i][tx];
            }
            
        }
    
        __syncthreads();
    }

    if (row < M && col < xuCol) {
        y4d(b, row, col / W_out, col % W_out) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}



__global__ void basicConvolutionConstK(float *__restrict__ y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int  HO_WO = H_out * W_out;
    const int  M_HO_WO = M * HO_WO;
    const int  H_W = H * W;
    const int  C_H_W = C * H_W;
    const int  K_K = K * K;
    const int  C_K_K = C * K_K;

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) const_k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#define y4d(i3, i2, i1, i0) y[(i3) * (M_HO_WO) + (i2) * (HO_WO) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C_H_W) + (i2) * (H_W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) const_k[(i3) * (C_K_K) + (i2) * (K_K) + (i1) * (K) + i0]


    int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH_1 + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH_1 + threadIdx.x;
    int b = blockIdx.z;
    int c = 0;

    if (h < H_out && w < W_out) {
        // float acc = 0.0f;
        // for (int p = 0; p < K; p++) {
        //     // for (int q = 0; q < K; q++) {
        //     //     acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
        //     // }

        //     int offset = h + p;
        //     acc += x4d(b, c, offset, w) * k4d(m, c, p, 0)
        //         + x4d(b, c, offset, w + 1) * k4d(m, c, p, 1)
        //         + x4d(b, c, offset, w + 2) * k4d(m, c, p, 2)
        //         + x4d(b, c, offset, w + 3) * k4d(m, c, p, 3)
        //         + x4d(b, c, offset, w + 4) * k4d(m, c, p, 4)
        //         + x4d(b, c, offset, w + 5) * k4d(m, c, p, 5)
        //         + x4d(b, c, offset, w + 6) * k4d(m, c, p, 6);
        // }

        y4d(b, m, h, w) = x4d(b, c, h, w) * k4d(m, c, 0, 0) + x4d(b, c, h, w + 1) * k4d(m, c, 0, 1)
            + x4d(b, c, h, w + 2) * k4d(m, c, 0, 2) + x4d(b, c, h, w + 3) * k4d(m, c, 0, 3)
            + x4d(b, c, h, w + 4) * k4d(m, c, 0, 4) + x4d(b, c, h, w + 5) * k4d(m, c, 0, 5)
            + x4d(b, c, h, w + 6) * k4d(m, c, 0, 6) + x4d(b, c, h + 1, w) * k4d(m, c, 1, 0)
            + x4d(b, c, h + 1, w + 1) * k4d(m, c, 1, 1) + x4d(b, c, h + 1, w + 2) * k4d(m, c, 1, 2)
            + x4d(b, c, h + 1, w + 3) * k4d(m, c, 1, 3) + x4d(b, c, h + 1, w + 4) * k4d(m, c, 1, 4)
            + x4d(b, c, h + 1, w + 5) * k4d(m, c, 1, 5) + x4d(b, c, h + 1, w + 6) * k4d(m, c, 1, 6)
            + x4d(b, c, h + 2, w) * k4d(m, c, 2, 0) + x4d(b, c, h + 2, w + 1) * k4d(m, c, 2, 1)
            + x4d(b, c, h + 2, w + 2) * k4d(m, c, 2, 2) + x4d(b, c, h + 2, w + 3) * k4d(m, c, 2, 3)
            + x4d(b, c, h + 2, w + 4) * k4d(m, c, 2, 4) + x4d(b, c, h + 2, w + 5) * k4d(m, c, 2, 5)
            + x4d(b, c, h + 2, w + 6) * k4d(m, c, 2, 6) + x4d(b, c, h + 3, w) * k4d(m, c, 3, 0)
            + x4d(b, c, h + 3, w + 1) * k4d(m, c, 3, 1) + x4d(b, c, h + 3, w + 2) * k4d(m, c, 3, 2)
            + x4d(b, c, h + 3, w + 3) * k4d(m, c, 3, 3) + x4d(b, c, h + 3, w + 4) * k4d(m, c, 3, 4)
            + x4d(b, c, h + 3, w + 5) * k4d(m, c, 3, 5) + x4d(b, c, h + 3, w + 6) * k4d(m, c, 3, 6)
            + x4d(b, c, h + 4, w) * k4d(m, c, 4, 0) + x4d(b, c, h + 4, w + 1) * k4d(m, c, 4, 1)
            + x4d(b, c, h + 4, w + 2) * k4d(m, c, 4, 2) + x4d(b, c, h + 4, w + 3) * k4d(m, c, 4, 3)
            + x4d(b, c, h + 4, w + 4) * k4d(m, c, 4, 4) + x4d(b, c, h + 4, w + 5) * k4d(m, c, 4, 5)
            + x4d(b, c, h + 4, w + 6) * k4d(m, c, 4, 6) + x4d(b, c, h + 5, w) * k4d(m, c, 5, 0)
            + x4d(b, c, h + 5, w + 1) * k4d(m, c, 5, 1) + x4d(b, c, h + 5, w + 2) * k4d(m, c, 5, 2)
            + x4d(b, c, h + 5, w + 3) * k4d(m, c, 5, 3) + x4d(b, c, h + 5, w + 4) * k4d(m, c, 5, 4)
            + x4d(b, c, h + 5, w + 5) * k4d(m, c, 5, 5) + x4d(b, c, h + 5, w + 6) * k4d(m, c, 5, 6)
            + x4d(b, c, h + 6, w) * k4d(m, c, 6, 0) + x4d(b, c, h + 6, w + 1) * k4d(m, c, 6, 1)
            + x4d(b, c, h + 6, w + 2) * k4d(m, c, 6, 2) + x4d(b, c, h + 6, w + 3) * k4d(m, c, 6, 3)
            + x4d(b, c, h + 6, w + 4) * k4d(m, c, 6, 4) + x4d(b, c, h + 6, w + 5) * k4d(m, c, 6, 5)
            + x4d(b, c, h + 6, w + 6) * k4d(m, c, 6, 6);
        
        // y4d(b, m, h, w) = acc;
    }
   

#undef y4d
#undef x4d
#undef k4d
}


__global__ void unrollTiledMatrixMultiply2(float *__restrict__ y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float subTileK[TILE_WIDTH_2][TILE_WIDTH_2];
    __shared__ float subTileX[TILE_WIDTH_2][TILE_WIDTH_2];

    // __shared__ half2 subTileK[TILE_WIDTH_2][TILE_WIDTH_2];
    // __shared__ half2 subTileX[TILE_WIDTH_2][TILE_WIDTH_2];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int xuRow = C * K * K;
    int xuCol = H_out * W_out;

    int iter = (xuRow - 1) / TILE_WIDTH_2 + 1;

    int m = row;
    int b = blockIdx.z;
    int K2 = K * K;

    const int HO_WO = H_out * W_out;
    const int  M_HO_WO = M * HO_WO;
    const int  H_W = H * W;
    const int  C_H_W = C * H_W;
    const int  K_K = K * K;
    const int  C_K_K = C * K_K;

// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) const_k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#define y4d(i3, i2, i1, i0) y[(i3) * (M_HO_WO) + (i2) * (HO_WO) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C_H_W) + (i2) * (H_W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C_K_K) + (i2) * (K_K) + (i1) * (K) + i0]

    float acc = 0.0f;
    // half2 zero = __float2half2_rn(0.0);
    // half2 acc = zero;

    for (int q = 0; q < iter; q++) {

        int tile_c = q * TILE_WIDTH_2 + tx;
        int ck = tile_c / K2;
        int k_remain = tile_c % K2;
        int hk = k_remain / K;
        int wk = k_remain % K;


        subTileK[ty][tx] = 0.0f;
        // subTileK[ty][tx] = zero;
        if (row < M && tile_c < xuRow) {
            subTileK[ty][tx] = k4d(m, ck, hk, wk);
            // subTileK[ty][tx] = __float2half2_rn(k4d(m, ck, hk, wk));
        }

        int tile_r = q * TILE_WIDTH_2 + ty;
        int cx = tile_r / K2;
        k_remain = tile_r % K2;
        int hx = col / W_out + k_remain / K;
        int wx = col % W_out + k_remain % K;


        subTileX[ty][tx] = 0.0f;
        // subTileX[ty][tx] = zero;
        if (tile_r < xuRow && col < xuCol) {
            subTileX[ty][tx] = x4d(b, cx, hx, wx);
            // subTileX[ty][tx] = __float2half2_rn(x4d(b, cx, hx, wx));
        }

        __syncthreads();

        // if (row < M && col < xuCol) {
            // for (int i = 0; i < TILE_WIDTH; i++) {
            //     acc += subTileK[ty][i] * subTileX[i][tx];

                // acc += subTileK[i][ty] * subTileX[i][tx];
                

                // acc = __hfma2( subTileK[ty][i], subTileX[i][tx], acc);
                // acc = __hadd2(acc, __hmul2(subTileK[ty][i], subTileX[i][tx]));

            // }
            acc += subTileK[ty][0] * subTileX[0][tx] + subTileK[ty][1] * subTileX[1][tx]
                + subTileK[ty][2] * subTileX[2][tx] + subTileK[ty][3] * subTileX[3][tx]
                + subTileK[ty][4] * subTileX[4][tx] + subTileK[ty][5] * subTileX[5][tx]
                + subTileK[ty][6] * subTileX[6][tx] + subTileK[ty][7] * subTileX[7][tx]
                + subTileK[ty][8] * subTileX[8][tx] + subTileK[ty][9] * subTileX[9][tx]
                + subTileK[ty][10] * subTileX[10][tx] + subTileK[ty][11] * subTileX[11][tx]
                + subTileK[ty][12] * subTileX[12][tx] + subTileK[ty][13] * subTileX[13][tx]
                + subTileK[ty][14] * subTileX[14][tx] + subTileK[ty][15] * subTileX[15][tx];
            // acc = __hmul2( subTileK[ty], subTileX[tx]);
        // }
    
        __syncthreads();
    }

    if (row < M && col < xuCol) {
        y4d(b, row, col / W_out, col % W_out) = acc;
        // y4d(b, row, col / W_out, col % W_out) = __high2float(acc);
    }

#undef y4d
#undef x4d
#undef k4d
}


/*
__global__ void unrollTiledMatrixMultiply3(float *__restrict__ y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float tileK[TILE_WIDTH_2][TILE_WIDTH_2];
    __shared__ float tileX[TILE_WIDTH_2][TILE_WIDTH_2];

    __shared__ float n_tileK[TILE_WIDTH_2][TILE_WIDTH_2];
    __shared__ float n_tileX[TILE_WIDTH_2][TILE_WIDTH_2];


    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int xuRow = C * K * K;
    int xuCol = H_out * W_out;

    int iter = (xuRow - 1) / TILE_WIDTH_2 + 1;

    int m = row;
    int b = blockIdx.z;
    int K2 = K * K;

    const int HO_WO = H_out * W_out;
    const int  M_HO_WO = M * HO_WO;
    const int  H_W = H * W;
    const int  C_H_W = C * H_W;
    const int  K_K = K * K;
    const int  C_K_K = C * K_K;

#define y4d(i3, i2, i1, i0) y[(i3) * (M_HO_WO) + (i2) * (HO_WO) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C_H_W) + (i2) * (H_W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C_K_K) + (i2) * (K_K) + (i1) * (K) + i0]

    float acc = 0.0f;
    float **subTileK = &tileK[0][0], **subTileX = &tileX[0][0];
    float **n_subTileK = &n_tileK[0][0], **n_subTileX = &n_tileX[0][0];
    float **tmp;


    // prefetch

    int tile_c = tx;
    int ck = tile_c / K2;
    int k_remain = tile_c % K2;
    int hk = k_remain / K;
    int wk = k_remain % K;

    subTileK[ty][tx] = 0.0f;
    if (row < M && tile_c < xuRow) {
        subTileK[ty][tx] = k4d(m, ck, hk, wk);
    }

    int tile_r = ty;
    int cx = tile_r / K2;
    k_remain = tile_r % K2;
    int hx = col / W_out + k_remain / K;
    int wx = col % W_out + k_remain % K;

    subTileX[ty][tx] = 0.0f;
    if (tile_r < xuRow && col < xuCol) {
        subTileX[ty][tx] = x4d(b, cx, hx, wx);
    }

    __syncthreads();


    for (int q = 1; q < iter; q++) {

        tile_c = q * TILE_WIDTH_2 + tx;
        ck = tile_c / K2;
        k_remain = tile_c % K2;
        hk = k_remain / K;
        wk = k_remain % K;

        n_subTileK[ty][tx] = 0.0f;
        if (row < M && tile_c < xuRow) {
            n_subTileK[ty][tx] = k4d(m, ck, hk, wk);
        }

        tile_r = q * TILE_WIDTH_2 + ty;
        cx = tile_r / K2;
        k_remain = tile_r % K2;
        hx = col / W_out + k_remain / K;
        wx = col % W_out + k_remain % K;


        n_subTileX[ty][tx] = 0.0f;
        if (tile_r < xuRow && col < xuCol) {
            n_subTileX[ty][tx] = x4d(b, cx, hx, wx);
        }

        // __syncthreads();

        if (row < M && col < xuCol) {
            acc += subTileK[ty][0] * subTileX[0][tx] + subTileK[ty][1] * subTileX[1][tx]
                + subTileK[ty][2] * subTileX[2][tx] + subTileK[ty][3] * subTileX[3][tx]
                + subTileK[ty][4] * subTileX[4][tx] + subTileK[ty][5] * subTileX[5][tx]
                + subTileK[ty][6] * subTileX[6][tx] + subTileK[ty][7] * subTileX[7][tx]
                + subTileK[ty][8] * subTileX[8][tx] + subTileK[ty][9] * subTileX[9][tx]
                + subTileK[ty][10] * subTileX[10][tx] + subTileK[ty][11] * subTileX[11][tx]
                + subTileK[ty][12] * subTileX[12][tx] + subTileK[ty][13] * subTileX[13][tx]
                + subTileK[ty][14] * subTileX[14][tx] + subTileK[ty][15] * subTileX[15][tx];
        }

        __syncthreads();

        tmp = subTileK;
        subTileK = n_subTileK;
        n_subTileK = tmp;

        tmp = subTileX;
        subTileX = n_subTileX;
        n_subTileX = tmp;
    }

    if (row < M && col < xuCol) {
        y4d(b, row, col / W_out, col % W_out) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}
*/



__global__ void jointUnrollTiledMatrixMultiply(float *__restrict__ y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    __shared__ float subTileX[TILE_WIDTH_RATIO][TILE_WIDTH_X];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;
    int col = by * TILE_WIDTH_X;

    float y_reg[TILE_WIDTH_X];

    int xuRow = C * K * K;
    int xuCol = H_out * W_out;

    int iter = (xuCol - 1) / TILE_WIDTH_RATIO + 1;

    int m = row;
    int b = blockIdx.z;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    for (int outIdx = 0; outIdx < TILE_WIDTH_X; outIdx++) {
        y_reg[outIdx] = 0;
    }

    for (int q = 0; q < iter; q++) {

        int i = tx / TILE_WIDTH_X;
        int j = tx % TILE_WIDTH_X;

        int tile_r = q * TILE_WIDTH_RATIO + i;
        int tile_c = col + j;
        int cx = tile_r / (K * K);
        int k_remain = tile_r % (K * K);
        int hx = tile_c / W_out + k_remain / K;
        int wx = tile_c % W_out + k_remain % K;

        if (tile_r < xuRow && tile_c < xuCol) {
            subTileX[i][j] = x4d(b, cx, hx, wx);
        } else {
            subTileX[i][j] = 0.0f;
        }

        __syncthreads();


        for (int idx = 0; idx < TILE_WIDTH_RATIO; idx++) {
            float k_reg;

            tile_c = q * TILE_WIDTH_RATIO + idx;
            int ck = tile_c / (K * K);
            k_remain = tile_c % (K * K);
            int hk = k_remain / K;
            int wk = k_remain % K;

            if (row < M && tile_c < xuRow) {
                k_reg = k4d(m, ck, hk, wk);
            } else {
                k_reg = 0.0f;
            }

            for (int outIdx = 0; outIdx < TILE_WIDTH_X; outIdx++) {
                y_reg[outIdx] += k_reg * subTileX[idx][outIdx];
            }
        }
    
        __syncthreads();
    }

    for (int outIdx = 0; outIdx < TILE_WIDTH_X; outIdx++) {
        int outCol = col + outIdx;
        if (row < M && outCol < xuCol) {
            y4d(b, row, outCol / W_out, outCol % W_out) = y_reg[outIdx];
        }
    }

#undef y4d
#undef x4d
#undef k4d
}



__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;


    // Basic Convolution
    // Kernel Fusion Unroll + Tiled Matrix Multiply
    // Kernel Fusion Unroll + Tiled Matrix Multiply + 2 Kernel

/*
    const int sizeK = C * M * K * K * sizeof(float);
    const int sizeX = B * C * H * W * sizeof(float);
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMalloc((void **)device_k_ptr, sizeK);
    cudaMalloc((void **)device_x_ptr, sizeX);
    cudaMalloc((void **)device_y_ptr, sizeY);

    cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_x_ptr, host_x, sizeX, cudaMemcpyHostToDevice);
*/




    // 2 kernel + Const K

/*
    const int sizeK = C * M * K * K * sizeof(float);
    const int sizeX = B * C * H * W * sizeof(float);
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMalloc((void **)device_x_ptr, sizeX);
    cudaMalloc((void **)device_y_ptr, sizeY);

    if (M == 4) {
        cudaMemcpyToSymbol(const_k, host_k, sizeK);
    } else {    // M == 16
        cudaMalloc((void **)device_k_ptr, sizeK);
        cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(*device_x_ptr, host_x, sizeX, cudaMemcpyHostToDevice);
*/




    // Kernel Fusion Unroll + Tiled Matrix Multiply + 2 Kernel + Const + Overlapping


    const int sizeK = C * M * K * K * sizeof(float);
    const int sizeX = B * C * H * W * sizeof(float);
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMalloc((void **)device_x_ptr, sizeX);
    cudaMalloc((void **)device_y_ptr, sizeY);

    if (M == 4) {
        cudaMemcpyToSymbol(const_k, host_k, sizeK);
    } else {    // M == 16
        cudaMalloc((void **)device_k_ptr, sizeK);
        cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);
        // cudaMemcpyToSymbol(const_k_2, host_k, sizeK);
    }

    int iterB = 100;
    // if (M == 4) {
    //     iterB = 80;
    // }
    // if (B != 10000) {
    //     iterB = 10;
    // }
    
    const int lenB = B / iterB;
    const int lenx = lenB * C * H * W;
    const int sizex = lenx * sizeof(float);
    const int leny = lenB * M * H_out * W_out;
    const int sizey = leny * sizeof(float);


    cudaStream_t copyStreamIn, kernelStream, copyStreamOut;
    cudaStreamCreate(&copyStreamIn);
    cudaStreamCreate(&kernelStream);
    cudaStreamCreate(&copyStreamOut);


    cudaEvent_t waitX[iterB], waitY[iterB];
    // cudaEvent_t waitY[iterB];

    // cudaStream_t stream[iterB];


/*
    // cudaHostRegister((float *)host_x, sizeX, 0x01);
    cudaHostRegister((float *)host_y, sizeY / 4, 0x01);

    // cudaHostRegisterDefault      0x00
    // cudaHostRegisterPortable     0x01
    // cudaHostRegisterMapped       0x02
    // cudaHostRegisterIoMemory     0x04

    if (M == 4) {

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, lenB);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        for (int b = 0, offsetX = 0, offsetY = 0; b < iterB / 4; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);

            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

        // for (int b = 0; b < iterB; b++) {
        //     int offsetY = b * leny;
        //     cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
        //     cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        // }

        // cudaStream_t stream[iterB];

        // float *hy = (float *)host_y;

        // for (int b = 0, offsetX = 0, offsetY = 0; b < iterB; b++, offsetX += lenx, offsetY += leny) {
        //     cudaStreamCreate(&stream[b]);

        //     cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, stream[b]);
        //     basicConvolutionConstK<<<dimGrid, dimBlock, 0, stream[b]>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
        //     cudaMemcpyAsync(hy + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, stream[b]);
        // }

    } else {    // M == 16

        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, lenB);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        for (int b = 0, offsetX = 0, offsetY = 0; b < iterB / 4; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);
            
            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            unrollTiledMatrixMultiply2<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

        // for (int b = 0; b < iterB; b++) {
        //     int offsetY = b * leny;
        //     cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
        //     cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        // }

        // for (int b = 0; b < iterB; b++) {
        //     cudaStreamCreate(&stream[b]);

        //     int offsetX = b * lenx;
        //     int offsetY = b * leny;
        //     cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, stream[b]);
        //     unrollTiledMatrixMultiply2<<<dimGrid, dimBlock, 0, stream[b]>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
        //     cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, stream[b]);
        // }

    }

    cudaHostRegister((float *)host_y + B * M * H_out * W_out / 4, sizeY / 4, 0x01);

    if (M == 4) {

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, lenB);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        for (int b = iterB / 4, offsetX = b * lenx, offsetY = b * leny; b < iterB / 2; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);

            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

    } else {    // M == 16

        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, lenB);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        for (int b = iterB / 4, offsetX = b * lenx, offsetY = b * leny; b < iterB / 2; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);
            
            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            unrollTiledMatrixMultiply2<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

    }

    cudaHostRegister((float *)host_y + B * M * H_out * W_out / 2, sizeY / 4, 0x01);

    if (M == 4) {

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, lenB);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        for (int b = iterB / 2, offsetX = b * lenx, offsetY = b * leny; b < iterB / 4 * 3; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);

            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

    } else {    // M == 16

        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, lenB);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        for (int b = iterB / 2, offsetX = b * lenx, offsetY = b * leny; b < iterB / 4 * 3; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);
            
            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            unrollTiledMatrixMultiply2<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

    }

    cudaHostRegister((float *)host_y + B * M * H_out * W_out / 4 * 3, sizeY / 4, 0x01);

    if (M == 4) {

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, lenB);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        for (int b = iterB / 4 * 3, offsetX = b * lenx, offsetY = b * leny; b < iterB; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);

            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

    } else {    // M == 16

        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, lenB);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        for (int b = iterB / 4 * 3, offsetX = b * lenx, offsetY = b * leny; b < iterB; b++, offsetX += lenx, offsetY += leny) {
            cudaEventCreate(&waitX[b]);
            cudaEventCreate(&waitY[b]);

            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
            cudaEventRecord(waitX[b], copyStreamIn);
            
            cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            unrollTiledMatrixMultiply2<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            cudaEventRecord(waitY[b], kernelStream);

            cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        }

    }
*/

    const int iter = 4;
    for (int it = 0; it < iter; it++) {
        
        cudaHostRegister((float *)host_y + B * M * H_out * W_out / iter * it, sizeY / iter, 0x01);

        if (M == 4) {

            const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
            const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
            const int Y = H_grid * W_grid;

            dim3 dimGrid(M, Y, lenB);
            dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

            for (int b = iterB / iter * it, offsetX = b * lenx, offsetY = b * leny; b < iterB / iter * (it + 1); b++, offsetX += lenx, offsetY += leny) {
                cudaEventCreate(&waitX[b]);
                cudaEventCreate(&waitY[b]);

                cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
                cudaEventRecord(waitX[b], copyStreamIn);

                cudaStreamWaitEvent(kernelStream, waitX[b], 0);
                basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
                cudaEventRecord(waitY[b], kernelStream);

                cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
                cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
            }

        } else {    // M == 16

            dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, lenB);
            dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

            for (int b = iterB / iter * it, offsetX = b * lenx, offsetY = b * leny; b < iterB / iter * (it + 1); b++, offsetX += lenx, offsetY += leny) {
                cudaEventCreate(&waitX[b]);
                cudaEventCreate(&waitY[b]);

                cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
                cudaEventRecord(waitX[b], copyStreamIn);
                
                cudaStreamWaitEvent(kernelStream, waitX[b], 0);
                unrollTiledMatrixMultiply2<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
                cudaEventRecord(waitY[b], kernelStream);

                cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
                cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
            }

        }
    }


/*
    // Overlapping

    const int sizeK = C * M * K * K * sizeof(float);
    const int sizeX = B * C * H * W * sizeof(float);
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    const int iterB = 10;
    const int lenB = B / iterB;
    const int lenx = B / iterB * C * H * W;
    const int sizex = lenx * sizeof(float);
    const int leny = B / iterB * M * H_out * W_out;
    const int sizey = leny * sizeof(float);

    cudaMalloc((void **)device_k_ptr, sizeK);
    cudaMalloc((void **)device_x_ptr, sizeX);
    cudaMalloc((void **)device_y_ptr, sizeY);

    cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);

    // cudaStream_t copyStreamIn, kernelStream, copyStreamOut;
    // cudaStreamCreate(&copyStreamIn);
    // cudaStreamCreate(&kernelStream);
    // cudaStreamCreate(&copyStreamOut);

    // cudaEvent_t waitX[iterB], waitY[iterB];

    cudaStream_t stream[iterB];

    if (M == 4) {

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, lenB);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        // for (int b = 0; b < iterB; b++) {
        //     // cudaEventCreate(&waitX[b]);
        //     // cudaEventCreate(&waitY[b]);

        //     int offsetX = b * lenx;
        //     int offsetY = b * leny;

        //     cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
        //     // cudaEventRecord(waitX[b], copyStreamIn);

        //     // cudaStreamWaitEvent(kernelStream, waitX[b], 0);
        //     basicConvolution<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
        //     // cudaEventRecord(waitY[b], kernelStream);
        // }

        // for (int b = 0; b < iterB; b++) {
        //     int offsetY = b * leny;
        //     // cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
        //     cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        // }

        for (int b = 0; b < iterB; b++) {
            cudaStreamCreate(&stream[b]);

            int offsetX = b * lenx;
            int offsetY = b * leny;
            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, stream[b]);
            basicConvolution<<<dimGrid, dimBlock, 0, stream[b]>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, stream[b]);
        }

    } else {    // M == 16

        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, lenB);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        // for (int b = 0; b < iterB; b++) {
        //     cudaEventCreate(&waitX[b]);
        //     cudaEventCreate(&waitY[b]);

        //     int offsetX = b * lenx;
        //     int offsetY = b * leny;

        //     cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, copyStreamIn);
        //     cudaEventRecord(waitX[b], copyStreamIn);
            
        //     cudaStreamWaitEvent(kernelStream, waitX[b], 0);
        //     unrollTiledMatrixMultiply<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
        //     cudaEventRecord(waitY[b], kernelStream);
        // }

        // for (int b = 0; b < iterB; b++) {
        //     int offsetY = b * leny;
        //     cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
        //     cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, copyStreamOut);
        // }

        for (int b = 0; b < iterB; b++) {
            cudaStreamCreate(&stream[b]);

            int offsetX = b * lenx;
            int offsetY = b * leny;
            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizex, cudaMemcpyHostToDevice, stream[b]);
            unrollTiledMatrixMultiply<<<dimGrid, dimBlock, 0, stream[b]>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            cudaMemcpyAsync((float *)host_y + offsetY, *device_y_ptr + offsetY, sizey, cudaMemcpyDeviceToHost, stream[b]);
        }

    }
*/





/*
    // Kernel Fusion Unroll + Tiled Matrix Multiply + 2 Kernel

    const int sizeK = C * M * K * K * sizeof(float);
    const int sizeX = B * C * H * W * sizeof(float);
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMalloc((void **)device_k_ptr, sizeK);
    cudaMalloc((void **)device_x_ptr, sizeX);
    cudaMalloc((void **)device_y_ptr, sizeY);

    if (M == 4) {
        cudaMemcpyToSymbol(const_k, host_k, sizeK);
    } else {    // M == 16
        cudaMalloc((void **)device_k_ptr, sizeK);
        cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);
    }
*/

    // const int bSize = 80;
    // const int bIter = B / bSize;

    // const int sizeK = C * M * K * K * sizeof(float);
    // const int sizeX = B * C * H * W * sizeof(float);
    // const int sizeX = bSize * C * H * W * sizeof(float);
    // const int sizeY = B * M * H_out * W_out * sizeof(float);
    // const int sizeY = bSize * M * H_out * W_out * sizeof(float);


    // cudaMalloc((void **)device_x_ptr, sizeX * bIter);
    // cudaMalloc((void **)device_y_ptr, sizeY * bIter);

    // cudaMalloc((void **)device_x_ptr, sizeX);
    // cudaMalloc((void **)device_y_ptr, sizeY);

    // cudaMalloc((void **)device_k_ptr, sizeK);
    // cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);

/*    
    if (M == 4) {
        cudaMemcpyToSymbol(const_k, host_k, sizeK);

        // cudaMalloc((void **)device_y_ptr, sizeY);
    } else {    // M == 16
        cudaMalloc((void **)device_k_ptr, sizeK);
        cudaMemcpy(*device_k_ptr, host_k, sizeK, cudaMemcpyHostToDevice);




        // cudaMalloc((void **) &device_k_unroll, MATRIX_M * MATRIX_K * sizeof(half));
        // cudaMalloc((void **) &device_x_unroll, B * MATRIX_K * MATRIX_N * sizeof(half));

        // cudaMalloc((void **)device_y_ptr, MATRIX_M * MATRIX_N * sizeof(float));
    }
*/
    
    // cudaMemcpy(*device_x_ptr, host_x, sizeX, cudaMemcpyHostToDevice);



    // const int lenX = C * H * W;
    // const int lenY = M * H_out * W_out;

    // cudaStream_t copyStreamIn, kernelStream, copyStreamOut;
    // cudaStreamCreate(&copyStreamIn);
    // cudaStreamCreate(&kernelStream);
    // cudaStreamCreate(&copyStreamOut);


    // cudaStream_t stream[bIter];
    // for (int i = 0; i < bIter; i++) {
    //     cudaStreamCreate(&stream[i]);
    // }

    
    // float *x_ptr[bIter], *y_ptr[bIter];
    

    // cudaEvent_t waitX[bIter], waitY[bIter];
    

    // for (int i = 0; i < bIter; i++) {
        // cudaEventCreate(&waitX[i]);
        // cudaEventCreate(&waitY[i]);
    //     cudaMalloc((void **) &x_ptr[i], sizeX);
    //     cudaMalloc((void **) &y_ptr[i], sizeY);
    // }
/*    
    if (M == 4) {
        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, bSize);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        for (int b = 0; b < bIter; b++) {
            // cudaEventCreate(&waitX[b]);
            // cudaEventCreate(&waitY[b]);

            cudaStreamCreate(&stream[b]);

            // cudaMalloc((void **) &x_ptr[b], sizeX);
            // cudaMalloc((void **) &y_ptr[b], sizeY);

            int offsetX = b * bSize * lenX;
            int offsetY = b * bSize * lenY;
            // cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizeX, cudaMemcpyHostToDevice, copyStreamIn);
            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + offsetX, sizeX, cudaMemcpyHostToDevice, stream[b]);
            // cudaEventRecord(waitX[b], copyStreamIn);

            // cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            // basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr, *device_x_ptr, B, M, C, H, W, K);
            // basicConvolutionConstK<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
            basicConvolutionConstK<<<dimGrid, dimBlock, 0, stream[b]>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, B, M, C, H, W, K);
            // cudaEventRecord(waitY[b], kernelStream);

            // cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            // cudaMemcpyAsync((float *)host_y + b * bSize * lenY, *device_y_ptr, sizeY, cudaMemcpyDeviceToHost, copyStreamOut);
            cudaMemcpyAsync((float *)host_y + b * bSize * lenY, *device_y_ptr + offsetY, sizeY, cudaMemcpyDeviceToHost, stream[b]);
        }

        // for (int b = 0; b < bIter; b++) {
        //     int offsetY = b * bSize * lenY;
        //     cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
        //     cudaMemcpyAsync((float *)host_y + b * bSize * lenY, *device_y_ptr + offsetY, sizeY, cudaMemcpyDeviceToHost, copyStreamOut);
        // }
    } else {
        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, bSize);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        for (int b = 0; b < bIter; b++) {
            // cudaEventCreate(&waitX[b]);
            // cudaEventCreate(&waitY[b]);

            cudaStreamCreate(&stream[b]);

            // cudaMalloc((void **) &x_ptr[b], sizeX);
            // cudaMalloc((void **) &y_ptr[b], sizeY);

            int offsetX = b * bSize * lenX;
            int offsetY = b * bSize * lenY;

            // cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + b * bSize * lenX, sizeX, cudaMemcpyHostToDevice, copyStreamIn);
            cudaMemcpyAsync(*device_x_ptr + offsetX, host_x + b * bSize * lenX, sizeX, cudaMemcpyHostToDevice, stream[b]);
            // cudaEventRecord(waitX[b], copyStreamIn);
            
            // cudaStreamWaitEvent(kernelStream, waitX[b], 0);
            // unrollTiledMatrixMultiply<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr, *device_x_ptr, *device_k_ptr, B, M, C, H, W, K);
            // unrollTiledMatrixMultiply<<<dimGrid, dimBlock, 0, kernelStream>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            unrollTiledMatrixMultiply<<<dimGrid, dimBlock, 0, stream[b]>>>(*device_y_ptr + offsetY, *device_x_ptr + offsetX, *device_k_ptr, B, M, C, H, W, K);
            // cudaEventRecord(waitY[b], kernelStream);
            
            // cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
            // cudaMemcpyAsync((float *)host_y + b * bSize * lenY, *device_y_ptr, sizeY, cudaMemcpyDeviceToHost, copyStreamOut);
            cudaMemcpyAsync((float *)host_y + b * bSize * lenY, *device_y_ptr + offsetY, sizeY, cudaMemcpyDeviceToHost, stream[b]);
        }

        // for (int b = 0; b < bIter; b++) {
        //     int offsetY = b * bSize * lenY;
        //     cudaStreamWaitEvent(copyStreamOut, waitY[b], 0);
        //     cudaMemcpyAsync((float *)host_y + b * bSize * lenY, *device_y_ptr + offsetY, sizeY, cudaMemcpyDeviceToHost, copyStreamOut);
        // }
    }



*/

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // cudaMalloc((void **) &device_k_unroll, M * (C * K * K) * sizeof(float));
    // cudaMalloc((void **) &device_x_unroll, B * (C * K * K) * (H_out * W_out) * sizeof(float));
    // h_CUBLAS = (float *)malloc(sizeY);
    

    // cublasCreate(&cublasHandle);
    // cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel

    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;



    // Basic Convolution

/*
    const int W_grid = (W_out - 1) / TILE_WIDTH + 1;
    const int H_grid = (H_out - 1) / TILE_WIDTH + 1;
    const int Y = H_grid * W_grid;

    dim3 dimGrid(M, Y, B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    basicConvolution<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
*/




    // Kernel Fusion Unroll + Tiled Matrix Multiply

/*
    dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH + 1, (M - 1) / TILE_WIDTH + 1, B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    unrollTiledMatrixMultiply<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
*/




    // Kernel Fusion Unroll + Tiled Matrix Multiply + 2 Kernel

/*
    if (M == 4) {

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, B);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        basicConvolution<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    } else {    // M == 16
        
        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, B);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        unrollTiledMatrixMultiply<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    }
*/




    // 2 kernel + Const K

/*
    if (M == 4) {
        
        //  Basic Convolution + Const K

        const int W_grid = (W_out - 1) / TILE_WIDTH_1 + 1;
        const int H_grid = (H_out - 1) / TILE_WIDTH_1 + 1;
        const int Y = H_grid * W_grid;

        dim3 dimGrid(M, Y, B);
        dim3 dimBlock(TILE_WIDTH_1, TILE_WIDTH_1, 1);

        basicConvolutionConstK<<<dimGrid, dimBlock>>>(device_y, device_x, B, M, C, H, W, K);

    } else {
        
        // Kernel Fusion Tiled Matrix Multiply
        

        dim3 dimGrid((H_out * W_out - 1) / TILE_WIDTH_2 + 1, (M - 1) / TILE_WIDTH_2 + 1, B);
        dim3 dimBlock(TILE_WIDTH_2, TILE_WIDTH_2, 1);

        unrollTiledMatrixMultiply2<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    }
*/


    /*
     *  Kernel Fusion Tiled Matrix Multiply
     */

    // printf("B: %d, M: %d, C: %d, H: %d, W: %d, K: %d\n\n", B, M, C, H, W, K);

    // dim3 dimGrid(ceil(1.0 * H_out * W_out / TILE_WIDTH), ceil(1.0 * M / TILE_WIDTH), 1);
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // unrollTiledMatrixMultiply<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // constUnrollTiledMatrixMultiply<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    // constUnrollTiledMatrixMultiply2<<<dimGrid, dimBlock>>>(device_y, device_x, B, M, C, H, W, K);

    // dim3 dimGrid( (M + TILE_WIDTH_K - 1) / TILE_WIDTH_K, (H_out * W_out + TILE_WIDTH_X - 1) / TILE_WIDTH_X, 1);
    // dim3 dimBlock(TILE_WIDTH_K, 1, 1);

    // jointUnrollTiledMatrixMultiply<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);


    







    // dim3 dimGrid( (M + TILE_WIDTH_K - 1) / TILE_WIDTH_K, (H_out * W_out + TILE_WIDTH_X - 1) / TILE_WIDTH_X, B);
    // dim3 dimBlock(TILE_WIDTH_K, 1, 1);

    // jointUnrollTiledMatrixMultiply2<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);




    
    // float *device_x_unroll;
    // float *device_k_unroll;

    // int Hk = M, Wk = C * K * K, Wx = H_out * W_out;



}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host

    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;


    // Basic Convolution
    // Kernel Fusion Unroll + Tiled Matrix Multiply
    // Kernel Fusion Unroll + Tiled Matrix Multiply + 2 Kernel

/*
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMemcpy(host_y, device_y, sizeY, cudaMemcpyDeviceToHost);

    cudaFree(device_k);
    cudaFree(device_x);
    cudaFree(device_y);
*/




    // 2 kernel + Const K

/*
    const int sizeY = B * M * H_out * W_out * sizeof(float);

    cudaMemcpy(host_y, device_y, sizeY, cudaMemcpyDeviceToHost);

    if (M == 16) {
        cudaFree(device_k);
    }
    cudaFree(device_x);
    cudaFree(device_y);
*/

    if (M == 16) {
        cudaFree(device_k);
    }
    cudaFree(device_x);
    cudaFree(device_y);



    // const int sizeY = B * M * H_out * W_out * sizeof(float);
    // memcpy(host_y, hy, sizeY);
    // host_y = hy;

    // Free device memory


    // cudaFree(device_k_unroll);
    // cudaFree(device_x_unroll);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
