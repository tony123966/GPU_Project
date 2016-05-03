#include "lab3.h"
#include <cstdio>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define R 0
#define G 1
#define B 2
__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb * 3 + R] = target[curt * 3 + R];
			output[curb * 3 + G] = target[curt * 3 + G];
			output[curb * 3 + B] = target[curt * 3 + B];
		}
	}
}
__global__ void CalculateFixed(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;

	float targetNeiborSum = 0;
	float backgroundSum = 0;
	if (yt < ht && xt < wt) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		
		for (int i = 0; i < 3; i++){
			targetNeiborSum=0;
			backgroundSum=0;
			/*
			targetNeiborLeft = (curt - 1 < 0) ? target[curt * 3 + i] : target[curt * 3 + i - 1 * 3];
			targetNeiborRight = (curt + 1 > wt*ht) ? target[curt * 3 + i] : target[curt * 3 + i + 1 * 3];
			targetNeiborUp = (curt - wt < 0) ? target[curt * 3 + i] : target[curt * 3 + i - wt * 3];
			targetNeiborDown = (curt + wt > wt*ht) ? target[curt * 3 + i] : target[curt * 3 + i + wt * 3];
					
			targetNeiborSum = targetNeiborLeft + targetNeiborRight + targetNeiborUp + targetNeiborDown;
			*/
			
			// 左
			if (curt - 1 < 0)
			{
				backgroundSum += background[curb * 3 + i - 1 * 3];
				targetNeiborSum += target[curt * 3 + i];//target的neibor超出target邊界 target不加neibor位置 是加自己
			}
			else
			{
				backgroundSum += mask[curt - 1] > 127.0 ? 0 : background[curb * 3 + i - 1 * 3];
				targetNeiborSum += target[curt * 3 + i - 1 * 3];
			}
			// 右
			if (curt + 1 > wt*ht) 
			{
				backgroundSum += background[curb * 3 + i + 1 * 3];
				targetNeiborSum += target[curt * 3 + i];//target的neibor超出target邊界 target不加neibor位置 是加自己
			}
			else
			{
				backgroundSum += mask[curt + 1] > 127.0 ? 0 : background[curb * 3 + i + 1 * 3];
				targetNeiborSum += target[curt * 3 + i + 1 * 3];
			}
			//上
			if (curt - wt < 0)
			{
				backgroundSum += background[curb * 3 + i - wb * 3];
				targetNeiborSum += target[curt * 3 + i];//target的neibor超出target邊界 target不加neibor位置 是加自己
			}
			else
			{
				backgroundSum += mask[curt - wt] > 127.0 ? 0 : background[curb * 3 + i - wb * 3];
				targetNeiborSum += target[curt * 3 + i - wt * 3];

			}
			//下
			if (curt + wt > wt*ht)
			{
				backgroundSum += background[curb * 3 + i + wb * 3];
				targetNeiborSum += target[curt * 3 + i];//target的neibor超出target邊界 target不加neibor位置 是加自己
			}
			else
			{
				backgroundSum += mask[curt + wt] > 127.0 ? 0 : background[curb * 3 + i + wb * 3];
				targetNeiborSum += target[curt * 3 + i + wt * 3];
			}
			/*
			targetNeiborSum = targetNeiborLeft + targetNeiborRight + targetNeiborUp + targetNeiborDown;
			backgroundSum = backgroundNeiborLeft + backgroundNeiborRight + backgroundNeiborUp + backgroundNeiborDown;*/
			fixed[curt * 3 + i] = 4.0 * target[curt * 3 + i] - targetNeiborSum + backgroundSum;
		}
	}

}
__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	float *target,
	float *output,
	const int wt, const int ht
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;

	float OutputNeiborSum=0;
	float w=1.8;//SOR paramater

	// w = 1 + (MAX_TIMES - time) / MAX_TIMES

	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		//Run 
		for (int i = 0; i < 3; i++){

			OutputNeiborSum = 0.0f;

			if (curt - 1 >= 0 && mask[curt - 1] > 127.0f) // 左
			{
				OutputNeiborSum += target[curt * 3 + i - 1 * 3];
			}
			if (curt + 1 < wt*ht && mask[curt + 1] > 127.0f) // 右
			{
				OutputNeiborSum += target[curt * 3 + i + 1 * 3];
			}
			if (curt - wt >= 0 && mask[curt - wt] > 127.0f) //上
			{
				OutputNeiborSum += target[curt * 3 + i - wt * 3];
			}
			if (curt + wt < wt*ht && mask[curt + wt] > 127.0f) //下
			{
				OutputNeiborSum += target[curt * 3 + i + wt * 3];
			}
			output[curt * 3 + i] = (1.0 / 4.0)*(fixed[curt * 3 + i] + OutputNeiborSum)*w + (1.0 - w)*output[curt * 3 + i];
		}
		
	}
}
void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3 * wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3 * wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3 * wt*ht*sizeof(float));

	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

	CalculateFixed << <gdim, bdim >> >(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
		);

	cudaMemcpy(buf1, target, sizeof(float)* 3 * wt*ht, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < 5000; ++i)
	{
		PoissonImageCloningIteration << <gdim, bdim >> >(fixed, mask, buf1, buf2, wt, ht);
		PoissonImageCloningIteration << <gdim, bdim >> >(fixed, mask, buf2, buf1, wt, ht);
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)* 3, cudaMemcpyDeviceToDevice);


	SimpleClone << <gdim, bdim >> >(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
		);

	/*SimpleClone << <dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >> >(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
		);*/

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
