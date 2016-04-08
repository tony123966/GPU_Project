#include "counting.h"
#include "SyncedMemory.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;
__device__ __host__ int CeilDiv(int a, int b)
{
	return (a - 1) / b + 1;
}
__device__ __host__ int CeilAlign(int a, int b)
{
	return CeilDiv(a, b) * b;
}

__constant__ char* text;
__global__ void EstablishButtonTree(const char* text, int* pos, int dataSize) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("tree :%d %d\n", idx, dataSize);
	if (idx < dataSize) {
		if (text[idx] == '\n') pos[idx] = 0;
		else pos[idx] = 1;
		//printf("i=%d,idx = %d, %d\n", dataSize,idx, pos[idx]);
	}
}
__global__ void EstablishLayerTree(int* layer_prev, int* layer, int layer_size_prev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("tree :%d %d\n", idx, layer_size);
	if (idx < layer_size_prev && idx % 2 == 0) {
		if (layer_prev[idx] & layer_prev[idx + 1])
			layer[idx/2] = layer_prev[idx] + layer_prev[idx+1];
		else
			layer[idx/2] = 0;
	}
	//printf("i=%d idx = %d, %d\n", layer_size_prev/2, idx, layer[idx]);
}
__device__ int xyToIndex(int y, int x, int text_size)
{
	int result = 0;
	for (int i = 0; i<y; i++) {
		result += (int)(text_size * pow(0.5, i));
	}
	result += x;
	return result;
}
__device__ int SetPositionNonRecursive(int curIdx, int *treeLayerBeginSize, int text_size)
{
	int curTreeIdx=0;
	int index=curIdx;
	int countNum=0;
	bool isTopDown=false;
	while (1){
		if (curIdx % 2 == 1 && !isTopDown)//右葉節點
		{
			if (treeLayerBeginSize[xyToIndex(curTreeIdx + 1, curIdx / 2, text_size)] == 0 && curTreeIdx == 0) { //若父親是0
				countNum+=1;
				return countNum;
			}
			if (curTreeIdx != 0)
			{
				while (1)//往父親走
				{
					curIdx /= 2; curTreeIdx += 1;
					if (curIdx % 2 == 0)//若父親是左節點
					{
						countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
						if (!curIdx) return countNum;
						if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx - 1, text_size)] == 0) { // 開始topDown
							isTopDown=true;
							break;
						}
						else//向左走
						{
							curIdx-=1;
						}
					}
					if (curIdx % 2 ==1)//若父親是右節點
					{
						if (treeLayerBeginSize[xyToIndex(curTreeIdx + 1, curIdx / 2, text_size)] == 0)
						{
							countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
							curIdx -= 1;
							// 開始topDown
							isTopDown=true;
							break;
						}
					}
					if (isTopDown)break;
				}
			}
			if (isTopDown)break;
		}
		else if (curIdx % 2 == 0 && !isTopDown)//左葉節點
		{
			countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
			if (!curIdx) return countNum;
			if (curTreeIdx == 0 && treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx - 1, text_size)] == 0)return countNum;
			if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx - 1, text_size)] != 0) {
				curIdx--;
			}

		}
		if (isTopDown)break;
	}
	if (isTopDown){
		while (1)
		{
			if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)] == 0 && curIdx % 2 == 0){//遇到0了且是左子點
				if (curTreeIdx == 0) return countNum;
				curTreeIdx += 1, curIdx / 2;
			}
			else if (curIdx % 2 == 1){ // right
				if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)] != 0){
					countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
					curIdx -=1;
				}
				else
				{
					if (curTreeIdx == 0) return countNum;
					curTreeIdx += 1, curIdx / 2;
				}
			}
		}
	}
	return countNum;
}
__device__ int SetPositionRecursive(int curTreeIdx, int curIdx, int *treeLayerBeginSize, bool isTopDown, int countNum, int text_size)
{
	if (!isTopDown)
	{
		if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)])
		{
			if (curTreeIdx == 0)//底層
			{
				if (curIdx % 2 == 0){//左子點
					countNum += 1;
					if (!curIdx) return countNum;
					if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx-1, text_size)] != 0) {
						return SetPositionRecursive(curTreeIdx, curIdx - 1, treeLayerBeginSize, false, countNum, text_size);
					}
					return countNum;
				}
				else if (curIdx % 2 == 1){// 右子點
					if (treeLayerBeginSize[xyToIndex(curTreeIdx + 1, curIdx / 2, text_size)] != 0) return SetPositionRecursive(curTreeIdx + 1, curIdx / 2, treeLayerBeginSize, false, countNum, text_size);
					else {
						if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)]) {
							countNum++;
							//return currentNum;
						}
						return countNum;
					}
				}	
			}
			else//非底層
			{
				if (curIdx % 2 == 0)//左子點
				{
					countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
					if (!curIdx) return countNum;
					if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx-1, text_size)] == 0) { // topDown
						return SetPositionRecursive(curTreeIdx, curIdx - 1, treeLayerBeginSize, true, countNum, text_size);
					}
					return	SetPositionRecursive(curTreeIdx, curIdx - 1, treeLayerBeginSize, false, countNum, text_size);
				}
				else if(curIdx % 2 == 1)// 右子點
				{
					if (treeLayerBeginSize[xyToIndex(curTreeIdx+1, curIdx/2, text_size)] != 0)
					{
						return SetPositionRecursive(curTreeIdx + 1, curIdx / 2, treeLayerBeginSize, false, countNum, text_size);
					}
					else{
						countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
						return SetPositionRecursive(curTreeIdx, curIdx - 1, treeLayerBeginSize, true, countNum, text_size);
					}
				}
			}	
		}
	}//topDown
	else{
		if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)] == 0 && curIdx % 2 == 0){//遇到0了且是左子點
			if (curTreeIdx == 0) return countNum;
			return SetPositionRecursive(curTreeIdx - 1, curIdx * 2 + 1, treeLayerBeginSize, true, countNum,text_size);
		}
		else if (curIdx % 2 == 1){ // right
			if (treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)] != 0){
				countNum += treeLayerBeginSize[xyToIndex(curTreeIdx, curIdx, text_size)];
				return SetPositionRecursive(curTreeIdx, curIdx - 1, treeLayerBeginSize, true, countNum, text_size);
			}
			else
			{
				if (curTreeIdx == 0) return countNum;
				return SetPositionRecursive(curTreeIdx - 1, curIdx * 2 + 1, treeLayerBeginSize, true, countNum, text_size);
			}
		}
	}
	return countNum;
}
__global__ void SetPosition(int  *allTree, int * treeResult, int text_size)
{
	/*
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d %d\n", idx, text_size);
	if (idx < text_size) {
		printf("ttt");
		int curIdx = idx;
		int height = 0;
		int count = 0;
		if (curIdx % 2 == 1)//右葉節點
		{
			if (curIdx / 2 >= text_size / 2)return;
			if (allTree[xyToIndex(height + 1, curIdx / 2, text_size)] == 0) { //若父親是0
				return;
			}
			else//若父親不是0
			{
				for (; height + 1 <= 9;)//不停往父親走
				{
					if (curIdx / 2 >= text_size / powf(2, height + 1))return;
					curIdx /= 2; height += 1;
					if (curIdx % 2 == 0)//若父親是左節點
					{
						if (curIdx - 1 < 0)//沒有左邊
						{
							return;
						}
						else//有左邊
						{
							if (allTree[xyToIndex(height, curIdx-1, text_size)] == 0)//左邊的是0
							{
								//開始topDown
								count += allTree[xyToIndex(height, curIdx, text_size)];
								curIdx -= 1;//往左邊平移
								for (; height - 1 > 0;)
								{
									curIdx = 2 * curIdx + 1; height -= 1;
									if (allTree[xyToIndex(height, curIdx, text_size)] != 0)
									{
										count += allTree[xyToIndex(height, curIdx, text_size)];
										allTree[xyToIndex(height, curIdx, text_size)] = count;
										return;
									}
								}
							}
							else//左邊的不是0
							{
								count += allTree[xyToIndex(height, curIdx, text_size)];
								curIdx -= 1;//往左邊平移
							}
						}
					}
					else//父親是右節點
					{
						if (allTree[xyToIndex(height, curIdx, text_size)] == 0)//找到0為止
						{
							curIdx = 2 * curIdx + 1;//走回子節點
							height -= 1;
							//開始topDown
							count += allTree[xyToIndex(height, curIdx, text_size)];
							if (curIdx - 1 < 0)//沒有左邊
							{
								return;
							}
							else{//有左邊
								curIdx -= 1;//往左邊平移
								for (; height - 1 > 0;)
								{
									curIdx = 2 * curIdx + 1; height -= 1;
									if (allTree[xyToIndex(height, curIdx, text_size)] != 0)
									{
										count += allTree[xyToIndex(height, curIdx, text_size)];
										allTree[xyToIndex(height, curIdx, text_size)] = count;
										return;
									}
								}
							}
						}
					}
				}
			}

		}
		else//左葉節點
		{
		}
	}
	*/
	
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < text_size)
	{
		treeResult[idx] = SetPositionRecursive(0, idx, allTree, false, 0, text_size);
		//treeResult[idx] = SetPositionNonRecursive(idx, allTree, text_size);
	}
}
__global__ void SetCount(int *pos, int  text_size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int count = 1;
	if (idx<text_size && pos[idx] == 1)
	{
		int index = idx;
		count = 1;
		while (index - 1 >= 0)
		{
			if (pos[index - 1]>0){
				count++;
				index -= 1;
			}
			else break;
		}
		pos[idx] = count;
	}
}
void CountPosition(const char *text, int *pos, int text_size)
{
	int * treeArray_gpu;
	int *treeLayer[10];

	for (int i = 0, size = text_size; i< 10; i++, size = size / 2){
		treeLayer[i] = (int *)malloc(sizeof(int)*size);
	}

	cudaMalloc((void**)& treeArray_gpu, sizeof(int)* text_size);
	cudaMemcpy(treeArray_gpu, treeLayer[0], sizeof(int)*text_size, cudaMemcpyHostToDevice);
	//底層
	int block_dim = text_size / 512 + 1;
	EstablishButtonTree << <block_dim, 512 >> >(text, treeArray_gpu, text_size);//想要用GPU計算一定要先cudaMalloc一段GPU記憶體空間
	cudaDeviceSynchronize();
	cudaMemcpy(treeLayer[0], treeArray_gpu, sizeof(int)*text_size, cudaMemcpyDeviceToHost);//將GPU算完的結果拿回CPU

	int totalSize = text_size;
	int treeButtomSize = 512;
	//上層
	for (int i = 1; i < 10; i++)
	{	
		int * treeArray_pre_gpu; 
		int * treeArray_now_gpu;
		cudaMalloc((void**)& treeArray_pre_gpu, sizeof(int)* totalSize);
		cudaMalloc((void**)& treeArray_now_gpu, sizeof(int)* totalSize/2);
		cudaMemcpy(treeArray_pre_gpu, treeLayer[i - 1], sizeof(int)*totalSize, cudaMemcpyHostToDevice);
		
		block_dim = (totalSize) / treeButtomSize + 1;
		EstablishLayerTree << <block_dim, treeButtomSize >> >(treeArray_pre_gpu, treeArray_now_gpu, totalSize);
		cudaDeviceSynchronize();

		cudaMemcpy(treeLayer[i], treeArray_now_gpu, sizeof(int)*totalSize / 2, cudaMemcpyDeviceToHost);
		if (totalSize == 1) break;
		treeButtomSize /= 2;
		totalSize /= 2;
	}
	//全塞到大ARRAY用一維陣列方式傳遞
	int * treeBigArray_cpu = (int *)malloc(sizeof(int)* text_size * 2);
	int * treeBigArray_gpu;
	cudaMalloc((void**)&treeBigArray_gpu, sizeof(int)*text_size * 2);
	
	int all_Idx=0;
	for (int i = 0, size = text_size; i<10; i++,size/=2)
	{
		int *treeTmp_cpu = (int *)malloc(sizeof(int)*size);
		cudaMemcpy(treeTmp_cpu, treeLayer[i], sizeof(int)*size, cudaMemcpyHostToHost);
		for (int j = 0; j<size; j++)
		{
			treeBigArray_cpu[all_Idx] = treeTmp_cpu[j];
			all_Idx++;
			printf("%d ", treeTmp_cpu[j]);
		}
		printf("\n");
	}
	cudaMemcpy(treeBigArray_gpu, treeBigArray_cpu, sizeof(int)* text_size * 2, cudaMemcpyHostToDevice);

	block_dim = text_size / 512 + 1;
	SetPosition << < text_size, 512 >> >(treeBigArray_gpu, pos, text_size);
	cudaDeviceSynchronize();
	/*
	int  *treeLayerBeginSize[10];
	int** treeArray;
	treeArray = (int **)malloc(10 * sizeof(int *));
	int size = text_size;
	for (int i = 0; i< 10; i++){
		treeArray[i] = (int *)malloc(sizeof(int)*size);
		treeLayerBeginSize[i] = treeArray[i];
		cudaMalloc((void**)&treeLayerBeginSize[i], sizeof(int)*size);
		cudaMemcpy(treeLayerBeginSize[i], treeArray[i], sizeof(int)*text_size, cudaMemcpyHostToDevice);
		size = size / 2;
	}
	int block_dim = text_size / 512+1;

	EstablishButtonTree << <block_dim, 512 >> >(text, treeLayerBeginSize[0], text_size);
	int totalSize = text_size;
	int treeButtomSize = 256;
	cudaDeviceSynchronize();
	for (int i = 1; i < 10; i++)
	{
		block_dim = (totalSize) / treeButtomSize+1;
		EstablishLayerTree << <block_dim, treeButtomSize >> >(treeLayerBeginSize[i - 1], treeLayerBeginSize[i], totalSize);
		cudaDeviceSynchronize();
		treeButtomSize /= 2;
		totalSize =  totalSize / 2;

	}
	 block_dim = text_size / 512+1;
	SetPosition << < text_size, 1 >> >(treeLayerBeginSize, text_size);
	//SetCount << <block_dim, 512 >> >(treeLayerBeginSize[0], text_size);
	cudaMemcpy(pos, treeLayerBeginSize[0], sizeof(int)*text_size, cudaMemcpyDeviceToHost);*/
}
template<int N>
class compare {
public:
	__device__ bool operator () (int x) { return x == N; }
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead = 0;
	cudaMalloc((void**)&buffer, sizeof(int)*text_size * 2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer + text_size);
	
	// TODO 如果pos_d是1就把thrust::counting_iterator<int>(text_size)內容存入head_d
	auto head_end_d =
	thrust::copy_if(
	thrust::counting_iterator<int>(0),
	thrust::counting_iterator<int>(text_size),
	pos_d,
	head_d,
	compare<1>()
	);
	nhead = head_end_d - head_d;
	cudaFree(buffer);
	return nhead;
}
__global__ void SomeTransform(char *text, int *pos, int textSize, int *head, int n_head) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < textSize ) {
		for (int i = 0; i<n_head;i++){//把每個頭設定7
		if (idx == head[i])pos[idx] = 7;
		}
	}
}
void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	int blockNum = text_size / 32+1;
	SomeTransform << <blockNum, 32 >> >(text, pos, text_size, head, n_head);
}
