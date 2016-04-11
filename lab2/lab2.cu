

#include "lab2.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;
#define bound(x) (x>255?255:x<0?0:x)
#define RGB2Y(R,G,B) bound((0.299*R)+(0.587*G)+(0.114*B))
#define RGB2U(R,G,B) bound((-0.169*R)+(-0.331*G)+(0.500*B)+128)
#define RGB2V(R,G,B) bound((0.500*R)+(-0.419*G)+(-0.081*B)+128)
#define M_PI  3.14
#define Two_M_PI  3.14*2
struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator() : impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};/*
  __device__ void Line(uint8_t *yuv, int x1, int y1, int x2, int y2, Vector3D color)
  {
  int idx = y1*W + x1;

  int rowOfY = (idx / W);
  int columnOfY = (idx%W);

  int rowOfUV = rowOfY / 2;
  int columnOfUV = columnOfY / 2;

  int uvWidth = W / 2;
  int diffX = abs(x1 - x2);
  int diffY = abs(y1 - y2);
  for (int i = 0; i<diffX; i++){
  for (int j = 0; j<diffY; j++){
  if ((rowOfY == ((y2 >= y1) ? (y1 + j) : (y1 - j))) && (columnOfY == ((x2 >= x1) ? (x1 + i) : (x1 - i)))){
  yuv[idx] = RGB2Y(color.x, color.y, color.z);
  yuv[W*H + rowOfUV *uvWidth + columnOfUV] = RGB2U(color.x, color.y, color.z);
  yuv[W*H + W*H / 4 + rowOfUV *uvWidth + columnOfUV] = RGB2V(color.x, color.y, color.z);
  }
  }
  }
  }
  __global__ void SetColor(uint8_t *yuv, Vector3D color)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < W*H) {
  int rowOfY = (idx / W);
  int columnOfY = (idx%W);
  int rowOfUV = rowOfY/2;
  int columnOfUV = columnOfY / 2;
  int uvWidth=W/2;
  yuv[idx] = RGB2Y(0, 255, 0);
  yuv[W*H + rowOfUV *uvWidth + columnOfUV] = RGB2U(0, 255, 0);
  yuv[W*H + W*H / 4 + rowOfUV *uvWidth + columnOfUV] = RGB2V(0, 255, 0);
  //yuv[W*H + idx/4] = RGB2U(color.x, color.y, color.z);
  //yuv[W*H + W*H / 4 + idx/4] = RGB2V(color.x, color.y, color.z);
  }
  }
  __device__ void koch(uint8_t *yuv, int x1, int y1, int x2, int y2, int it, Vector3D colorRGB)
  {
  float angle = 60 * M_PI / 180;
  int x3 = (2 * x1 + x2) / 3;
  int y3 = (2 * y1 + y2) / 3;

  int x4 = (x1 + 2 * x2) / 3;
  int y4 = (y1 + 2 * y2) / 3;

  int x = x3 + (x4 - x3)*cos(angle) + (y4 - y3)*sin(angle);
  int y = y3 - (x4 - x3)*sin(angle) + (y4 - y3)*cos(angle);

  if (it > 0)
  {
  koch(yuv, x1, y1, x3, y3, it - 1, colorRGB);
  koch(yuv, x3, y3, x, y, it - 1, colorRGB);
  koch(yuv, x, y, x4, y4, it - 1, colorRGB);
  koch(yuv, x4, y4, x2, y2, it - 1, colorRGB);
  }
  else{
  Line(yuv, x1, y1, x3, y3, colorRGB);
  Line(yuv, x3, y3, x, y, colorRGB);
  Line(yuv, x, y, x4, y4, colorRGB);
  Line(yuv, x4, y4, x2, y2, colorRGB);
  }
  }

  __global__ void DrawLine(uint8_t *yuv, int x1, int y1, int x2, int y2, Vector3D color)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < W*H){
  int rowOfY = (idx / W);
  int columnOfY = (idx%W);
  int rowOfUV = rowOfY / 2;
  int columnOfUV = columnOfY / 2;
  int uvWidth = W / 2;
  int diffX=abs(x1 - x2);
  int diffY = abs(y1 - y2);
  for (int i = 0; i<diffX; i++){
  for (int j = 0; j<diffY; j++){
  if ((rowOfY == ((y2>=y1) ? (y1 + j) : (y1 - j))) && (columnOfY ==(( x2>=x1) ? (x1 + i) : (x1 - i)))){
  yuv[idx] = RGB2Y(0, 255, 0);
  yuv[W*H + rowOfUV *uvWidth + columnOfUV] = RGB2U(0, 255, 0);
  yuv[W*H + W*H / 4 + rowOfUV *uvWidth + columnOfUV] = RGB2V(0, 255, 0);
  }
  }
  }

  //yuv[W*H + idx/4] = RGB2U(color.x, color.y, color.z);
  //yuv[W*H + W*H / 4 + idx/4] = RGB2V(color.x, color.y, color.z);
  }
  }
  __global__ void SetKoch(uint8_t *yuv, int t, Vector3D color)
  {
  int CenterX = W / 2;
  int CenterY = H / 2;
  int offset=20+t*2;
  int cloneNum=5;
  int iterNum = 6* t / NFRAME;
  int angle = Two_M_PI*t / NFRAME;
  for (int i = 0; i < cloneNum; i++){
  int newXPos = (CenterX + offset) * cosf(i * Two_M_PI / cloneNum) - (CenterY + offset) * sinf(i * Two_M_PI / cloneNum);
  int newYPos = (CenterX + +offset) * sinf(i *Two_M_PI / cloneNum) + (CenterY + offset) * cosf(i * Two_M_PI / cloneNum);
  koch(yuv, CenterX, CenterY, newXPos* cosf(angle) - newYPos*sinf(angle), newXPos*sinf(angle) + newYPos*cosf(angle), iterNum, color);
  }
  }

  void  Lab2VideoGenerator::CreateQuad(uint8_t *yuv, unsigned char size, int pos, Vector3D color)
  {
  //Y
  for (int i = 0; i< size; i++)
  {
  for (int j = 0; j < size; j++)
  {
  int pos_ = pos + (i * W + j)>W*H ? W*H : pos + (i * W + j);
  cudaMemset(yuv + pos_, RGB2Y(color.x, color.y, color.z), 1);
  }
  }
  //U
  for (int i = 0; i< size / 2; i++)
  {
  for (int j = 0; j < size / 2; j++)
  {
  int pos_ = pos / 4 + (i * W / 2 + j)>W*H/4 ? W*H/4 : pos / 4 + (i * W / 2 + j);
  cudaMemset(yuv + W*H + pos_, RGB2U(color.x, color.y, color.z), 1);
  }
  }
  //V
  for (int i = 0; i< size / 2; i++)
  {
  for (int j = 0; j < size / 2; j++)
  {
  int pos_ = pos / 4 + (i * W / 2 + j)>W*H / 4 ? W*H / 4 : pos / 4 + (i * W / 2 + j);
  cudaMemset(yuv + W*H + W*H / 4 + pos_, RGB2V(color.x, color.y, color.z), 1);
  }
  }
  }
  void Lab2VideoGenerator::Generate(uint8_t *yuv) {//³Ð«Ø­Óframe
  //cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
  //int posID = ((impl->t))*(W*H / (NFRAME));
  Vector3D background_ColorRGB(0, 0, 0);

  //background
  //Y
  cudaMemset(yuv, RGB2Y(background_ColorRGB.x, background_ColorRGB.y, background_ColorRGB.z), W*H);
  //U
  cudaMemset(yuv + W*H, RGB2U(background_ColorRGB.x, background_ColorRGB.y, background_ColorRGB.z), W*H / 4);
  //V
  cudaMemset(yuv + W*H + W*H / 4, RGB2V(background_ColorRGB.x, background_ColorRGB.y, background_ColorRGB.z), W*H / 4);

  //CreateQuad(yuv, 64, posID, colorRGB);
  int block_dim = W*H / 10 + 1;
  //SetColor << <block_dim, 10 >> > (yuv,colorRGB);
  Vector3D colorRGB(0, 255, 0);
  SetKoch << <1, 1 >> > (yuv, (impl->t), colorRGB);
  ++(impl->t);
  }*/
void  Lab2VideoGenerator::CreateQuad(uint8_t *yuv, unsigned char size, int pos, Vector3D color)
{
	//Y
	for (int i = 0; i< size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int pos_ = pos + (i * W + j)>W*H ? W*H : pos + (i * W + j);
			cudaMemset(yuv + pos_, RGB2Y(color.x, color.y, color.z), 1);
		}
	}
	//U
	for (int i = 0; i< size / 2; i++)
	{
		for (int j = 0; j < size / 2; j++)
		{
			int pos_ = pos / 4 + (i * W / 2 + j)>W*H / 4 ? W*H / 4 : pos / 4 + (i * W / 2 + j);
			cudaMemset(yuv + W*H + pos_, RGB2U(color.x, color.y, color.z), 1);
		}
	}
	//V
	for (int i = 0; i< size / 2; i++)
	{
		for (int j = 0; j < size / 2; j++)
		{
			int pos_ = pos / 4 + (i * W / 2 + j)>W*H / 4 ? W*H / 4 : pos / 4 + (i * W / 2 + j);
			cudaMemset(yuv + W*H + W*H / 4 + pos_, RGB2V(color.x, color.y, color.z), 1);
		}
	}
}
__global__ void SetColor(uint8_t *yuv, Vector3D color)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < W*H) {
		int rowOfY = (idx / W);
		int columnOfY = (idx%W);
		int rowOfUV = rowOfY / 2;
		int columnOfUV = columnOfY / 2;
		int uvWidth = W / 2;
		yuv[idx] = RGB2Y(0, 255, 0);
		yuv[W*H + rowOfUV *uvWidth + columnOfUV] = RGB2U(0, 255, 0);
		yuv[W*H + W*H / 4 + rowOfUV *uvWidth + columnOfUV] = RGB2V(0, 255, 0);
		//yuv[W*H + idx/4] = RGB2U(color.x, color.y, color.z);
		//yuv[W*H + W*H / 4 + idx/4] = RGB2V(color.x, color.y, color.z);
	}
}
__device__ void Line(uint8_t *yuv, int x1, int y1, int x2, int y2, Vector3D color)
{
	unsigned char colorZ[3] = { 0, 0, 255 };
	int windowDis = W;
	int dis = x1;
	float disMod = (float)dis / (float)windowDis;

	int idx = y1*W + x1;
	if (idx>W*H || x1<0 || x2<0 || x1>W || x2>W || y1<0 || y2<0 || y1>H || y2>H)return;
	//printf("%f \n", disMod);
	int rowOfY = (idx / W);
	int columnOfY = (idx%W);
	int rowOfUV = rowOfY / 2;
	int columnOfUV = columnOfY / 2;
	int uvWidth = W / 2;
	int uvIdx = rowOfUV *uvWidth + columnOfUV; /*
	yuv[idx] = RGB2Y(color.x*disMod + colorZ[0] * (1 - disMod), 
	                 color.y*disMod + colorZ[1] * (1 - disMod), 
	                  color.z*disMod+ colorZ[2] * (1 - disMod));
	//printf("x=%f,y=%f,z=%f\n", color.x*disMod + colorZ[0] * (1 - disMod), color.y*disMod + colorZ[1] * (1 - disMod), color.z*disMod + colorZ[2] * (1 - disMod));
	yuv[W*H + uvIdx] = RGB2U(color.x*disMod + colorZ[0] * (1 - disMod), 
							 color.y*disMod + colorZ[1] * (1 - disMod), 
							 color.z*disMod + colorZ[2] * (1 - disMod));
	yuv[W*H + W*H / 4 + uvIdx] = RGB2V(color.x*disMod + colorZ[0] * (1 - disMod), 
	                                    color.y*disMod + colorZ[1] * (1 - disMod),
	                                    color.z*disMod + colorZ[2] * (1 - disMod));*/
	yuv[idx] = RGB2Y(color.x, abs(color.y - disMod * color.y), abs(color.z - disMod * color.z));
	yuv[W*H + uvIdx] = RGB2U(color.x, abs(color.y - disMod *color.y), abs(color.z - disMod * color.z));
	yuv[W*H + W*H / 4 + uvIdx] = RGB2V(color.x, abs(color.y - disMod * color.y), abs(color.z - disMod * 95));
}

__device__ void koch(uint8_t *yuv, int x1, int y1, int x2, int y2, int it, Vector3D colorRGB)
{
	float angle = 60 * M_PI / 180;
	int x3 = (2 * x1 + x2) / 3;
	int y3 = (2 * y1 + y2) / 3;

	int x4 = (x1 + 2 * x2) / 3;
	int y4 = (y1 + 2 * y2) / 3;

	int x = x3 + (x4 - x3)*cos(angle) + (y4 - y3)*sin(angle);
	int y = y3 - (x4 - x3)*sin(angle) + (y4 - y3)*cos(angle);

	if (it > 0)
	{
		koch(yuv, x1, y1, x3, y3, it - 1, colorRGB);
		koch(yuv, x3, y3, x, y, it - 1, colorRGB);
		koch(yuv, x, y, x4, y4, it - 1, colorRGB);
		koch(yuv, x4, y4, x2, y2, it - 1, colorRGB);
	}
	else{
		Line(yuv, x1, y1, x3, y3, colorRGB);
		Line(yuv, x3, y3, x, y, colorRGB);
		Line(yuv, x, y, x4, y4, colorRGB);
		Line(yuv, x4, y4, x2, y2, colorRGB);
	}
}
__global__ void SetKoch(uint8_t *yuv, int x1, int y1, int  offset, int it, int cloneNum, float angle, Vector3D colorRGB)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < cloneNum)
	{
		//printf("it=%d ,ff=%d \n", it, offset);
		idx += 1;
		float param = (360.0 / cloneNum)*idx;
		int newXPos = (float)(x1 + offset) * cosf(param*Two_M_PI / 360.0) - (float)(y1)* sinf(param*Two_M_PI / 360.0);
		int newYPos = (float)(x1 + offset) * sinf(param*Two_M_PI / 360.0) + (float)(y1)* cosf(param*Two_M_PI / 360.0);
		koch(yuv, x1, y1, newXPos*cosf(angle) - newYPos*sinf(angle), newYPos*cosf(angle) + newXPos*sinf(angle), it, colorRGB);
	}
}
void Lab2VideoGenerator::Generate(uint8_t *yuv) {//³Ð«Ø­Óframe
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	int posID = ((impl->t))*(W*H / (NFRAME));
	Vector3D background_ColorRGB(255,255, 250);

	//background
	//Y
	cudaMemset(yuv, RGB2Y(background_ColorRGB.x, background_ColorRGB.y, background_ColorRGB.z), W*H);
	//U
	cudaMemset(yuv + W*H, RGB2U(background_ColorRGB.x, background_ColorRGB.y, background_ColorRGB.z), W*H / 4);
	//V
	cudaMemset(yuv + W*H + W*H / 4, RGB2V(background_ColorRGB.x, background_ColorRGB.y, background_ColorRGB.z), W*H / 4);


	int block_dim = W*H / 10 + 1;
	//SetColor << <block_dim, 10 >> > (yuv,colorRGB);

	Vector3D colorRGBX(255, 0, 0);
	Vector3D colorRGBY(0, 255,0);
	Vector3D colorRGBZ(0, 0, 255);
	int CenterX = W / 2;
	int CenterY = H / 2;

	int cloneNum = 10;
	int offset = 20 + impl->t * 2;
	int iterNum = 5 * impl->t / NFRAME;
	float angle = (float)Two_M_PI*(impl->t + 1) /(NFRAME);
	//CreateQuad(yuv, 8, posID, colorRGBY);
	SetKoch << <1, cloneNum >> > (yuv, CenterX, CenterY, offset, iterNum, cloneNum, angle, colorRGBX);
	SetKoch << <1, cloneNum >> > (yuv, CenterX, CenterY, offset, iterNum, cloneNum, -angle, colorRGBY);
	//SetKochZ << <1,1 >> > (yuv, CenterX, CenterY, 100, iterNum, colorRGBZ);
	(impl->t)++;
}
