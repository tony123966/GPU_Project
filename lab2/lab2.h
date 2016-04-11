#pragma once
#include <cstdint>
#include <memory>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using std::unique_ptr;
struct Vector3D
{
	unsigned char x, y, z;
	Vector3D(unsigned char x, unsigned char y, unsigned char z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}
};
struct Lab2VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

class Lab2VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
public:
	Lab2VideoGenerator();
	~Lab2VideoGenerator();
	void get_info(Lab2VideoInfo &info);
	void Generate(uint8_t *yuv);
	void CreateQuad(uint8_t *yuv, unsigned char size, int pos, Vector3D color);
	void koch(uint8_t *yuv, int x1, int y1, int x2, int y2, int it);
};
