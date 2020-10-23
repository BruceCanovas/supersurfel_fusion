/*************************************************************************
* Copyright (C) 2020, Bruce Canovas, Amaury Negre, GIPSA-lab
* This file is part of https://github.com/BruceCanovas/supersurfel_fusion

* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*************************************************************************/


#ifndef VECTOR_MATH_CUH
#define VECTOR_MATH_CUH

#include <cuda_runtime.h>
#include <math.h>


// additional constructors
inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}

// negate
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
  a.x += b.x;
  a.y += b.y;
}

// subtraction
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
  a.x -= b.x;
  a.y -= b.y;
}

// multiply
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(float2 a, float s)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(float s, float2 a)
{
    return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, float s)
{
    a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ float2 operator/(float2 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float2 operator/(float s, float2 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float2 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ float dot(float2 a, float2 b)
{ 
    return a.x * b.x + a.y * b.y;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

// length
inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float2 normalize(float2 v)
{
    //float invLen = 1.0f / sqrtf(dot(v, v));
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float2 floor(const float2 v)
{
    return make_float2(floor(v.x), floor(v.y));
}


// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
  return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(float3 &a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

// subtraction
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s;
	a.y *= s;
    a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float3 normalize(float3 v)
{
    //float invLen = 1.0f / sqrtf(dot(v, v));
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
  return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// min
inline __host__ __device__ float3 min(const float3 a, const float3 b)
{
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

// max
inline __host__ __device__ float3 max(const float3 a, const float3 b)
{
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

inline __host__ __device__ float get_coord(const float3 a, const int k)
{
    switch(k)
    {
    case 0:
        return a.x;
    case 1:
        return a.y;
    case 2:
        return a.z;
    default:
        return a.x;
    }
}

inline __host__ __device__ int get_coord(const int4 a, const int k)
{
    switch(k)
    {
    case 0:
        return a.x;
    case 1:
        return a.y;
    case 2:
        return a.z;
    case 3:
        return a.w;
    default:
        return a.x;
    }
}

inline __host__ __device__ float get_coord(const float4 a, const int k)
{
    switch(k)
    {
    case 0:
        return a.x;
    case 1:
        return a.y;
    case 2:
        return a.z;
    case 3:
        return a.w;
    default:
        return a.x;
    }
}

inline __host__ __device__ void set_coord(float3& a, const int k, const float val)
{
    switch(k)
    {
    case 0:
        a.x = val;
        break;
    case 1:
        a.y =  val;
        break;
    case 2:
        a.z =  val;
        break;
    }
}

inline __host__ __device__ void set_coord(int4& a, const int k, const int val)
{
    switch(k)
    {
    case 0:
        a.x = val;
        break;
    case 1:
        a.y =  val;
        break;
    case 2:
        a.z =  val;
        break;
    case 3:
        a.w = val;
    }
}

inline __host__ __device__ void set_coord(float4& a, const int k, const float val)
{
    switch(k)
    {
    case 0:
        a.x = val;
        break;
    case 1:
        a.y =  val;
        break;
    case 2:
        a.z =  val;
        break;
    case 3:
        a.w = val;
    }
}

// additional constructors
inline __host__ __device__ float4 make_float4(float s)
{
  return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), 0.f);
}

// negate
inline __host__ __device__ float4 operator-(float4 &a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// addition
inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// subtraction
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}

// multiply
inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ float4 operator*(float4 a, float s)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ float4 operator*(float s, float4 a)
{
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
inline __host__ __device__ void operator*=(float4 &a, float s)
{
    a.x *= s; a.y *= s;
    a.z *= s; a.w *= s;	
}

// divide
inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ float4 operator/(float4 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float4 operator/(float s, float4 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float4 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ float dot(float4 a, float4 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// length
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ float4 normalize(float4 v)
{
    //float invLen = 1.0f / sqrtf(dot(v, v));
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float4 floor(const float4 v)
{
  return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}

inline __host__ __device__ float4 quatSlerp(const float4& q0, const float4& q1_param, const float alpha)
{
    if(alpha<=0)
        return q0;
    if(alpha>=1)
        return q1_param;

    float cos_omega = dot(q0, q1_param);

    float4 q1 = q1_param;

    if(cos_omega<0.0f)
    {
        q1 = -q1;
        cos_omega = -cos_omega;
    }

    if(cos_omega>0.9999f)
        return (1.0f - alpha)*q0 + alpha*q1;

    const float sin_omega = sqrtf(1.0f - cos_omega*cos_omega);
    const float omega = atan2f(sin_omega, cos_omega);
    const float one_over_sin_omega = 1.0f/sin_omega;

    const float k0 = sinf((1.0f - alpha)*omega)*one_over_sin_omega;
    const float k1 = sinf(alpha*omega)*one_over_sin_omega;

    return k0*q0 + k1*q1;
}

inline __host__ __device__ float4 quatProduct(const float4& q1, const float4& q2)
{
    return make_float4(q1.x*q2.w + q1.w*q2.x + q1.y*q2.z - q1.z*q2.y,
                      q1.y*q2.w + q1.w*q2.y + q1.z*q2.x - q1.x*q2.z,
                      q1.z*q2.w + q1.w*q2.z + q1.x*q2.y - q1.y*q2.x,
                      q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z);
}

inline __host__ __device__ float4 conjugatedQuat(const float4& quat)
{
    return make_float4(-quat.x, -quat.y, -quat.z, quat.w);
}

inline __host__ __device__ float3 applyQuatToVec(const float4& quat, const float3& vec)
{
    const float4 z = make_float4(vec);
    float4 q = quat*z*conjugatedQuat(quat);

    return make_float3(q.x, q.y, q.z);
}

inline __host__ __device__ float3 labToRgb(float3 const& color)
{
    float y = (color.x+16.0f)/116.0f;
    float x = color.y/500.0f + y;
    float z = y - color.z/200.0f;

    x = 0.95047f*((powf(x, 3.0f)>0.008856f) ? powf(x, 3.0f) : (x-16.0f/116.0f)/7.787f);
    y = 1.0f*((powf(y, 3.0f)>0.008856f) ? powf(y, 3.0f) : (y-16.0f/116.0f)/7.787f);
    z = 1.08883f*((powf(z, 3.0f)>0.008856f) ? powf(z, 3.0f) : (z-16.0f/116.0f)/7.787f);

    float r = x*3.2406f - y*1.5372f - z*0.4986f;
    float g = -x*0.9689f + y*1.8758 + z*0.0415f;
    float b = x*0.0557f - y*0.2040f + z*1.0570;

    r = (r>0.0031308f) ? (1.055f * powf(r, 1.0f/2.4f)-0.055f) : 12.92f*r;
    g = (g>0.0031308f) ? (1.055f * powf(g, 1.0f/2.4f)-0.055f) : 12.92f*g;
    b = (b>0.0031308f) ? (1.055f * powf(b, 1.0f/2.4f)-0.055f) : 12.92f*b;

    return make_float3(fmaxf(0.0f, fminf(1.0f, r))*255.0f,
                       fmaxf(0.0f, fminf(1.0f, g))*255.0f,
                       fmaxf(0.0f, fminf(1.0f, b))*255.0f);
}

inline __host__ __device__ float3 rgbToLab(float3 const& color)
{
    float r = color.x/255.0f;
    float g = color.y/255.0f;
    float b = color.z/255.0f;

    r = (r>0.04045f) ? powf((r+0.055f)/1.055f, 2.4f) : r/12.92f;
    g = (g>0.04045f) ? powf((g+0.055f)/1.055f, 2.4f) : g/12.92f;
    b = (b>0.04045f) ? powf((b+0.055f)/1.055f, 2.4f) : b/12.92f;

    float x = (r*0.4124f + g*0.3575f + b*0.1805f)/0.95047f;
    float y = (r*0.2126f + g*0.7152f + b*0.0722f);
    float z = (r*0.0193f + g*0.1192f + b*0.9505f)/1.08883f;

    x = (x>0.008856f) ? cbrtf(x) : 7.787f*x + 16.0f/116.0f;
    y = (y>0.008856f) ? cbrtf(y) : 7.787f*y + 16.0f/116.0f;
    z = (z>0.008856f) ? cbrtf(z) : 7.787f*z + 16.0f/116.0f;

    return make_float3(116.0f*y-16.0f, 500.0f*(x-y), 200.0f*(y-z));
}

#endif // VECTOR_MATH_CUH
