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


#ifndef MATRIX_MATH_CUH
#define MATRIX_MATH_CUH

#include <cuda_runtime.h>
#include <supersurfel_fusion/matrix_types.h>
#include <math.h>
#include <supersurfel_fusion/vector_math.cuh>


inline __host__ __device__ Cov3 make_cov3(float xx, float xy, float xz, float yy, float yz, float zz)
{
  Cov3 res;
  res.xx = xx;
  res.xy = xy;
  res.xz = xz;
  res.yy = yy;
  res.yz = yz;
  res.zz = zz;
  return res;
}

inline __host__ __device__ bool inverse(const Cov3 &in, Cov3 &out)
{
  out.xx = in.zz*in.yy - in.yz*in.yz;
  out.xy = in.xz*in.yz - in.zz*in.xy;
  out.xz = in.xy*in.yz - in.xz*in.yy;
  out.yy = in.zz*in.xx - in.xz*in.xz;
  out.yz = in.xy*in.xz - in.xx*in.yz;
  out.zz = in.xx*in.yy - in.xy*in.xy;

  float det = in.xx*out.xx+in.xy*out.xy+in.xz*out.xz;
  if(fabs(det) > 1e-9)
  {
	out.xx /= det;
	out.xy /= det;
	out.xz /= det;
	out.yy /= det;
	out.yz /= det;
	out.zz /= det;
	return true;
  }else{
	return false;
  }
}

inline __host__ __device__ Cov3 operator+( const Cov3& a, const Cov3& b)
{
  Cov3 res;
  res.xx = a.xx+b.xx;
  res.xy = a.xy+b.xy;
  res.xz = a.xz+b.xz;
  res.yy = a.yy+b.yy;
  res.yz = a.yz+b.yz;
  res.zz = a.zz+b.zz;
  return res;
}

inline __host__ __device__ Cov3 operator-( const Cov3& a, const Cov3& b)
{
  Cov3 res;
  res.xx = a.xx-b.xx;
  res.xy = a.xy-b.xy;
  res.xz = a.xz-b.xz;
  res.yy = a.yy-b.yy;
  res.yz = a.yz-b.yz;
  res.zz = a.zz-b.zz;
  return res;
}

inline __host__ __device__ void operator+=( Cov3& a, const Cov3& b)
{
  a.xx+=b.xx;
  a.xy+=b.xy;
  a.xz+=b.xz;
  a.yy+=b.yy;
  a.yz+=b.yz;
  a.zz+=b.zz;
}

inline __host__ __device__ void operator-=( Cov3& a, const Cov3& b)
{
  a.xx-=b.xx;
  a.xy-=b.xy;
  a.xz-=b.xz;
  a.yy-=b.yy;
  a.yz-=b.yz;
  a.zz-=b.zz;
}

inline __host__ __device__ Cov3 operator*( const Cov3& a, float b)
{
  Cov3 res;
  res.xx = a.xx*b;
  res.xy = a.xy*b;
  res.xz = a.xz*b;
  res.yy = a.yy*b;
  res.yz = a.yz*b;
  res.zz = a.zz*b;
  return res;
}
inline __host__ __device__ Cov3 operator*( float b, const Cov3& a)
{
  Cov3 res;
  res.xx = b*a.xx;
  res.xy = b*a.xy;
  res.xz = b*a.xz;
  res.yy = b*a.yy;
  res.yz = b*a.yz;
  res.zz = b*a.zz;
  return res;
}

inline __host__ __device__ Cov3 operator/( const Cov3& a, float b)
{
  Cov3 res;
  res.xx = a.xx/b;
  res.xy = a.xy/b;
  res.xz = a.xz/b;
  res.yy = a.yy/b;
  res.yz = a.yz/b;
  res.zz = a.zz/b;
  return res;
}

inline __host__ __device__ void operator*=( Cov3& a, float b)
{
  a.xx*=b;
  a.xy*=b;
  a.xz*=b;
  a.yy*=b;
  a.yz*=b;
  a.zz*=b;
}
inline __host__ __device__ void operator/=( Cov3& a, float b)
{
  a.xx/=b;
  a.xy/=b;
  a.xz/=b;
  a.yy/=b;
  a.yz/=b;
  a.zz/=b;
}


inline __host__ __device__ float3 operator*( const Cov3& m, const float3& b)
{
  return make_float3(m.xx * b.x + m.xy * b.y + m.xz*b.z,
					 m.xy * b.x + m.yy * b.y + m.yz*b.z,
					 m.xz * b.x + m.yz * b.y + m.zz*b.z);
}

// inline __host__ __device__ Cov3 operator*( const Cov3& a, const Cov3& b)
// {
  
//   Cov3 res;
//   res.xx = a.xx*b.xx + a.xy*b.xy + a.xz*b.xz;
//   res.xy = a.xx*b.xy + a.xy*b.yy + a.xz*b.yz;
//   res.xz = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;
//   res.yy = a.xy*b.xy + a.yy*b.yy + a.yz*b.yz;
//   res.yz = a.xy*b.xz + a.yy*b.yz + a.yz*b.zz;
//   res.zz = a.xz*b.xz + a.yz*b.yz + a.zz*b.zz;
//   return res;
// }

inline __host__ __device__ Cov3 square( const Cov3& a)
{
  Cov3 res;
  res.xx = a.xx*a.xx + a.xy*a.xy + a.xz*a.xz;
  res.xy = a.xx*a.xy + a.xy*a.yy + a.xz*a.yz;
  res.xz = a.xx*a.xz + a.xy*a.yz + a.xz*a.zz;
  res.yy = a.xy*a.xy + a.yy*a.yy + a.yz*a.yz;
  res.yz = a.xy*a.xz + a.yy*a.yz + a.yz*a.zz;
  res.zz = a.xz*a.xz + a.yz*a.yz + a.zz*a.zz;
  return res;
}

inline __host__ __device__ Mat33 operator*( const Cov3& a, const Cov3& b)
{
  Mat33 res;
  res.rows[0].x = a.xx*b.xx + a.xy*b.xy + a.xz*b.xz;
  res.rows[0].y = a.xx*b.xy + a.xy*b.yy + a.xz*b.yz;
  res.rows[0].z = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;
  res.rows[1].x = a.xy*b.xx + a.yy*b.xy + a.yz*b.xz;
  res.rows[1].y = a.xy*b.xy + a.yy*b.yy + a.yz*b.yz;
  res.rows[1].z = a.xy*b.xz + a.yy*b.yz + a.yz*b.zz;
  res.rows[2].x = a.xz*b.xx + a.yz*b.xy + a.zz*b.xz;
  res.rows[2].y = a.xz*b.xy + a.yz*b.yy + a.zz*b.yz;
  res.rows[2].z = a.xz*b.xz + a.yz*b.yz + a.zz*b.zz;

  return res;
}

inline __host__ __device__ Cov3 outer_product( const float3& v)
{
  Cov3 res;
  res.xx = v.x*v.x;
  res.xy = v.x*v.y;
  res.xz = v.x*v.z;
  res.yy = v.y*v.y;
  res.yz = v.y*v.z;
  res.zz = v.z*v.z;
  return res;
}

inline __host__ __device__ Cov3 outer_product( const float3& u, const float3& v)
{
  Cov3 res;
  res.xx = u.x*v.x;
  res.xy = u.x*v.y;
  res.xz = u.x*v.z;
  res.yy = u.y*v.y;
  res.yz = u.y*v.z;
  res.zz = u.z*v.z;
  return res;
}

inline __host__ __device__ float trace( const Cov3& a)
{
  return a.xx+a.yy+a.zz;
}

inline __host__ __device__ Mat33 make_mat33(float3 e1, float3 e2, float3 e3)
{
  Mat33 res;
  res.rows[0] = e1;
  res.rows[1] = e2;
  res.rows[2] = e3;
  return res;
}

inline __host__ __device__ Mat33 make_mat33(float m11, float m12, float m13,
                                            float m21, float m22, float m23,
                                            float m31, float m32, float m33)
{
  return make_mat33(make_float3(m11, m12, m13),
                    make_float3(m21, m22, m23),
                    make_float3(m31, m32, m33));
}


inline __host__ __device__ Mat33 operator+( const Mat33& a, const Mat33& b)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]+b.rows[i];
  }*/
  res.rows[0] = a.rows[0]+b.rows[0];
  res.rows[1] = a.rows[1]+b.rows[1];
  res.rows[2] = a.rows[2]+b.rows[2];

  return res;
}

inline __host__ __device__ Mat33 operator-( Mat33& a)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = -a.rows[i];
  }*/
  res.rows[0] = -a.rows[0];
  res.rows[1] = -a.rows[1];
  res.rows[2] = -a.rows[2];
  return res;
}

inline __host__ __device__ Mat33 operator-( const Mat33& a, const Mat33& b)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]-b.rows[i];
  }*/
  res.rows[0] = a.rows[0]-b.rows[0];
  res.rows[1] = a.rows[1]-b.rows[1];
  res.rows[2] = a.rows[2]-b.rows[2];
  return res;
}

inline __host__ __device__ Mat33 operator*( const Mat33& a, const float b)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]*b;
  }*/
  res.rows[0] = a.rows[0]*b;
  res.rows[1] = a.rows[1]*b;
  res.rows[2] = a.rows[2]*b;
  return res;
}

inline __host__ __device__ Mat33 operator*( const float b, const Mat33& a)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = b*a.rows[i];
  }*/
  res.rows[0] = b*a.rows[0];
  res.rows[1] = b*a.rows[1];
  res.rows[2] = b*a.rows[2];
  return res;
}

inline __host__ __device__ Mat33 operator/( const Mat33& a, const float b)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = a.rows[i]/b;
  }*/
  res.rows[0] = a.rows[0]/b;
  res.rows[1] = a.rows[1]/b;
  res.rows[2] = a.rows[2]/b;
  return res;
}

inline __host__ __device__ void operator+=( Mat33& a, const Mat33& b)
{
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] += b.rows[i];
  }*/
  a.rows[0] += b.rows[0];
  a.rows[1] += b.rows[1];
  a.rows[2] += b.rows[2];
}

inline __host__ __device__ void operator-=( Mat33& a, const Mat33& b)
{
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] -= b.rows[i];
  }*/
  a.rows[0] -= b.rows[0];
  a.rows[1] -= b.rows[1];
  a.rows[2] -= b.rows[2];
}

inline __host__ __device__ void operator*=( Mat33& a, const float b)
{
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] *= b;
  }*/
  a.rows[0] *= b;
  a.rows[1] *= b;
  a.rows[2] *= b;
}

inline __host__ __device__ Mat33 operator*( const Mat33& a, const Mat33& b)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = make_float3(a.rows[i].x*b.rows[0].x+a.rows[i].y*b.rows[1].x+a.rows[i].z*b.rows[2].x,
                              a.rows[i].x*b.rows[0].y+a.rows[i].y*b.rows[1].y+a.rows[i].z*b.rows[2].y,
                              a.rows[i].x*b.rows[0].z+a.rows[i].y*b.rows[1].z+a.rows[i].z*b.rows[2].z);
  }*/
  res.rows[0] = make_float3(a.rows[0].x*b.rows[0].x+a.rows[0].y*b.rows[1].x+a.rows[0].z*b.rows[2].x,
                            a.rows[0].x*b.rows[0].y+a.rows[0].y*b.rows[1].y+a.rows[0].z*b.rows[2].y,
                            a.rows[0].x*b.rows[0].z+a.rows[0].y*b.rows[1].z+a.rows[0].z*b.rows[2].z);

  res.rows[1] = make_float3(a.rows[1].x*b.rows[0].x+a.rows[1].y*b.rows[1].x+a.rows[1].z*b.rows[2].x,
                            a.rows[1].x*b.rows[0].y+a.rows[1].y*b.rows[1].y+a.rows[1].z*b.rows[2].y,
                            a.rows[1].x*b.rows[0].z+a.rows[1].y*b.rows[1].z+a.rows[1].z*b.rows[2].z);

  res.rows[2] = make_float3(a.rows[2].x*b.rows[0].x+a.rows[2].y*b.rows[1].x+a.rows[2].z*b.rows[2].x,
                            a.rows[2].x*b.rows[0].y+a.rows[2].y*b.rows[1].y+a.rows[2].z*b.rows[2].y,
                            a.rows[2].x*b.rows[0].z+a.rows[2].y*b.rows[1].z+a.rows[2].z*b.rows[2].z);
  return res;
}

inline __host__ __device__ Mat33 operator*( const Mat33& a, const Cov3& b)
{
  Mat33 res;
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    res.rows[i] = make_float3(a.rows[i].x*b.xx+a.rows[i].y*b.xy+a.rows[i].z*b.xz,
                              a.rows[i].x*b.xy+a.rows[i].y*b.yy+a.rows[i].z*b.yz,
                              a.rows[i].x*b.xz+a.rows[i].y*b.yz+a.rows[i].z*b.zz);
  }*/
  res.rows[0] = make_float3(a.rows[0].x*b.xx+a.rows[0].y*b.xy+a.rows[0].z*b.xz,
                            a.rows[0].x*b.xy+a.rows[0].y*b.yy+a.rows[0].z*b.yz,
                            a.rows[0].x*b.xz+a.rows[0].y*b.yz+a.rows[0].z*b.zz);
  res.rows[1] = make_float3(a.rows[1].x*b.xx+a.rows[1].y*b.xy+a.rows[1].z*b.xz,
                            a.rows[1].x*b.xy+a.rows[1].y*b.yy+a.rows[1].z*b.yz,
                            a.rows[1].x*b.xz+a.rows[1].y*b.yz+a.rows[1].z*b.zz);
  res.rows[2] = make_float3(a.rows[2].x*b.xx+a.rows[2].y*b.xy+a.rows[2].z*b.xz,
                            a.rows[2].x*b.xy+a.rows[2].y*b.yy+a.rows[2].z*b.yz,
                            a.rows[2].x*b.xz+a.rows[2].y*b.yz+a.rows[2].z*b.zz);
  return res;
}

inline __host__ __device__ Mat33 operator*( const Cov3& a, const Mat33& b)
{
  Mat33 res;
  res.rows[0] = make_float3(a.xx*b.rows[0].x+a.xy*b.rows[1].x+a.xz*b.rows[2].x,
                            a.xx*b.rows[0].y+a.xy*b.rows[1].y+a.xz*b.rows[2].y,
                            a.xx*b.rows[0].z+a.xy*b.rows[1].z+a.xz*b.rows[2].z);
  res.rows[1] = make_float3(a.xy*b.rows[0].x+a.yy*b.rows[1].x+a.yz*b.rows[2].x,
                            a.xy*b.rows[0].y+a.yy*b.rows[1].y+a.yz*b.rows[2].y,
                            a.xy*b.rows[0].z+a.yy*b.rows[1].z+a.yz*b.rows[2].z);
  res.rows[2] = make_float3(a.xz*b.rows[0].x+a.yz*b.rows[1].x+a.zz*b.rows[2].x,
                            a.xz*b.rows[0].y+a.yz*b.rows[1].y+a.zz*b.rows[2].y,
                            a.xz*b.rows[0].z+a.yz*b.rows[1].z+a.zz*b.rows[2].z);
  return res;
}

inline __host__ __device__ Cov3 mult_ABAt( const Mat33& A, const Cov3& B)
{
  float3 r1 = make_float3(B.xx, B.xy, B.xz);
  float3 r2 = make_float3(B.xy, B.yy, B.yz);
  float3 r3 = make_float3(B.xz, B.yz, B.zz);

  Mat33 BAtt = make_mat33(dot(r1, A.rows[0]), dot(r2, A.rows[0]), dot(r3, A.rows[0]),
                          dot(r1, A.rows[1]), dot(r2, A.rows[1]), dot(r3, A.rows[1]),
                          dot(r1, A.rows[2]), dot(r2, A.rows[2]), dot(r3, A.rows[2]));
  
  Cov3 res = make_cov3(dot(A.rows[0], BAtt.rows[0]),
                       dot(A.rows[0], BAtt.rows[1]),
                       dot(A.rows[0], BAtt.rows[2]),
                       dot(A.rows[1], BAtt.rows[1]),
                       dot(A.rows[1], BAtt.rows[2]),
                       dot(A.rows[2], BAtt.rows[2]));
  return res;
}


inline __host__ __device__ void operator*=( Mat33& a, const Mat33& b)
{
  /*#pragma unroll
  for(int i=0; i<3; i++)
  {
    a.rows[i] = make_float3(a.rows[i].x*b.rows[0].x+a.rows[i].y*b.rows[1].x+a.rows[i].z*b.rows[2].x,
                            a.rows[i].x*b.rows[0].y+a.rows[i].y*b.rows[1].y+a.rows[i].z*b.rows[2].y,
                            a.rows[i].x*b.rows[0].z+a.rows[i].y*b.rows[1].z+a.rows[i].z*b.rows[2].z);
  }*/
  a.rows[0] = make_float3(a.rows[0].x*b.rows[0].x+a.rows[0].y*b.rows[1].x+a.rows[0].z*b.rows[2].x,
                          a.rows[0].x*b.rows[0].y+a.rows[0].y*b.rows[1].y+a.rows[0].z*b.rows[2].y,
                          a.rows[0].x*b.rows[0].z+a.rows[0].y*b.rows[1].z+a.rows[0].z*b.rows[2].z);

  a.rows[1] = make_float3(a.rows[1].x*b.rows[0].x+a.rows[1].y*b.rows[1].x+a.rows[1].z*b.rows[2].x,
                          a.rows[1].x*b.rows[0].y+a.rows[1].y*b.rows[1].y+a.rows[1].z*b.rows[2].y,
                          a.rows[1].x*b.rows[0].z+a.rows[1].y*b.rows[1].z+a.rows[1].z*b.rows[2].z);

  a.rows[2] = make_float3(a.rows[2].x*b.rows[0].x+a.rows[2].y*b.rows[1].x+a.rows[2].z*b.rows[2].x,
                          a.rows[2].x*b.rows[0].y+a.rows[2].y*b.rows[1].y+a.rows[2].z*b.rows[2].y,
                          a.rows[2].x*b.rows[0].z+a.rows[2].y*b.rows[1].z+a.rows[2].z*b.rows[2].z);
}

inline __host__ __device__ float3 operator*( const Mat33& a, const float3& b)
{
  return make_float3(dot(a.rows[0], b),
                     dot(a.rows[1], b),
                     dot(a.rows[2], b));
}

inline __host__ __device__ float3 operator*( const float3& a, const Mat33& b)
{
  return make_float3(a.x*b.rows[0].x+a.y*b.rows[1].x+a.z*b.rows[2].x,
                     a.x*b.rows[0].y+a.y*b.rows[1].y+a.z*b.rows[2].y,
                     a.x*b.rows[0].z+a.y*b.rows[1].z+a.z*b.rows[2].z);
}

inline __host__ __device__ Mat33 transpose( const Mat33& a )
{
  Mat33 res;
  res.rows[0].x = a.rows[0].x; res.rows[0].y = a.rows[1].x; res.rows[0].z = a.rows[2].x;
  res.rows[1].x = a.rows[0].y; res.rows[1].y = a.rows[1].y; res.rows[1].z = a.rows[2].y;
  res.rows[2].x = a.rows[0].z; res.rows[2].y = a.rows[1].z; res.rows[2].z = a.rows[2].z;
  return res;
}

inline __host__ __device__ float trace( const Mat33& a )
{
  return a.rows[0].x+a.rows[1].y+a.rows[2].z;
}

inline __host__ __device__ Mat33 quatToRotMat(const float4& q)
{
    float const x2 = q.x*q.x;
    float const y2 = q.y*q.y;
    float const z2 = q.z*q.z;
    float const xy = q.x*q.y;
    float const xz = q.x*q.z;
    float const yz = q.y*q.z;
    float const wx = q.w*q.x;
    float const wy = q.w*q.z;
    float const wz = q.w*q.z;

    return make_mat33(1.0f - 2.0f*(y2 + z2), 2.0f*(xy - wz), 2.0f*(xz + wy),
                      2.0f*(xy + wz), 1.0f - 2.0f*(x2 + z2), 2.0f*(yz - wx),
                      2.0f*(xz - wy), 2.0f*(yz + wx), 1.0f - 2.0f*(x2 + y2));
}

inline __host__ __device__ float4 rotMatToQuat(const Mat33& m)
{
    float4 q;
    float s;
    float trace = m.rows[0].x + m.rows[1].y + m.rows[2].z;

    if(trace>0)
    {
        //float s = 0.5f*rsqrtf(trace + 1.0f);
        s = sqrtf(trace + 1);
        q.w = 0.5f * s;
        s = 0.5f / s;
        //q.w = 0.25f/s;
        q.x = (m.rows[2].y - m.rows[1].z)*s;
        q.y = (m.rows[0].z - m.rows[2].x)*s;
        q.z = (m.rows[1].x - m.rows[0].y)*s;
    }
    else
    {
        int i = 0;
        if(m.rows[1].y > m.rows[0].x)
            i = 1;

        if(m.rows[2].z > m.rows[0].x || m.rows[2].z > m.rows[1].y)
            i = 2;

        switch(i)
        {
        case 0:
            s = sqrtf(1.0f + m.rows[0].x - m.rows[1].y - m.rows[2].z);
            q.x = 0.5f * s;
            s = 0.5f / s;
            q.w = (m.rows[2].y - m.rows[1].z) * s;
            q.y = (m.rows[0].y + m.rows[1].x) * s;
            q.z = (m.rows[0].z + m.rows[2].x) * s;
            break;

        case 1:
             s = sqrtf(1.0f + m.rows[1].y - m.rows[0].x - m.rows[2].z);
             q.y = 0.5f * s;
             s = 0.5f / s;
             q.w = (m.rows[0].z - m.rows[2].x) * s;
             q.x = (m.rows[0].y + m.rows[1].x) * s;
             q.z = (m.rows[1].z + m.rows[2].y) * s;
            break;

        case 2:
            s = sqrtf(1.0f + m.rows[2].z - m.rows[0].x - m.rows[1].y);
            q.z = 0.5f * s;
            s = 0.5f / s;
            q.w = (m.rows[1].x - m.rows[0].y) * s;
            q.x = (m.rows[0].z + m.rows[2].x) * s;
            q.y = (m.rows[1].z + m.rows[2].y) * s;
            break;
        }

//        if(m.rows[0].x>m.rows[1].y && m.rows[0].x>m.rows[2].z)
//        {
//            //float s = 2.0f*sqrtf(1.0f + m.rows[0].x - m.rows[1].y - m.rows[2].z);
//            float rs = 0.5f*rsqrtf(1.0f + m.rows[0].x - m.rows[1].y - m.rows[2].z);

//            q.w = (m.rows[2].y - m.rows[1].z)*rs;
//            q.x = 0.25f/rs;
//            q.y = (m.rows[0].y + m.rows[1].x)*rs;
//            q.z = (m.rows[0].z + m.rows[2].x)*rs;
//        }
//        else if(m.rows[1].y>m.rows[2].z)
//        {
//            //float s = 2.0f*sqrtf(1.0f + m.rows[1].y - m.rows[0].x - m.rows[2].z);
//            float rs = 0.5f*rsqrtf(1.0f + m.rows[1].y - m.rows[0].x - m.rows[2].z);

//            q.w = (m.rows[0].z - m.rows[2].x)*rs;
//            q.x = (m.rows[0].y + m.rows[1].x)*rs;
//            q.y = 0.25f/rs;
//            q.z = (m.rows[1].z + m.rows[2].y)*rs;
//        }
//        else
//        {
//            //float s = 2.0f*sqrtf(1.0f + m.rows[2].z - m.rows[0].x - m.rows[1].y);
//            float rs = 0.5f*rsqrtf(1.0f + m.rows[2].z - m.rows[0].x - m.rows[1].y);

//            q.w = (m.rows[1].x - m.rows[0].y)*rs;
//            q.x = (m.rows[0].z + m.rows[2].x)*rs;
//            q.y = (m.rows[1].z + m.rows[2].y)*rs;
//            q.z = 0.25f/rs;
//        }
    }

    return q;
}

//inline __host__ __device__ float4 rotMatToQuat(const Mat33& m)
//{
//    float4 q;
//    float t;

//    if(m.rows[2].z < 0)
//    {
//        if(m.rows[0].x > m.rows[1].y)
//        {
//            t = 1.0f + m.rows[0].x - m.rows[1].y - m.rows[2].z;
//            q = make_float4(t,
//                            m.rows[0].y + m.rows[1].x,
//                            m.rows[2].x + m.rows[0].z,
//                            m.rows[1].z - m.rows[2].y);
//        }
//        else
//        {
//            t = 1.0f - m.rows[0].x + m.rows[1].y - m.rows[2].z;
//            q = make_float4(m.rows[0].y + m.rows[1].x,
//                            t,
//                            m.rows[1].z + m.rows[2].y,
//                            m.rows[2].x - m.rows[0].z);
//        }
//    }
//    else
//    {
//        if(m.rows[0].x < -m.rows[1].y)
//        {
//            t = 1.0f - m.rows[0].x - m.rows[1].y + m.rows[2].z;
//            q = make_float4(m.rows[2].x + m.rows[0].z,
//                            m.rows[1].z + m.rows[2].y,
//                            t,
//                            m.rows[0].y - m.rows[1].x);
//        }
//        else
//        {
//            t = 1.0f + m.rows[0].x + m.rows[1].y + m.rows[2].z;
//            q = make_float4(m.rows[1].z - m.rows[2].y,
//                            m.rows[2].x - m.rows[0].z,
//                            m.rows[0].y - m.rows[1].x,
//                            t);
//        }
//    }

//    return 0.5f*q/sqrtf(t);
//}

// Simple, computationally cheap and stable procedure to re-orthogonalize the rotation matrix.
// This orthogonalization procedure is taken from "Direction Cosine Matrix IMU: Theory" by William Premerlani and Paul Bizard; equations 19-21.
// It should be noted that the normalization procedure here uses a Taylor expansion to approximate vector magnitude.
// If high precision is needed or the matrix is far from orthonormal then another method should be used
// (such as QR decomposition, or converting matrix to axis angle representation and axis angle back to rot matrix).
// See https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix
inline __host__ __device__ Mat33 orthogonalize(const Mat33& m)
{
    Mat33 m_T = transpose(m);

    float3 x_ort = normalize(m_T.rows[0]);
    float3 y_ort = normalize(m_T.rows[1]);
    float3 z_ort = normalize(m_T.rows[2]);

    float3 x_new = 0.5f * (3.0f - dot(x_ort, x_ort)) * x_ort;
    float3 y_new = 0.5f * (3.0f - dot(y_ort, y_ort)) * y_ort;
    float3 z_new = 0.5f * (3.0f - dot(z_ort, z_ort)) * z_ort;

    return make_mat33(x_new.x, x_new.y, z_new.z,
                      y_new.x, y_new.y, y_new.z,
                      z_new.x, z_new.y, z_new.z);
}

#endif // MATRIX_MATH_CUH
