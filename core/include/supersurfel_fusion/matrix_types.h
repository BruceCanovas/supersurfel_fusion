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


#ifndef MATRIX_TYPES_HPP
#define MATRIX_TYPES_HPP

#include <cuda_runtime.h>


struct Cov3
{
  float xx, xy, xz,
    yy, yz,
    zz;
};

struct Mat33
{
  float3 rows[3];
};

struct Transform3
{
    Mat33 R;
    float3 t;
};

#endif // MATRIX_TYPES_HPP
