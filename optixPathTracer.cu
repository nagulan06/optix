//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include <optix.h>

#include "optixPathTracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include <cuda_runtime.h>

//#include <stdio.h>

#define EPS     1.19209290E-07F
#define TWO_PI  6.28318530717959f       //2*pi

/*const unsigned int WIDTH = 600;
const unsigned int HEIGHT = 600;
const unsigned int DEPTH = 1100;*/

const unsigned int WIDTH = 400;
const unsigned int HEIGHT = 400;
const unsigned int DEPTH = 500;

extern "C" {
    __constant__ Params params;
}



//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

struct RadiancePRD
{
    // TODO: move some state directly into payload registers?
    float3       emitted;
    float3       radiance;
    float3       attenuation;
    float3       origin;
    float3       direction;
    unsigned int seed;
    int          countEmitted;
    int          done;
    float        slen;
    float        dist_so_far;
    unsigned int mc_seed[4];
    float        weight;
    int          print;
    int          depth;
    //int          pad;
};


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------



static __forceinline__ __device__ float xorshift128p_nextf(unsigned long t[2]) {
    union {
        unsigned long  i;
        float f[2];
        unsigned int  u[2];
    } s1;
    const unsigned long s0 = t[1];
    s1.i = t[0];
    t[0] = s0;
    s1.i ^= s1.i << 23; // a
    t[1] = s1.i ^ s0 ^ (s1.i >> 18) ^ (s0 >> 5); // b, c
    s1.i = t[1] + s0;
    s1.u[0] = 0x3F800000U | (s1.u[0] >> 9);

    return s1.f[0] - 1.0f;
}

static __forceinline__ __device__ float mc_next_scatter(float g, unsigned long ran[2], float3* dir) {

    float nextslen;
    float sphi, cphi, tmp0, theta, stheta, ctheta, tmp1;
    float3 p;

    //random scattering length (normalized)
    nextslen = -log(xorshift128p_nextf(ran) + EPS);

    tmp0 = TWO_PI * xorshift128p_nextf(ran); //next arimuth angle
    sphi = sin(tmp0);
    cphi = cos(tmp0);

    if (g > EPS) {  //if g is too small, the distribution of theta is bad
        tmp0 = (1.f - g * g) / (1.f - g + 2.f * g * xorshift128p_nextf(ran));
        tmp0 *= tmp0;
        tmp0 = (1.f + g * g - tmp0) / (2.f * g);
        tmp0 = clamp(tmp0, -1.f, 1.f);

        theta = acos(tmp0);
        stheta = sqrt(1.f - tmp0 * tmp0);
        //stheta=MCX_MATHFUN(sin)(theta);
        ctheta = tmp0;
    }
    else {
        theta = acos(2.f * xorshift128p_nextf(ran) - 1.f);
        stheta = sin(theta);
        ctheta = cos(theta);
    }

    if (dir->z > -1.f + EPS && dir->z < 1.f - EPS) {
        tmp0 = 1.f - dir->z * dir->z;   //reuse tmp to minimize registers
        tmp1 = 1 / sqrt(tmp0);
        tmp1 = stheta * tmp1;

        p.x = tmp1 * (dir->x * dir->z * cphi - dir->y * sphi) + dir->x * ctheta;
        p.y = tmp1 * (dir->y * dir->z * cphi + dir->x * sphi) + dir->y * ctheta;
        p.z = -tmp1 * tmp0 * cphi + dir->z * ctheta;
    }
    else {
        p.x = stheta * cphi;
        p.y = stheta * sphi;
        p.z = (dir->z > 0.f) ? ctheta : -ctheta;
    }

    dir->x = p.x;
    dir->y = p.y;
    dir->z = p.z;
    return nextslen;
}


static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ RadiancePRD* getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<RadiancePRD*>(unpackPointer(u0, u1));
}


static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}


static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD* prd
)
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer(prd, u0, u1);
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        RAY_TYPE_RADIANCE,        // missSBTIndex
        u0, u1);
}


static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    unsigned int occluded = 0u;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                    // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RAY_TYPE_OCCLUSION,      // SBT offset
        RAY_TYPE_COUNT,          // SBT stride
        RAY_TYPE_OCCLUSION,      // missSBTIndex
        occluded);
    return occluded;
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w = params.width;
    const int    h = params.height;
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);
    unsigned int seed1 = tea<4>((idx.y * w + idx.x) + 1, subframe_index);
    unsigned int seed2 = tea<4>((idx.y * w + idx.x) + 2, subframe_index);
    unsigned int seed3 = tea<4>((idx.y * w + idx.x) + 3, subframe_index);

    float3 result = make_float3(0.0f);
    int i = params.samples_per_launch;
    //do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
        ) - 1.0f;

        float3 ray_direction = normalize(-0.3 * U + 0.5 * V + W); //normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        RadiancePRD prd;
        prd.emitted = make_float3(0.f);
        prd.radiance = make_float3(0.f);
        prd.attenuation = make_float3(1.f);
        prd.countEmitted = true;
        prd.done = false;
        prd.mc_seed[0] = seed;
        prd.mc_seed[1] = seed1;
        prd.mc_seed[2] = seed2;
        prd.mc_seed[3] = seed3;

        prd.slen = rnd(seed) * 10;
        prd.dist_so_far = 0.0f;
        prd.weight = 5.0f;

        prd.origin = ray_origin;
        prd.direction = ray_direction;

        int depth = 0;
        for (;; )
        {
            printf("origin: %f, %f, %f; direction: %f, %f, %f \n", ray_origin.x, ray_origin.y, ray_origin.z,
                ray_direction.x, ray_direction.y, ray_direction.z);
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                &prd);

            result += prd.emitted;
            result += prd.radiance * prd.attenuation;

            if (prd.done || depth >= 5) // TODO RR, variable for depth
                break;

            // Ray origin and direction are updated from the trace
            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++depth;
        }
    } 
    //while (--i);

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;
    float3         accum_color = result / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);
}


extern "C" __global__ void __miss__radiance()
{
    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD* prd = getPRD();

    prd->radiance = make_float3(rt_data->bg_color);
    prd->done = true;
}


extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion(true);
}


extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    const int    prim_idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx * 3;

    const float3 v0 = make_float3(rt_data->vertices[vert_idx_offset + 0]);
    const float3 v1 = make_float3(rt_data->vertices[vert_idx_offset + 1]);
    const float3 v2 = make_float3(rt_data->vertices[vert_idx_offset + 2]);
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));

    const float3 N = faceforward(N_0, -ray_dir, N_0);

    const float dist_travelled = optixGetRayTmax();
    const float3 inters_point = optixGetWorldRayOrigin() + dist_travelled * ray_dir;
    RadiancePRD* prd = getPRD();

    printf("hitPoint: %f, %f, %f \n", inters_point.x, inters_point.y, inters_point.z);

    // Smaller scene 
    // x:100-500 ; y:0-400 ; z: 300-800
    if ( (inters_point.x > 500 || inters_point.x < 100) || (inters_point.y > 400 || inters_point.y < 0) || (inters_point.z > 800 || inters_point.z < 300) )
    {
        prd->radiance = make_float3(0.0, 0.0, 0.0);
        printf("out of bounds \n");
        prd->done = true;
        return;
    }


    if (prd->countEmitted)
        prd->emitted = rt_data->emission_color;
    else
        prd->emitted = make_float3(0.0f);

    unsigned int seed = prd->seed;

    float3 prev_origin = prd->origin;
    float3 prev_dir = prd->direction;

    // CHECK g and medium ID
    //printf("g: %f , ID: %f \n", rt_data->g, rt_data->medium_id);

    int medium_id = rt_data->medium_id_down;

    if(dot(-ray_dir, N_0) < 0)
        medium_id = rt_data->medium_id_up;

    // Update ray origin and direction
    // Ray has travelled past its scattering length

    unsigned long rand[2];
    rand[0] = (unsigned long)prd->mc_seed[0] << 32 | prd->mc_seed[1];
    rand[1] = (unsigned long)prd->mc_seed[2] << 32 | prd->mc_seed[3];
    if (prd->dist_so_far >= (prd->slen / params.mu_s[medium_id]))
    {
        prd->origin = prd->origin + (prd->slen * prd->direction);
        prd->dist_so_far = 0;

        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        prd->slen = mc_next_scatter(params.g[medium_id], rand, &prd->direction);
    }
    // Ray has not reached scatter length
    else
    {
        prd->dist_so_far += dist_travelled * params.mu_s[medium_id]; // multiply by mu_s
        prd->origin = inters_point;

        //float slen = mc_next_scatter(params.g[medium_id], rand, &prd->direction);
    }

    //printf("changed origin: %f, %f, %f ; direction: %f, %f, %f \n", prd->origin.x, prd->origin.y, prd->origin.z,
      //  prd->direction.x, prd->direction.y, prd->direction.z);
    // Compute the ray attenuation
    //float distance2 = (prev_origin.x - prd->origin.x) * (prev_origin.x - prd->origin.x) + (prev_origin.y - prd->origin.y) * (prev_origin.y - prd->origin.y) + (prev_origin.z - prd->origin.z) * (prev_origin.z - prd->origin.z);
    //float distance = sqrt(distance2);

    float3 t3 = (prd->origin - prev_origin) / (prev_dir);
    float distance = (t3.x + t3.y + t3.z) / 3;

    uint3 prev_index;
    int change_color = 1;
    for (float i = 0; i < distance; i += 10)
    {
        float3 curr_location = prev_origin + i * prev_dir;
        printf("i: %f, curr_location: %f, %f, %f; weight: %f \n", i, curr_location.x, curr_location.y, curr_location.z, prd->weight);
        uint3 index = make_uint3(curr_location.x-100, curr_location.y, curr_location.z-300);
        if (i > 0 && prev_index == index)
            continue;
        prev_index = index;

        if (index.x > WIDTH || index.y > HEIGHT || index.z > DEPTH)
        {
            //printf("inters: %f, origin: %f, slen: %f \n", inters_point, prd->origin, prd->slen);
            printf("X: %d, Y: %d, Z: %d \n", index.x, index.y, index.z);
        }
        //printf("mu_a: %f\n", params.mu_a[medium_id]);
        float weight_change = prd->weight * (1 - exp(-params.mu_a[medium_id]));
        params.atten_buffer[index.x + ((index.y + (index.z * HEIGHT)) * WIDTH)] += weight_change;
        //printf("weight: %f; index: %d, %d, %d; bufferVal: %f\n", prd->weight, index.x, index.y, index.z, params.atten_buffer[index.x + ((index.y + (index.z * HEIGHT)) * WIDTH)]);
        //printf("weight: %f \n", prd->weight);
        prd->weight -= weight_change;
    }

    printf("======= Final weight value ========== : %f \n", prd->weight);

    {
        {
            prd->attenuation *= rt_data->diffuse_color;
        }
        prd->countEmitted = false;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd->seed = seed;

    ParallelogramLight light = params.light;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - inters_point);
    const float3 L = normalize(light_pos - inters_point);
    const float  nDl = dot(N, L);
    const float  LnDl = -dot(light.normal, L);

    float weight = 0.0f;
    if (nDl > 0.0f && LnDl > 0.0f)
    {
        {
            const float A = length(cross(light.v1, light.v2));
            weight = 3 * nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd->radiance += light.emission * weight;

    /*traceRadiance(
        params.handle,
        prd->origin,
        prd->direction,
        0.01f,  // tmin       // TODO: smarter offset
        1e16f,  // tmax
        prd);*/
}


