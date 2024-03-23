//--------------------------------------------------------------------------------------------------------------------//
// #version 430 
#version 430 compatibility
//--------------------------------------------------------------------------------------------------------------------//
#extension GL_NV_shader_atomic_float : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_compute_variable_group_size : enable
//--------------------------------------------------------------------------------------------------------------------//
#define GLSL
const float PI = 3.1415926535897932384626433832795;
const float InvPI = 1.f / PI;
const float EPS = 0.00001f;
//--------------------------------------------------------------------------------------------------------------------//



