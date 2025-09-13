#version 450

#include "config.h"

layout(location = 0) in vec2 i_uv;
layout(location = 0) out vec4 o_color;
layout(set = 2, binding = 0) uniform sampler3D s_trail;

const vec3 colors[] = vec3[COLOR_COUNT]
(
    vec3(1.0f, 0.0f, 0.0f), /* red */
    vec3(0.0f, 1.0f, 0.0f), /* green */
    vec3(0.0f, 0.0f, 1.0f), /* blue */
    vec3(1.0f, 1.0f, 1.0f), /* white */
    vec3(1.0f, 0.0f, 1.0f), /* magenta */
    vec3(0.0f, 1.0f, 1.0f), /* cyan */
    vec3(1.0f, 1.0f, 0.0f)  /* yellow */
);

void main()
{
    float highest = 0.0;
    o_color = vec4(0.0f);
    
    for (int i = 0; i < COLOR_COUNT; i++)
    {
        const ivec3 coord = ivec3(i_uv * vec2(textureSize(s_trail, 0)), i); 
        const float count = texelFetch(s_trail, coord, 0).x;
        if (count > highest)
        {
            o_color = vec4(colors[i], count / 2.0f) * 3.0f;
            highest = count;
        }
    }
}