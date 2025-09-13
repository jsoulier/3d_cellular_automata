#pragma once

#include <SDL3/SDL.h>
#include <assert.h>

#undef assert
#ifndef NDEBUG
#define assert(e) SDL_assert_always(e)
#else
#define assert(e)
#endif

SDL_GPUShader* load_shader(
    SDL_GPUDevice* device,
    const char* file);
SDL_GPUComputePipeline* load_compute_pipeline(
    SDL_GPUDevice* device,
    const char* file);