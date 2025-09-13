#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <stdbool.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "util.h"

typedef struct
{
    float x;
    float y;
    float angle;
    uint32_t color;
}
agent_t;

static SDL_Window* window;
static SDL_GPUDevice* device;
static SDL_GPUComputePipeline* update_pipeline;
static SDL_GPUComputePipeline* blur_pipeline;
static SDL_GPUGraphicsPipeline* draw_pipeline;
static SDL_GPUBuffer* agent_buffer;
static SDL_GPUTexture* trail_texture1;
static SDL_GPUTexture* trail_texture2;
static SDL_GPUSampler* sampler;
static bool loaded;

static bool reload(const char* path)
{
    loaded = false;
    srand(time(NULL));
    SDL_ReleaseGPUBuffer(device, agent_buffer);
    SDL_ReleaseGPUTexture(device, trail_texture1);
    SDL_ReleaseGPUTexture(device, trail_texture2);
    agent_buffer = NULL;
    trail_texture1 = NULL;
    trail_texture2 = NULL;
    int channels;
    int w;
    int h;
    stbi_uc* src = stbi_load(path, &w, &h, &channels, 3);
    if (!src)
    {
        SDL_Log("Failed to load image: %s", path);
        return false;
    }
    channels = 3;
    stbi_uc* dst = malloc(WIDTH * HEIGHT * channels);
    if (!dst)
    {
        SDL_Log("Failed to allocate image");
        return false;
    }
    if (!stbir_resize_uint8_srgb(src, w, h, 0, dst, WIDTH, HEIGHT, 0, channels))
    {
        SDL_Log("Failed to resize image");
        return false;
    }
    stbi_image_free(src);
    const uint32_t colors[COLOR_COUNT] =
    {
        0x0000FF, /* red */
        0x00FF00, /* green */
        0xFF0000, /* blue */
        0xFFFFFF, /* white */
        0xFF00FF, /* magenta */
        0xFFFF00, /* cyan */
        0x00FFFF, /* yellow */
    };
    agent_t* agents = malloc(WIDTH * HEIGHT * sizeof(agent_t));
    if (!agents)
    {
        SDL_Log("Failed to allocate agents");
        return false;
    }
    for (uint32_t x = 0; x < WIDTH; x++)
    for (uint32_t y = 0; y < HEIGHT; y++)
    {
        const uint32_t index = (y * WIDTH + x) * channels;
        uint32_t color1 = 0;
        color1 |= dst[index + 0] << 0;
        color1 |= dst[index + 1] << 8;
        color1 |= dst[index + 2] << 16;
        double distance1 = DBL_MAX;
        uint32_t color = UINT32_MAX;
        for (uint32_t i = 0; i < COLOR_COUNT; i++)
        {
            const uint32_t color2 = colors[i];
            const int r1 = (color1 >> 0) & 0xFF;
            const int g1 = (color1 >> 8) & 0xFF;
            const int b1 = (color1 >> 16) & 0xFF;
            const int r2 = (color2 >> 0) & 0xFF;
            const int g2 = (color2 >> 8) & 0xFF;
            const int b2 = (color2 >> 16) & 0xFF;
            const double distance2 =
                (r1 - r2) * (r1 - r2) + 
                (g1 - g2) * (g1 - g2) + 
                (b1 - b2) * (b1 - b2);
            if (distance2 < distance1)
            {
                distance1 = distance2;
                color = i;
            }
        }
        agent_t* agent = &agents[y * WIDTH + x];
        agent->x = x;
        agent->y = y;
        agent->angle = (float) rand() / RAND_MAX * SDL_PI_F * 2.0f;
        agent->color = color;
    }
    SDL_GPUCommandBuffer* cb = SDL_AcquireGPUCommandBuffer(device);
    if (!cb)
    {
        SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
        return false;
    }
    SDL_GPUBufferCreateInfo bci = {0};
    bci.size = WIDTH * HEIGHT * sizeof(agent_t);
    bci.usage =
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ |
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;
    agent_buffer = SDL_CreateGPUBuffer(device, &bci);
    if (!agent_buffer)
    {
        SDL_Log("Failed to create buffer: %s", SDL_GetError());
        return false;
    }
    SDL_GPUTransferBufferCreateInfo tbci = {0};
    tbci.size = bci.size;
    tbci.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
    SDL_GPUTransferBuffer* tbo = SDL_CreateGPUTransferBuffer(device, &tbci);
    if (!tbo)
    {
        SDL_Log("Failed to create transfer buffer: %s", SDL_GetError());
        return false;
    }
    void* data = SDL_MapGPUTransferBuffer(device, tbo, false);
    if (!data)
    {
        SDL_Log("Failed to map transfer buffer: %s", SDL_GetError());
        return false;
    }
    memcpy(data, agents, bci.size);
    SDL_UnmapGPUTransferBuffer(device, tbo);
    SDL_GPUTransferBufferLocation tbl = {0};
    SDL_GPUBufferRegion br = {0};
    tbl.transfer_buffer = tbo;
    br.buffer = agent_buffer;
    br.size = bci.size;
    SDL_GPUCopyPass* pass = SDL_BeginGPUCopyPass(cb);
    if (!pass)
    {
        SDL_Log("Failed to begin copy pass: %s", SDL_GetError());
        return false;
    }
    SDL_UploadToGPUBuffer(pass, &tbl, &br, false);
    SDL_EndGPUCopyPass(pass);
    SDL_ReleaseGPUTransferBuffer(device, tbo);
    SDL_GPUTextureCreateInfo tci = {0};
    tci.type = SDL_GPU_TEXTURETYPE_3D;
    tci.format = SDL_GPU_TEXTUREFORMAT_R32_FLOAT;
    tci.usage =
        SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
        SDL_GPU_TEXTUREUSAGE_SAMPLER |
        SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    tci.width = WIDTH;
    tci.height = HEIGHT;
    tci.layer_count_or_depth = COLOR_COUNT;
    tci.num_levels = 1;
    trail_texture1 = SDL_CreateGPUTexture(device, &tci);
    trail_texture2 = SDL_CreateGPUTexture(device, &tci);
    if (!trail_texture1 || !trail_texture2)
    {
        SDL_Log("Failed to create texture(s): %s", SDL_GetError());
        return false;
    }
    {
        SDL_GPUColorTargetInfo cti[2] = {0};
        cti[0].texture = trail_texture1;
        cti[0].load_op = SDL_GPU_LOADOP_CLEAR;
        cti[0].store_op = SDL_GPU_STOREOP_STORE;
        cti[1].texture = trail_texture2;
        cti[1].load_op = SDL_GPU_LOADOP_CLEAR;
        cti[1].store_op = SDL_GPU_STOREOP_STORE;
        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cb, cti, 2, NULL);
        if (!pass)
        {
            SDL_Log("Failed to begin render pass: %s", SDL_GetError());
            return false;
        }
        SDL_EndGPURenderPass(pass);
    }
    SDL_SubmitGPUCommandBuffer(cb);
    free(agents);
    free(dst);
    loaded = true;
    return true;
}

int main(int argc, char** argv)
{
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
    SDL_SetAppMetadata("png2slime", NULL, NULL);
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return 1;
    }
    if (!(window = SDL_CreateWindow("png2slime", 960, 540, SDL_WINDOW_RESIZABLE)))
    {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        return 1;
    }
    if (!(device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, NULL)))
    {
        SDL_Log("Failed to create device: %s", SDL_GetError());
        return 1;
    }
    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("Failed to create swapchain: %s", SDL_GetError());
        return 1;
    }
    SDL_GPUShader* draw_shader = load_shader(device, "draw.frag");
    SDL_GPUShader* quad_shader = load_shader(device, "quad.vert");
    blur_pipeline = load_compute_pipeline(device, "blur.comp");
    update_pipeline = load_compute_pipeline(device, "update.comp");
    if (!draw_shader || !quad_shader || !blur_pipeline || !update_pipeline)
    {
        SDL_Log("Failed to load shader(s)");
        return false;
    }
    draw_pipeline = SDL_CreateGPUGraphicsPipeline(device,
        &(SDL_GPUGraphicsPipelineCreateInfo)
    {
        .vertex_shader = quad_shader,
        .fragment_shader = draw_shader,
        .target_info =
        {
            .num_color_targets = 1,
            .color_target_descriptions = &(SDL_GPUColorTargetDescription)
            {
                .format = SDL_GetGPUSwapchainTextureFormat(device, window),
                .blend_state =
                {
                    .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
                    .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                    .color_blend_op = SDL_GPU_BLENDOP_ADD,
                    .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
                    .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                    .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
                    .enable_blend = true,
                }
            },
        },
    });
    if (!draw_pipeline)
    {
        SDL_Log("Failed to create draw pipeline: %s", SDL_GetError());
        return 1;
    }
    SDL_ReleaseGPUShader(device, draw_shader);
    SDL_ReleaseGPUShader(device, quad_shader);
    SDL_GPUSamplerCreateInfo sci = {0};
    sci.min_filter = SDL_GPU_FILTER_NEAREST;
    sci.mag_filter = SDL_GPU_FILTER_NEAREST;
    sci.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sci.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sci.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sci.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler = SDL_CreateGPUSampler(device, &sci);
    if (!sampler)
    {
        SDL_Log("Failed to create sampler: %s", SDL_GetError());
        return 1;
    }
    if (argc >= 2 && !reload(argv[1]))
    {
        SDL_Log("Failed to load image");
        return 1;
    }
    bool running = true;
    uint64_t t1 = SDL_GetPerformanceCounter();
    uint64_t t2 = 0;
    while (running)
    {
        t2 = t1;
        t1 = SDL_GetPerformanceCounter();
        const float frequency = SDL_GetPerformanceFrequency();
        const float dt = (t1 - t2) / frequency;
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_DROP_FILE:
                reload(event.drop.data);
                break;
            }
        }
        if (!loaded)
        {
            continue;
        }
        SDL_WaitForGPUSwapchain(device, window);
        SDL_GPUCommandBuffer* cb = SDL_AcquireGPUCommandBuffer(device);
        if (!cb)
        {
            SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
            continue;
        }
        SDL_GPUTexture* texture;
        if (!SDL_WaitAndAcquireGPUSwapchainTexture(cb, window, &texture, NULL, NULL))
        {
            SDL_Log("Failed to acquire swapchain texture: %s", SDL_GetError());
            SDL_CancelGPUCommandBuffer(cb);
            continue;
        }
        {
            SDL_PushGPUDebugGroup(cb, "blit");
            SDL_GPUBlitInfo info = {0};
            info.source.texture = trail_texture1;
            info.source.w = WIDTH;
            info.source.h = HEIGHT;
            info.destination.texture = trail_texture2;
            info.destination.w = WIDTH;
            info.destination.h = HEIGHT;
            for (int i = 0; i < COLOR_COUNT; i++)
            {
                info.source.layer_or_depth_plane = i;
                info.destination.layer_or_depth_plane = i;
                SDL_BlitGPUTexture(cb, &info);
            }
            SDL_PopGPUDebugGroup(cb);
        }
        {
            SDL_PushGPUDebugGroup(cb, "update");
            SDL_GPUStorageBufferReadWriteBinding sbb = {0};
            sbb.buffer = agent_buffer;
            SDL_GPUStorageTextureReadWriteBinding stb = {0};
            stb.texture = trail_texture2;
            stb.cycle = true;
            SDL_GPUComputePass* pass = SDL_BeginGPUComputePass(cb, &stb, 1, &sbb, 1);
            if (!pass)
            {
                SDL_PopGPUDebugGroup(cb);
                SDL_SubmitGPUCommandBuffer(cb);
                SDL_Log("Failed to begin update pass: %s", SDL_GetError());
                continue;
            }
            SDL_GPUTextureSamplerBinding tsb = {0};
            tsb.sampler = sampler;
            tsb.texture = trail_texture1;
            SDL_BindGPUComputePipeline(pass, update_pipeline);
            SDL_BindGPUComputeSamplers(pass, 0, &tsb, 1);
            const int x = (float) (WIDTH + THREADS_X - 1) / THREADS_X;
            const int y = (float) (HEIGHT + THREADS_Y - 1) / THREADS_Y;
            SDL_PushGPUComputeUniformData(cb, 0, &t2, sizeof(t2));
            SDL_PushGPUComputeUniformData(cb, 1, &dt, sizeof(dt));
            SDL_DispatchGPUCompute(pass, x, y, 1);
            SDL_EndGPUComputePass(pass);
            SDL_PopGPUDebugGroup(cb);
        }
        {
            SDL_PushGPUDebugGroup(cb, "blur");
            SDL_GPUStorageTextureReadWriteBinding stb = {0};
            stb.texture = trail_texture1;
            stb.cycle = true;
            SDL_GPUComputePass* pass = SDL_BeginGPUComputePass(cb, &stb, 1, NULL, 0);
            if (!pass)
            {
                SDL_PopGPUDebugGroup(cb);
                SDL_SubmitGPUCommandBuffer(cb);
                SDL_Log("Failed to begin blur pass: %s", SDL_GetError());
                continue;
            }
            SDL_GPUTextureSamplerBinding tsb = {0};
            tsb.sampler = sampler;
            tsb.texture = trail_texture2;
            SDL_BindGPUComputePipeline(pass, blur_pipeline);
            SDL_BindGPUComputeSamplers(pass, 0, &tsb, 1);
            const int x = (float) (WIDTH + THREADS_X - 1) / THREADS_X;
            const int y = (float) (HEIGHT + THREADS_Y - 1) / THREADS_Y;
            SDL_DispatchGPUCompute(pass, x, y, 1);
            SDL_EndGPUComputePass(pass);
            SDL_PopGPUDebugGroup(cb);
        }
        {
            SDL_PushGPUDebugGroup(cb, "draw");
            SDL_GPUColorTargetInfo cti = {0};
            cti.texture = texture;
            cti.load_op = SDL_GPU_LOADOP_CLEAR;
            cti.store_op = SDL_GPU_STOREOP_STORE;
            SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cb, &cti, 1, NULL);
            if (!pass)
            {
                SDL_PopGPUDebugGroup(cb);
                SDL_SubmitGPUCommandBuffer(cb);
                SDL_Log("Failed to begin draw pass: %s", SDL_GetError());
                continue;
            }
            SDL_GPUTextureSamplerBinding binding = {0};
            binding.texture = trail_texture1;
            binding.sampler = sampler;
            SDL_BindGPUGraphicsPipeline(pass, draw_pipeline);
            SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
            SDL_DrawGPUPrimitives(pass, 4, 1, 0, 0);
            SDL_EndGPURenderPass(pass);
            SDL_PopGPUDebugGroup(cb);
        }
        SDL_SubmitGPUCommandBuffer(cb);
    }
    SDL_ReleaseGPUBuffer(device, agent_buffer);
    SDL_ReleaseGPUTexture(device, trail_texture1);
    SDL_ReleaseGPUTexture(device, trail_texture2);
    SDL_ReleaseGPUSampler(device, sampler);
    SDL_ReleaseGPUGraphicsPipeline(device, draw_pipeline);
    SDL_ReleaseGPUComputePipeline(device, blur_pipeline);
    SDL_ReleaseGPUComputePipeline(device, update_pipeline);
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}