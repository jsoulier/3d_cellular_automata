#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlgpu3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "config.hpp"
#include "shader.hpp"

static_assert(BOUNDS < 1024);
static_assert(FRAMES == 2, "not implemented");

static SDL_Window* window;
static SDL_GPUDevice* device;
static SDL_GPUGraphicsPipeline* graphicsPipeline;
static SDL_GPUComputePipeline* computePipeline;
static SDL_GPUTexture* textures[FRAMES];
static int readFrame{0};
static int writeFrame{1};
static SDL_GPUBuffer* vertexBuffer;
static SDL_GPUBuffer* instanceBuffer;
static SDL_GPUTexture* depthTexture;
static int depthTextureWidth;
static int depthTextureHeight;
static float pitch;
static float yaw;
static float distance{256.0f};
static uint64_t time1;
static uint64_t time2;
static float delta;
static float delay{10.0f};
static bool imGuiFocused;
static bool imGuiHovered;

struct
{
    uint32_t seed{0};
    uint32_t surviveMask{16};
    uint32_t birthMask{96};
    uint32_t life{32};
    uint32_t neighborhood{MOORE};
    uint32_t frame{0};
}
static rules;

static bool Init()
{
    SDL_SetAppMetadata("3D Cellular Automata", nullptr, nullptr);
    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Failed to initialize SDL: %s", SDL_GetError());
        return false;
    }
    window = SDL_CreateWindow("3D Cellular Automata", 960, 720, SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        SDL_Log("Failed to create window: %s", SDL_GetError());
        return false;
    }
#if defined(SDL_PLATFORM_WIN32)
    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_DXIL, true, nullptr);
#elif defined(SDL_PLATFORM_APPLE)
    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, true, nullptr);
#else
    device = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, true, nullptr);
#endif
    if (!device)
    {
        SDL_Log("Failed to create device: %s", SDL_GetError());
        return false;
    }
    if (!SDL_ClaimWindowForGPUDevice(device, window))
    {
        SDL_Log("Failed to create swapchain: %s", SDL_GetError());
        return false;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplSDL3_InitForSDLGPU(window);
    ImGui_ImplSDLGPU3_InitInfo info{};
    info.Device = device;
    info.ColorTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
    ImGui_ImplSDLGPU3_Init(&info);
    return true;
}

static bool CreatePipelines()
{
    SDL_GPUShader* vertShader = LoadShader(device, "render.vert");
    SDL_GPUShader* fragShader = LoadShader(device, "render.frag");
    if (!vertShader || !fragShader)
    {
        SDL_Log("Failed to load shader(s)");
        return false;
    }
    SDL_GPUVertexBufferDescription buffers[2] =
    {{
        .slot = 0,
        .pitch = sizeof(float) * 3,
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0,
    },
    {
        .slot = 1,
        .pitch = sizeof(uint32_t),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_INSTANCE,
        .instance_step_rate = 0,
    }};
    SDL_GPUVertexAttribute attribs[2] =
    {{
        .location = 0,
        .buffer_slot = 0,
        .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
        .offset = 0,
    },
    {
        .location = 1,
        .buffer_slot = 1,
        .format = SDL_GPU_VERTEXELEMENTFORMAT_UINT,
        .offset = 0,
    }};
    SDL_GPUColorTargetDescription targets[1] =
    {{
        .format = SDL_GetGPUSwapchainTextureFormat(device, window),
    }};
    SDL_GPUGraphicsPipelineCreateInfo info{};
    info.vertex_shader = vertShader;
    info.fragment_shader = fragShader;
    info.vertex_input_state.vertex_buffer_descriptions = buffers;
    info.vertex_input_state.num_vertex_buffers = 2;
    info.vertex_input_state.vertex_attributes = attribs;
    info.vertex_input_state.num_vertex_attributes = 2;
    info.target_info.color_target_descriptions = targets;
    info.target_info.num_color_targets = 1;
    info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    info.target_info.has_depth_stencil_target = true;
    info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
    info.depth_stencil_state.enable_depth_test = true;
    info.depth_stencil_state.enable_depth_write = true;
    graphicsPipeline = SDL_CreateGPUGraphicsPipeline(device, &info);
    computePipeline = LoadComputePipeline(device, "automata.comp");
    if (!graphicsPipeline || !computePipeline)
    {
        SDL_Log("Failed to create pipeline(s): %s", SDL_GetError());
        return false;
    }
    SDL_ReleaseGPUShader(device, vertShader);
    SDL_ReleaseGPUShader(device, fragShader);
    return true;
}

static bool CreateResources()
{
    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(device);
    if (!commandBuffer)
    {
        SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
        return false;
    }
    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
    if (!copyPass)
    {
        SDL_Log("Failed to begin copy pass: %s", SDL_GetError());
        return false;
    }
    for (int i = 0; i < FRAMES; i++)
    {
        SDL_GPUTextureCreateInfo info{};
        info.type = SDL_GPU_TEXTURETYPE_3D;
        info.format = SDL_GPU_TEXTUREFORMAT_R8_UINT;
        info.usage =
            SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
            SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
            SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ;
        info.width = BOUNDS;
        info.height = BOUNDS;
        info.layer_count_or_depth = BOUNDS;
        info.num_levels = 1;
        textures[i] = SDL_CreateGPUTexture(device, &info);
        if (!textures[i])
        {
            SDL_Log("Failed to create texture: %s", SDL_GetError());
            return false;
        }
    }
    {
        float vertices[36 * 3] =
        {
           -0.5f,-0.5f, 0.5f,
            0.5f,-0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
           -0.5f,-0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
           -0.5f, 0.5f, 0.5f,
           -0.5f,-0.5f,-0.5f,
            0.5f, 0.5f,-0.5f,
            0.5f,-0.5f,-0.5f,
           -0.5f,-0.5f,-0.5f,
           -0.5f, 0.5f,-0.5f,
            0.5f, 0.5f,-0.5f,
           -0.5f,-0.5f,-0.5f,
           -0.5f,-0.5f, 0.5f,
           -0.5f, 0.5f, 0.5f,
           -0.5f,-0.5f,-0.5f,
           -0.5f, 0.5f, 0.5f,
           -0.5f, 0.5f,-0.5f,
            0.5f,-0.5f,-0.5f,
            0.5f, 0.5f, 0.5f,
            0.5f,-0.5f, 0.5f,
            0.5f,-0.5f,-0.5f,
            0.5f, 0.5f,-0.5f,
            0.5f, 0.5f, 0.5f,
           -0.5f, 0.5f,-0.5f,
           -0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
           -0.5f, 0.5f,-0.5f,
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f,-0.5f,
           -0.5f,-0.5f,-0.5f,
            0.5f,-0.5f, 0.5f,
           -0.5f,-0.5f, 0.5f,
           -0.5f,-0.5f,-0.5f,
            0.5f,-0.5f,-0.5f,
            0.5f,-0.5f, 0.5f,
        };
        SDL_GPUTransferBuffer* transferBuffer;
        {
            SDL_GPUTransferBufferCreateInfo info{};
            info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
            info.size = sizeof(vertices);
            transferBuffer = SDL_CreateGPUTransferBuffer(device, &info);
            if (!transferBuffer)
            {
                SDL_Log("Failed to create transfer buffer: %s", SDL_GetError());
                return false;
            }
        }
        void* data = SDL_MapGPUTransferBuffer(device, transferBuffer, false);
        if (!data)
        {
            SDL_Log("Failed to map transfer buffer: %s", SDL_GetError());
            return false;
        }
        {
            SDL_GPUBufferCreateInfo info{};
            info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
            info.size = sizeof(vertices);
            vertexBuffer = SDL_CreateGPUBuffer(device, &info);
            if (!vertexBuffer)
            {
                SDL_Log("Failed to create buffer: %s", SDL_GetError());
                return false;
            }
        }
        std::memcpy(data, vertices, sizeof(vertices));
        SDL_GPUTransferBufferLocation location{};
        SDL_GPUBufferRegion region{};
        location.transfer_buffer = transferBuffer;
        region.buffer = vertexBuffer;
        region.size = sizeof(vertices);
        SDL_UploadToGPUBuffer(copyPass, &location, &region, false);
        SDL_ReleaseGPUTransferBuffer(device, transferBuffer);
    }
    {
        SDL_GPUTransferBuffer* transferBuffer;
        {
            SDL_GPUTransferBufferCreateInfo info{};
            info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
            info.size = BOUNDS * BOUNDS * BOUNDS * sizeof(uint32_t);
            transferBuffer = SDL_CreateGPUTransferBuffer(device, &info);
            if (!transferBuffer)
            {
                SDL_Log("Failed to create transfer buffer: %s", SDL_GetError());
                return 1;
            }
        }
        uint32_t* data = static_cast<uint32_t*>(SDL_MapGPUTransferBuffer(device, transferBuffer, false));
        if (!data)
        {
            SDL_Log("Failed to map transfer buffer: %s", SDL_GetError());
            return false;
        }
        {
            SDL_GPUBufferCreateInfo info{};
            info.usage = SDL_GPU_BUFFERUSAGE_VERTEX;
            info.size = BOUNDS * BOUNDS * BOUNDS * sizeof(uint32_t);
            instanceBuffer = SDL_CreateGPUBuffer(device, &info);
            if (!instanceBuffer)
            {
                SDL_Log("Failed to create buffer: %s", SDL_GetError());
                return false;
            }
        }
        uint32_t i = 0;
        for (int x = 0; x < BOUNDS; x++)
        for (int y = 0; y < BOUNDS; y++)
        for (int z = 0; z < BOUNDS; z++)
        {
            data[i] = 0;
            data[i] |= x << 0;
            data[i] |= y << 10;
            data[i] |= z << 20;
            i++;
        }
        SDL_GPUTransferBufferLocation location{};
        SDL_GPUBufferRegion region{};
        location.transfer_buffer = transferBuffer;
        region.buffer = instanceBuffer;
        region.size = BOUNDS * BOUNDS * BOUNDS * sizeof(uint32_t);
        SDL_UploadToGPUBuffer(copyPass, &location, &region, false);
        SDL_ReleaseGPUTransferBuffer(device, transferBuffer);
    }
    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(commandBuffer);
    return true;
}

static void DrawImGui()
{
    ImGui_ImplSDLGPU3_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Settings");
    imGuiHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
    imGuiFocused = ImGui::IsWindowFocused();
    if (ImGui::Button("Reset"))
    {
        rules.seed = std::rand() % RAND_MAX;
        rules.frame = 0;
    }
    ImGui::SliderFloat("Speed", &delay, 0.0f, 1000.0f);
    ImGui::Text("Survive");
    for (int i = 1; i < 27; i++)
    {
        bool bit = (rules.surviveMask >> i) & 1;
        if (i % 7 == 0)
        {
            ImGui::NewLine();
        }
        ImGui::PushID(i);
        if (ImGui::Checkbox("", &bit))
        {
            if (bit)
            {
                rules.surviveMask |= (1 << i);
            }
            else
            {
                rules.surviveMask &= ~(1 << i);
            }
        }
        ImGui::SameLine();
        ImGui::Text("%02d", i);
        ImGui::SameLine();
        ImGui::PopID();
    }
    ImGui::NewLine();
    ImGui::Text("Birth");
    for (int i = 1; i < 27; i++)
    {
        bool bit = (rules.birthMask >> i) & 1;
        if (i % 7 == 0)
        {
            ImGui::NewLine();
        }
        ImGui::PushID(32 + i);
        if (ImGui::Checkbox("", &bit))
        {
            if (bit)
            {
                rules.birthMask |= (1 << i);
            }
            else
            {
                rules.birthMask &= ~(1 << i);
            }
        }
        ImGui::SameLine();
        ImGui::Text("%02d", i);
        ImGui::SameLine();
        ImGui::PopID();
    }
    ImGui::NewLine();
    int life = rules.life;
    int neighborhood = rules.neighborhood;
    ImGui::SliderInt("Life", &life, 1, 64);
    ImGui::Text("Neighborhood");
    ImGui::RadioButton("Moore", &neighborhood, 0);
    ImGui::RadioButton("Von Neumann", &neighborhood, 1);
    rules.life = life;
    rules.neighborhood = neighborhood;
    ImGui::End();
    ImGui::Render();
}

static void Draw()
{
    SDL_WaitForGPUSwapchain(device, window);
    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(device);
    if (!commandBuffer)
    {
        SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
        return;
    }
    SDL_GPUTexture* texture;
    uint32_t width;
    uint32_t height;
    if (!SDL_AcquireGPUSwapchainTexture(commandBuffer, window, &texture, &width, &height))
    {
        SDL_Log("Failed to acquire swapchain texture: %s", SDL_GetError());
        SDL_CancelGPUCommandBuffer(commandBuffer);
        return;
    }
    if (!texture || !width || !height)
    {
        /* happens on minimize */
        SDL_SubmitGPUCommandBuffer(commandBuffer);
        return;
    }
    if (width != depthTextureWidth || height != depthTextureHeight)
    {
        SDL_ReleaseGPUTexture(device, depthTexture);
        SDL_GPUTextureCreateInfo info{};
        info.type = SDL_GPU_TEXTURETYPE_2D;
        info.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
        info.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
        info.width = width;
        info.height = height;
        info.layer_count_or_depth = 1;
        info.num_levels = 1;
        info.props = SDL_CreateProperties();
        SDL_SetFloatProperty(info.props, SDL_PROP_GPU_TEXTURE_CREATE_D3D12_CLEAR_DEPTH_FLOAT, 1.0f);
        depthTexture = SDL_CreateGPUTexture(device, &info);
        SDL_DestroyProperties(info.props);
        if (!depthTexture)
        {
            SDL_Log("Failed to create texture: %s", SDL_GetError());
            SDL_SubmitGPUCommandBuffer(commandBuffer);
            return;
        }
        depthTextureWidth = width;
        depthTextureHeight = height;
    }
    glm::vec3 vector;
    vector.x = std::cos(pitch) * std::cos(yaw);
    vector.y = std::sin(pitch);
    vector.z = std::cos(pitch) * std::sin(yaw);
    float ratio = static_cast<float>(width) / height;
    glm::vec3 center = glm::vec3{BOUNDS / 2};
    glm::vec3 position = center - vector * distance;
    glm::mat4 view = glm::lookAt(position, position + vector, glm::vec3{0.0f, 1.0f, 0.0f});
    glm::mat4 proj = glm::perspective(FOV, ratio, NEAR, FAR);
    glm::mat4 viewProjMatrix = proj * view;
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize.x = width;
    io.DisplaySize.y = height;
    DrawImGui();
    ImDrawData* drawData = ImGui::GetDrawData();
    ImGui_ImplSDLGPU3_PrepareDrawData(drawData, commandBuffer);
    {
        SDL_GPUColorTargetInfo colorInfo{};
        SDL_GPUDepthStencilTargetInfo depthInfo{};
        colorInfo.texture = texture;
        colorInfo.load_op = SDL_GPU_LOADOP_CLEAR;
        colorInfo.store_op = SDL_GPU_STOREOP_STORE;
        depthInfo.texture = depthTexture;
        depthInfo.load_op = SDL_GPU_LOADOP_CLEAR;
        depthInfo.stencil_load_op = SDL_GPU_LOADOP_CLEAR;
        depthInfo.store_op = SDL_GPU_STOREOP_STORE;
        depthInfo.clear_depth = 1.0f;
        depthInfo.cycle = true;
        SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, &depthInfo);
        if (!renderPass)
        {
            SDL_Log("Failed to begin render pass: %s", SDL_GetError());
            SDL_SubmitGPUCommandBuffer(commandBuffer);
            return;
        }
        SDL_BindGPUGraphicsPipeline(renderPass, graphicsPipeline);
        SDL_GPUBufferBinding vertexBuffers[2]{};
        vertexBuffers[0].buffer = vertexBuffer;
        vertexBuffers[1].buffer = instanceBuffer;
        /* TODO: read or write, which is better? */
        SDL_BindGPUVertexBuffers(renderPass, 0, vertexBuffers, 2);
        SDL_BindGPUVertexStorageTextures(renderPass, 0, &textures[writeFrame], 1);
        SDL_PushGPUVertexUniformData(commandBuffer, 0, &viewProjMatrix, sizeof(viewProjMatrix));
        SDL_PushGPUFragmentUniformData(commandBuffer, 0, &rules, sizeof(rules));
        SDL_DrawGPUPrimitives(renderPass, 36, BOUNDS * BOUNDS * BOUNDS, 0, 0);
        SDL_EndGPURenderPass(renderPass);
    }
    {
        SDL_GPUColorTargetInfo info{};
        info.texture = texture;
        info.load_op = SDL_GPU_LOADOP_LOAD;
        info.store_op = SDL_GPU_STOREOP_STORE;
        SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &info, 1, nullptr);
        if (!renderPass)
        {
            SDL_Log("Failed to begin render pass: %s", SDL_GetError());
            SDL_SubmitGPUCommandBuffer(commandBuffer);
            return;
        }
        ImGui_ImplSDLGPU3_RenderDrawData(drawData, commandBuffer, renderPass);
        SDL_EndGPURenderPass(renderPass);
    }
    SDL_SubmitGPUCommandBuffer(commandBuffer);
}

static void Simulate()
{
    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(device);
    if (!commandBuffer)
    {
        SDL_Log("Failed to acquire command buffer: %s", SDL_GetError());
        return;
    }
    SDL_GPUStorageTextureReadWriteBinding textureBinding{};
    textureBinding.texture = textures[writeFrame];
    SDL_GPUComputePass* computePass = SDL_BeginGPUComputePass(commandBuffer, &textureBinding, 1, nullptr, 0);
    if (!computePass)
    {
        SDL_Log("Failed to begin compute pass: %s", SDL_GetError());
        SDL_SubmitGPUCommandBuffer(commandBuffer);
        return;
    }
    SDL_BindGPUComputePipeline(computePass, computePipeline);
    SDL_PushGPUComputeUniformData(commandBuffer, 0, &rules, sizeof(rules));
    SDL_BindGPUComputeStorageTextures(computePass, 0, &textures[readFrame], 1);
    int groups = (BOUNDS + THREADS - 1) / THREADS;
    SDL_DispatchGPUCompute(computePass, groups, groups, groups);
    SDL_EndGPUComputePass(computePass);
    SDL_SubmitGPUCommandBuffer(commandBuffer);
    readFrame = (readFrame + 1) % FRAMES;
    writeFrame = (writeFrame + 1) % FRAMES;
    rules.frame++;
}

int main(int argc, char** argv)
{
    if (!Init())
    {
        SDL_Log("Failed to initialize");
        return 1;
    }
    if (!CreatePipelines())
    {
        SDL_Log("Failed to create pipelines");
        return 1;
    }
    if (!CreateResources())
    {
        SDL_Log("Failed to create resources");
        return 1;
    }
    std::srand(std::time(nullptr));
    rules.seed = std::rand() % RAND_MAX;
    bool running = true;
    while (running)
    {
        time2 = SDL_GetTicks();
        delta += time2 - time1;
        time1 = time2;
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL3_ProcessEvent(&event);
            switch (event.type)
            {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (!imGuiFocused && !imGuiHovered && event.motion.state & SDL_BUTTON_LMASK)
                {
                    yaw += event.motion.xrel * PAN;
                    pitch -= event.motion.yrel * PAN;
                    float clamp = glm::pi<float>() / 2.0f - 0.01f;
                    pitch = std::clamp(pitch, -clamp, clamp);
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                if (!imGuiHovered)
                {
                    distance -= event.wheel.y * ZOOM;
                    distance = std::max(1.0f, distance);
                }
                break;
            case SDL_EVENT_KEY_DOWN:
                if (event.key.scancode == SDL_SCANCODE_R)
                {
                    rules.seed = std::rand() % RAND_MAX;
                    rules.frame = 0;
                }
                break;
            }
        }
        if (!running)
        {
            break;
        }
        Draw();
        if (delta < delay)
        {
            continue;
        }
        delta = 0.0f;
        Simulate();
    }
    for (int i = 0; i < FRAMES; i++)
    {
        SDL_ReleaseGPUTexture(device, textures[i]);
    }
    SDL_ReleaseGPUTexture(device, depthTexture);
    SDL_ReleaseGPUBuffer(device, vertexBuffer);
    SDL_ReleaseGPUBuffer(device, instanceBuffer);
    ImGui_ImplSDLGPU3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_ReleaseGPUGraphicsPipeline(device, graphicsPipeline);
    SDL_ReleaseGPUComputePipeline(device, computePipeline);
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyGPUDevice(device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}