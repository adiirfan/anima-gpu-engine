// Vulkan runtime for the engine: one compute queue, plain storage buffers, pipelines built from SPIR-V with
// specialization constants (tile sizes chosen by the tuner), push constants for sizes, timestamp timing.
// The second route next to OpenCL (rt.h): for phones that block OpenCL for apps (Pixel) or do better on Vulkan.
#pragma once
#include <functional>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vk {

#define VKC(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s failed: %d (%s:%d)\n", #x, (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

struct Features {
    bool fp16 = false;        // 16-bit arithmetic in shaders (shaderFloat16)
    bool storage16 = false;   // 16-bit values in buffers (storageBuffer16BitAccess)
    bool int8 = false;        // 8-bit values in buffers and arithmetic (storageBuffer8BitAccess + shaderInt8)
    bool dot8 = false;        // the 4 x 8-bit dot product instruction (VK_KHR_shader_integer_dot_product)
    bool dot8fast = false;    // ... and the driver says it is accelerated, not emulated
    bool coopmat = false;     // matrix-multiply hardware (VK_KHR_cooperative_matrix)
    uint32_t subgroup = 0;    // threads that run in lockstep
    uint32_t sharedBytes = 0; // shared memory per work group
};

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkQueryPool qpool = VK_NULL_HANDLE;
    VkPipelineCache pcache = VK_NULL_HANDLE;   // built shaders, kept on disk between runs (KCACHE/vk_pipelines.bin)
    std::string pcachePath;
    bool pcacheDirty = false;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    Features f;
    std::vector<std::string> exts;
};
static Ctx C;

static bool hasExt(const std::vector<VkExtensionProperties> &all, const char *n) {
    for (auto &e : all) if (!strcmp(e.extensionName, n)) return true;
    return false;
}

// device: index among the devices that have a compute queue (VKDEV, default 0)
static void init(int devIndex = -1) {
    if (devIndex < 0) devIndex = getenv("VKDEV") ? atoi(getenv("VKDEV")) : 0;
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.pApplicationName = "ugpu"; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VKC(vkCreateInstance(&ici, nullptr, &C.inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(C.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(C.inst, &n, pds.data());
    int k = 0;
    for (auto pd : pds) {
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qs.data());
        for (uint32_t i = 0; i < nq; i++)
            if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qs[i].timestampValidBits) {
                if (k++ == devIndex) { C.pd = pd; C.qf = i; }
                break;
            }
    }
    if (!C.pd) { fprintf(stderr, "no Vulkan device %d with a compute queue\n", devIndex); exit(1); }
    vkGetPhysicalDeviceProperties(C.pd, &C.props);
    vkGetPhysicalDeviceMemoryProperties(C.pd, &C.mem);

    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(C.pd, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> all(ne); vkEnumerateDeviceExtensionProperties(C.pd, nullptr, &ne, all.data());

    // what the chip offers (Vulkan 1.2 core structures, plus the two extensions when present)
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceShaderIntegerDotProductFeatures fdot{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11; f11.pNext = &f12;
    void **tail = &f12.pNext;
    const bool hDot = hasExt(all, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    const bool hCm = hasExt(all, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (hDot) { *tail = &fdot; tail = &fdot.pNext; }
    if (hCm) { *tail = &fcm; tail = &fcm.pNext; }
    vkGetPhysicalDeviceFeatures2(C.pd, &f2);
    C.f.fp16 = f12.shaderFloat16;
    C.f.storage16 = f11.storageBuffer16BitAccess;
    C.f.int8 = f12.shaderInt8 && f12.storageBuffer8BitAccess;
    C.f.dot8 = hDot && fdot.shaderIntegerDotProduct;
    C.f.coopmat = hCm && fcm.cooperativeMatrix;
    if (C.f.dot8) {
        VkPhysicalDeviceShaderIntegerDotProductProperties pdot{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &pdot;
        vkGetPhysicalDeviceProperties2(C.pd, &p2);
        C.f.dot8fast = pdot.integerDotProduct4x8BitPackedSignedAccelerated;
    }
    {
        VkPhysicalDeviceSubgroupProperties ps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &ps;
        vkGetPhysicalDeviceProperties2(C.pd, &p2);
        C.f.subgroup = ps.subgroupSize;
    }
    C.f.sharedBytes = C.props.limits.maxComputeSharedMemorySize;

    // switch on only what is there
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceShaderIntegerDotProductFeatures edot{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    e11.pNext = &e12;
    e11.storageBuffer16BitAccess = C.f.storage16;
    e12.shaderFloat16 = C.f.fp16;
    e12.shaderInt8 = f12.shaderInt8; e12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
    e12.hostQueryReset = f12.hostQueryReset;
    tail = &e12.pNext;
    std::vector<const char *> en;
    if (C.f.dot8) { edot.shaderIntegerDotProduct = VK_TRUE; *tail = &edot; tail = &edot.pNext; en.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME); }
    if (C.f.coopmat) { ecm.cooperativeMatrix = VK_TRUE; *tail = &ecm; tail = &ecm.pNext; en.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); }
    for (auto e : en) C.exts.push_back(e);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = C.qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &e11; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)en.size(); dci.ppEnabledExtensionNames = en.data();
    VKC(vkCreateDevice(C.pd, &dci, nullptr, &C.dev));
    vkGetDeviceQueue(C.dev, C.qf, 0, &C.q);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pci.queueFamilyIndex = C.qf;
    VKC(vkCreateCommandPool(C.dev, &pci, nullptr, &C.pool));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
    VKC(vkCreateQueryPool(C.dev, &qpi, nullptr, &C.qpool));
    // the driver's own record of built shaders: building them took 6.5 s per run on the Adreno 840. The driver checks
    // the header (vendor, device, driver version) and ignores a file from another driver.
    std::vector<char> blob;
    if (const char *d = getenv("KCACHE")) {
        C.pcachePath = std::string(d) + "/vk_pipelines.bin";
        if (FILE *f = fopen(C.pcachePath.c_str(), "rb")) {
            fseek(f, 0, SEEK_END); blob.resize(ftell(f)); fseek(f, 0, SEEK_SET);
            if (fread(blob.data(), 1, blob.size(), f) != blob.size()) blob.clear();
            fclose(f);
        }
    }
    VkPipelineCacheCreateInfo pci2{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pci2.initialDataSize = blob.size(); pci2.pInitialData = blob.empty() ? nullptr : blob.data();
    if (vkCreatePipelineCache(C.dev, &pci2, nullptr, &C.pcache) != VK_SUCCESS) {
        pci2.initialDataSize = 0; pci2.pInitialData = nullptr;
        VKC(vkCreatePipelineCache(C.dev, &pci2, nullptr, &C.pcache));
    }
}
// writes the built-shader record back when something new was built (atomically: a temporary file, then rename)
static void savePipelines() {
    if (!C.pcacheDirty || C.pcachePath.empty()) return;
    size_t n = 0;
    if (vkGetPipelineCacheData(C.dev, C.pcache, &n, nullptr) != VK_SUCCESS || !n) return;
    std::vector<char> b(n);
    if (vkGetPipelineCacheData(C.dev, C.pcache, &n, b.data()) != VK_SUCCESS) return;
    const std::string tmp = C.pcachePath + ".tmp";
    if (FILE *f = fopen(tmp.c_str(), "wb")) {
        const bool ok = fwrite(b.data(), 1, n, f) == n;
        fclose(f);
        if (ok) rename(tmp.c_str(), C.pcachePath.c_str());
    }
    C.pcacheDirty = false;
}

static void describe() {
    printf("Vulkan device: %s (driver %u.%u.%u), subgroup %u, shared %u KB\n", C.props.deviceName,
           VK_VERSION_MAJOR(C.props.driverVersion), VK_VERSION_MINOR(C.props.driverVersion), VK_VERSION_PATCH(C.props.driverVersion),
           C.f.subgroup, C.f.sharedBytes / 1024);
    printf("  16-bit maths %s, 16-bit buffers %s, 8-bit %s, 8-bit dot product %s%s, matrix hardware %s\n",
           C.f.fp16 ? "yes" : "no", C.f.storage16 ? "yes" : "no", C.f.int8 ? "yes" : "no", C.f.dot8 ? "yes" : "no",
           C.f.dot8 ? (C.f.dot8fast ? " (accelerated)" : " (emulated)") : "", C.f.coopmat ? "yes" : "no");
    if (C.f.coopmat) {
        auto get = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(C.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        uint32_t n = 0; get(C.pd, &n, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> ps(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        get(C.pd, &n, ps.data());
        auto tn = [](VkComponentTypeKHR t) {
            switch (t) { case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16"; case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
                case VK_COMPONENT_TYPE_SINT8_KHR: return "s8"; case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
                case VK_COMPONENT_TYPE_SINT32_KHR: return "s32"; case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
                case VK_COMPONENT_TYPE_SINT16_KHR: return "s16"; default: return "other"; } };
        for (auto &p : ps)
            printf("  matrix unit: %ux%u x %ux%u  A %s B %s C %s result %s  scope %s\n", p.MSize, p.KSize, p.KSize, p.NSize,
                   tn(p.AType), tn(p.BType), tn(p.CType), tn(p.ResultType), p.scope == VK_SCOPE_SUBGROUP_KHR ? "subgroup" : "other");
    }
}

// ---- buffers: device memory the processor can also map (phones: all of it; desktops with resizable BAR) ----
struct Buf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; void *map = nullptr; size_t size = 0; };

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < C.mem.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (C.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static Buf buf(size_t size) {
    Buf r; r.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VKC(vkCreateBuffer(C.dev, &bi, nullptr, &r.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(C.dev, r.b, &mr);
    uint32_t t = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (t == UINT32_MAX) t = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (t == UINT32_MAX) { fprintf(stderr, "no mappable memory\n"); exit(1); }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = t;
    VKC(vkAllocateMemory(C.dev, &mai, nullptr, &r.m));
    VKC(vkBindBufferMemory(C.dev, r.b, r.m, 0));
    VKC(vkMapMemory(C.dev, r.m, 0, size, 0, &r.map));
    return r;
}
static void freeBuf(Buf &b) {
    if (b.map) vkUnmapMemory(C.dev, b.m);
    vkDestroyBuffer(C.dev, b.b, nullptr); vkFreeMemory(C.dev, b.m, nullptr);
    b = Buf{};
}

// ---- pipelines ----
struct Pipe {
    VkShaderModule sm = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline p = VK_NULL_HANDLE;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    uint32_t nbuf = 0, pc = 0;
};

// spec: the shader's specialization constants 0, 1, 2 ... in order (uint32 each)
static bool pipe(Pipe &P, const uint32_t *spirv, size_t bytes, uint32_t nbuf, uint32_t pcBytes, const std::vector<uint32_t> &spec) {
    P.nbuf = nbuf; P.pc = pcBytes;
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = bytes; smi.pCode = spirv;
    VKC(vkCreateShaderModule(C.dev, &smi, nullptr, &P.sm));
    std::vector<VkDescriptorSetLayoutBinding> bs(nbuf);
    for (uint32_t i = 0; i < nbuf; i++) {
        bs[i] = {}; bs[i].binding = i; bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = nbuf; dli.pBindings = bs.data();
    VKC(vkCreateDescriptorSetLayout(C.dev, &dli, nullptr, &P.dsl));
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &P.dsl;
    pli.pushConstantRangeCount = pcBytes ? 1 : 0; pli.pPushConstantRanges = &pr;
    VKC(vkCreatePipelineLayout(C.dev, &pli, nullptr, &P.pl));
    std::vector<VkSpecializationMapEntry> me(spec.size());
    for (size_t i = 0; i < spec.size(); i++) me[i] = {(uint32_t)i, (uint32_t)(i * 4), 4};
    VkSpecializationInfo si{(uint32_t)me.size(), me.data(), spec.size() * 4, spec.data()};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = P.sm; cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    cpi.layout = P.pl;
    // a tile setting the chip cannot build (too many registers, too much shared memory) is skipped by the tuner
    if (vkCreateComputePipelines(C.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &P.p) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nbuf};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VKC(vkCreateDescriptorPool(C.dev, &dpi, nullptr, &P.dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = P.dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &P.dsl;
    VKC(vkAllocateDescriptorSets(C.dev, &dai, &P.ds));
    return true;
}
static void freePipe(Pipe &P) {
    if (P.p) vkDestroyPipeline(C.dev, P.p, nullptr);
    if (P.dp) vkDestroyDescriptorPool(C.dev, P.dp, nullptr);
    vkDestroyPipelineLayout(C.dev, P.pl, nullptr);
    vkDestroyDescriptorSetLayout(C.dev, P.dsl, nullptr);
    vkDestroyShaderModule(C.dev, P.sm, nullptr);
    P = Pipe{};
}
static void bind(Pipe &P, const std::vector<Buf *> &bufs) {
    std::vector<VkDescriptorBufferInfo> bi(bufs.size());
    std::vector<VkWriteDescriptorSet> w(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) {
        bi[i] = {bufs[i]->b, 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = P.ds; w[i].dstBinding = (uint32_t)i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(C.dev, (uint32_t)w.size(), w.data(), 0, nullptr);
}

// runs P reps times back to back (a barrier between runs), returns the chip's milliseconds per run
static double run(Pipe &P, const void *pc, uint32_t gx, uint32_t gy, uint32_t gz, int reps = 1) {
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = C.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cb; VKC(vkAllocateCommandBuffers(C.dev, &cai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, C.qpool, 0, 2);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pl, 0, 1, &P.ds, 0, nullptr);
    if (P.pc) vkCmdPushConstants(cb, P.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, P.pc, pc);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, C.qpool, 0);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (int r = 0; r < reps; r++) {
        if (r) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        vkCmdDispatch(cb, gx, gy, gz);
    }
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, C.qpool, 1);
    VKC(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; VKC(vkCreateFence(C.dev, &fi, nullptr, &fence));
    VKC(vkQueueSubmit(C.q, 1, &si, fence));
    VKC(vkWaitForFences(C.dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull));
    vkDestroyFence(C.dev, fence, nullptr);
    uint64_t ts[2] = {0, 0};
    vkGetQueryPoolResults(C.dev, C.qpool, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    vkFreeCommandBuffers(C.dev, C.pool, 1, &cb);
    return (double)(ts[1] - ts[0]) * C.props.limits.timestampPeriod / 1e6 / reps;
}

// ---- graphs: a whole model step recorded once as one command buffer and re-run every step ----
// A step is hundreds of small dispatches; submitting them one by one costs more than the work on a phone. A Graph
// holds the dispatch list (each with its own descriptor set, so the buffers can differ per dispatch), records it once
// with a barrier after every dispatch, and is then submitted as a whole. Buffer contents (inputs, per-step values)
// change between runs; the recording does not. PROF=1 records a timestamp after every dispatch for per-tag times.
struct Ref { Buf *b; size_t off = 0, range = 0; };   // range 0: to the end of the buffer
static Ref R(Buf &b, size_t off = 0, size_t range = 0) { return Ref{&b, off, range}; }

struct Kern {   // a pipeline without its own descriptor set (a Graph gives each dispatch one)
    VkShaderModule sm = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline p = VK_NULL_HANDLE;
    uint32_t nbuf = 0, pc = 0;
    std::string name;
};
static bool kern(Kern &K, const char *name, const uint32_t *spirv, size_t bytes, uint32_t nbuf, uint32_t pcBytes,
                 const std::vector<uint32_t> &spec) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = bytes; smi.pCode = spirv;
    VKC(vkCreateShaderModule(C.dev, &smi, nullptr, &K.sm));
    std::vector<VkDescriptorSetLayoutBinding> bs(nbuf);
    for (uint32_t i = 0; i < nbuf; i++) {
        bs[i] = {}; bs[i].binding = i; bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = nbuf; dli.pBindings = bs.data();
    VKC(vkCreateDescriptorSetLayout(C.dev, &dli, nullptr, &K.dsl));
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &K.dsl;
    pli.pushConstantRangeCount = pcBytes ? 1 : 0; pli.pPushConstantRanges = &pr;
    VKC(vkCreatePipelineLayout(C.dev, &pli, nullptr, &K.pl));
    std::vector<VkSpecializationMapEntry> me(spec.size());
    for (size_t i = 0; i < spec.size(); i++) me[i] = {(uint32_t)i, (uint32_t)(i * 4), 4};
    VkSpecializationInfo si{(uint32_t)me.size(), me.data(), spec.size() * 4, spec.data()};
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = K.sm; cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    cpi.layout = K.pl;
    K.nbuf = nbuf; K.pc = pcBytes; K.name = name;
    C.pcacheDirty = true;
    return vkCreateComputePipelines(C.dev, C.pcache, 1, &cpi, nullptr, &K.p) == VK_SUCCESS;
}

struct Graph {
    struct Op { Kern *k; VkDescriptorSet ds; std::vector<uint8_t> pc; uint32_t g[3]; std::string tag; };
    std::vector<Op> ops;
    std::vector<VkDescriptorPool> pools;
    uint32_t left = 0;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    bool prof = false;
    VkDescriptorSet alloc(Kern &k) {
        if (!left) {
            VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 * 12};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpi.maxSets = 256; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
            VkDescriptorPool dp; VKC(vkCreateDescriptorPool(C.dev, &dpi, nullptr, &dp));
            pools.push_back(dp); left = 256;
        }
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pools.back(); dai.descriptorSetCount = 1; dai.pSetLayouts = &k.dsl;
        VkDescriptorSet ds; VKC(vkAllocateDescriptorSets(C.dev, &dai, &ds));
        left--;
        return ds;
    }
    // one dispatch: buffers in binding order, push constants (any struct), work groups
    template <class PC>
    void add(Kern &k, const std::vector<Ref> &bufs, const PC &pc, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1, const char *tag = nullptr) {
        if (bufs.size() != k.nbuf) { fprintf(stderr, "%s: %zu buffers given, %u wanted\n", k.name.c_str(), bufs.size(), k.nbuf); exit(1); }
        if (sizeof(PC) > k.pc) { fprintf(stderr, "%s: push constants %zu > %u\n", k.name.c_str(), sizeof(PC), k.pc); exit(1); }
        Op o{&k, alloc(k), std::vector<uint8_t>(k.pc, 0), {gx, gy, gz}, tag ? tag : k.name};
        memcpy(o.pc.data(), &pc, sizeof(PC));
        std::vector<VkDescriptorBufferInfo> bi(bufs.size());
        std::vector<VkWriteDescriptorSet> w(bufs.size());
        for (size_t i = 0; i < bufs.size(); i++) {
            const Ref &r = bufs[i];
            bi[i] = {r.b->b, r.off, r.range ? r.range : VK_WHOLE_SIZE};
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstSet = o.ds; w[i].dstBinding = (uint32_t)i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(C.dev, (uint32_t)w.size(), w.data(), 0, nullptr);
        if (gx && gy && gz) ops.push_back(std::move(o));
    }
    // The list is recorded as several command buffers of `chunk` dispatches (VKCHUNK, default 48): one buffer for a
    // whole 2-second step made the phone's driver declare the chip hung (device lost) every few runs. The barrier after
    // every dispatch also orders work across the buffers, since they go to one queue in order.
    std::vector<VkCommandBuffer> cbs;
    // Segments (streamed runs): mark() starts a new one at the next dispatch; segCb[i] is segment i's first command
    // buffer (plus the end). A run without marks is one segment.
    std::vector<size_t> marks, segCb;
    void mark() { marks.push_back(ops.size()); }
    size_t segs() const { return segCb.size() - 1; }
    void record() {
        prof = getenv("PROF") != nullptr;
        const size_t chunk = getenv("VKCHUNK") ? (size_t)atoi(getenv("VKCHUNK")) : 48;
        if (prof) {
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = (uint32_t)ops.size() + 1;
            VKC(vkCreateQueryPool(C.dev, &qpi, nullptr, &qp));
        }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        std::vector<size_t> bounds = {0};
        for (size_t m : marks) bounds.push_back(std::max(bounds.back(), m));   // empty segments stay: segment i = mark i
        bounds.push_back(ops.size());
        for (size_t s = 0; s + 1 < bounds.size(); s++) {
            segCb.push_back(cbs.size());
            for (size_t c0 = bounds[s]; c0 < bounds[s + 1] || c0 == bounds[s]; c0 += chunk) {
                VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cai.commandPool = C.pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
                VkCommandBuffer b; VKC(vkAllocateCommandBuffers(C.dev, &cai, &b));
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VKC(vkBeginCommandBuffer(b, &bi));
                if (prof && s == 0 && c0 == 0) { vkCmdResetQueryPool(b, qp, 0, (uint32_t)ops.size() + 1); vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0); }
                Kern *bound = nullptr;
                for (size_t i = c0; i < std::min(bounds[s + 1], c0 + chunk); i++) {
                    Op &o = ops[i];
                    if (o.k != bound) { vkCmdBindPipeline(b, VK_PIPELINE_BIND_POINT_COMPUTE, o.k->p); bound = o.k; }
                    vkCmdBindDescriptorSets(b, VK_PIPELINE_BIND_POINT_COMPUTE, o.k->pl, 0, 1, &o.ds, 0, nullptr);
                    if (o.k->pc) vkCmdPushConstants(b, o.k->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, o.k->pc, o.pc.data());
                    vkCmdDispatch(b, o.g[0], o.g[1], o.g[2]);
                    vkCmdPipelineBarrier(b, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
                    if (prof) vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, (uint32_t)i + 1);
                }
                VKC(vkEndCommandBuffer(b));
                cbs.push_back(b);
                if (bounds[s + 1] == bounds[s]) break;
            }
        }
        segCb.push_back(cbs.size());
        cb = cbs[0];
    }
    void submit(size_t first, size_t end, VkFence f) {   // command buffers [first, end), the fence on the last
        for (size_t i = first; i < end; i++) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &cbs[i];
            VKC(vkQueueSubmit(C.q, 1, &si, i + 1 == end ? f : VK_NULL_HANDLE));
        }
    }
    static void waitFence(VkFence f) {
        const VkResult r = vkWaitForFences(C.dev, 1, &f, VK_TRUE, 120ull * 1000000000ull);
        if (r != VK_SUCCESS) { fprintf(stderr, "graph did not finish: %d\n", (int)r); exit(1); }
        VKC(vkResetFences(C.dev, 1, &f));
    }
    void collect(std::vector<std::pair<std::string, double>> *times) {
        if (!prof || !times) return;
        std::vector<uint64_t> ts(ops.size() + 1);
        vkGetQueryPoolResults(C.dev, qp, 0, (uint32_t)ts.size(), ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        for (size_t i = 0; i < ops.size(); i++) {
            const double ms = (double)(ts[i + 1] - ts[i]) * C.props.limits.timestampPeriod / 1e6;
            bool f = false;
            for (auto &t : *times) if (t.first == ops[i].tag) { t.second += ms; f = true; break; }
            if (!f) times->push_back({ops[i].tag, ms});
        }
    }
    // runs the recorded step and waits; with PROF=1 adds each dispatch's time to times[tag]
    void run(std::vector<std::pair<std::string, double>> *times = nullptr) {
        if (!cb) record();
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence; VKC(vkCreateFence(C.dev, &fi, nullptr, &fence));
        submit(0, cbs.size(), fence);
        waitFence(fence);
        vkDestroyFence(C.dev, fence, nullptr);
        collect(times);
    }
    // A streamed run: segment i reads its weights from slot i % 2, which fill(i) loads. fill(i + 1) runs on the
    // processor while segment i is on the chip, once segment i - 1 (the slot's last user) is done; segment i + 1 is
    // queued right behind, so the chip does not wait as long as a fill is quicker than a segment. With an even number
    // of segments the next run's segment 0 is loaded during the last one. ready0 (slot 0 holds segment 0's weights) is
    // shared by every graph, since graphs that stream the same blocks share the slots (Qwen's prompt pass and step).
    VkFence sf[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    static inline bool ready0 = false;
    static inline size_t stalls = 0;   // segments that finished before the next one's weights were in
    void runStreamed(const std::function<void(size_t)> &fill, std::vector<std::pair<std::string, double>> *times = nullptr) {
        if (!cb) record();
        const size_t n = segs();
        if (n < 2) { fill(0); run(times); return; }
        if (!sf[0]) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VKC(vkCreateFence(C.dev, &fi, nullptr, &sf[0])); VKC(vkCreateFence(C.dev, &fi, nullptr, &sf[1]));
        }
        if (!ready0) fill(0);
        submit(segCb[0], segCb[1], sf[0]);
        for (size_t i = 0; i < n; i++) {
            if (i >= 1) waitFence(sf[(i - 1) % 2]);
            if (i + 1 < n) {
                fill(i + 1);
                if (vkGetFenceStatus(C.dev, sf[i % 2]) == VK_SUCCESS) stalls++;   // the chip ran dry before this fill was done
                submit(segCb[i + 1], segCb[i + 2], sf[(i + 1) % 2]);
            } else if (n % 2 == 0) fill(0);
        }
        waitFence(sf[(n - 1) % 2]);
        ready0 = n % 2 == 0;
        collect(times);
    }
};

}  // namespace vk
