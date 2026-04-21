/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"

/* TODO: Implement warp initialization */

void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->pc = pc;
    warp->thread_id_base = thread_id_base;
    warp->block_id = block_id;
    warp->num_threads = num_threads;
    warp->warp_id = warp_id;
    warp->block_id_linear = block_id_linear;

    for (int i = 0; i < num_threads; ++i) {
        warp->thread_ids = ;
    }
}
/**
 * gpgpu_core_init_warp - 初始化一个 warp
 * @warp: warp 状态指针
 * @pc: 初始程序计数器（内核代码地址）
 * @thread_id_base: 起始线程 ID
 * @block_id: block ID 数组 [x, y, z]
 * @num_threads: 活跃线程数量 (最多 32)
 * @warp_id: warp 在 block 内的编号
 * @block_id_linear: 线性化的 block ID
 */

/* TODO: Implement kernel dispatch and execution */
int gpgpu_core_exec_kernel(GPGPUState *s)
{
    uint32_t gx = s->kernel.grid_dim[0];
    uint32_t gy = s->kernel.grid_dim[1];
    uint32_t gz = s->kernel.grid_dim[2];
    uint32_t bx = s->kernel.block_dim[0];
    uint32_t by = s->kernel.block_dim[1];
    uint32_t bz = s->kernel.block_dim[2];

    uint64_t total_blocks = gx * gy * gz; // 一个grid里的block数
    uint64_t thread_per_block = bx * by * bz; // 一个block里的thread数

    uint64_t warp_num = (thread_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE; //0
    uint64_t lane_num = thread_per_block % GPGPU_WARP_SIZE; //8

    for (int i = 0; i < total_blocks; ++i) { // 创建block，维护block ID 数组 [x, y, z]

        uint32_t bx_tmp = i % gx;
        uint32_t by_tmp = i / gx % gy;
        uint32_t bz_tmp = (i / (gx * gy)) % gz;

        const uint32_t block_id_tmp[3] = {bx_tmp, by_tmp, bz_tmp};

        for (int j = 0; j < warp_num; ++j) {
            GPGPUWarp warp;
            uint32_t active_thread_full = GPGPU_WARP_SIZE;
            gpgpu_core_init_warp(&warp, (uint32_t)s->kernel.kernel_addr, i, block_id_tmp, active_thread_full, j, i);
            if ((j == (warp_num - 1)) & lane_num) {
                gpgpu_core_init_warp(&warp, (uint32_t)s->kernel.kernel_addr, i, block_id_tmp, lane_num, j, i);
            }
        }
    }

    return 0;
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    (void)s;
    (void)warp;
    (void)max_cycles;
    return 0;
}