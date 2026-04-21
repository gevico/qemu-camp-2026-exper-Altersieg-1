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
    warp->active_mask = 0xFFFFFFFF >> (GPGPU_WARP_SIZE - num_threads);
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    warp->num_threads = num_threads;
    warp->warp_id = warp_id;
    warp->block_id_linear = block_id_linear;

    for (int i = 0; i < num_threads; ++i) {
        warp->lanes[i].pc = pc;
        warp->lanes[i].active = (num_threads > i);
        warp->lanes[i].fp_status = 0; // 这个怎么判断？
        uint32_t mhartid = 0 | MHARTID_ENCODE(block_id_linear, warp_id, i);
        warp->lanes[i].mhartid = mhartid;
    }
}

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
            uint32_t active_thread = GPGPU_WARP_SIZE;
            uint32_t thread_id_base = j * GPGPU_WARP_SIZE;
            if ((j == (warp_num - 1)) && lane_num) {
                active_thread = lane_num;
            }
            gpgpu_core_init_warp(&warp, (uint32_t)s->kernel.kernel_addr, thread_id_base, block_id_tmp, active_thread, j, i);

            if (gpgpu_core_exec_warp(s, &warp, 100000) == -1) {
                return -1;
            }
        }
    }
    return 0;
}

/* TODO: Implement warp execution (RV32I + RV32F interpreter) */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycles = 0;
    while (cycles < max_cycles && warp->active_mask != 0) { // 0 还是 F？
        // 抓一个活跃线程，取指
        uint32_t cmd = 0;
        for (int i = 0; i < GPGPU_WARP_SIZE; ++i) {
            if (warp->lanes[i].active) { //warp->lanes[i].active 和 warp 的 mask 需要某种同步？
                cmd = warp->lanes[i].pc;
                break;
            }
        }
        cycles++;
    }

    return 0;
}