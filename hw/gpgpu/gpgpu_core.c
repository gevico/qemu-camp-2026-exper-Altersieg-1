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
//#include "qemu/bitops.h"

/* TODO: Implement warp initialization */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));

    warp->thread_id_base = thread_id_base;
    warp->active_mask = 0xFFFFFFFF >> (GPGPU_WARP_SIZE - num_threads);
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    warp->warp_id = warp_id;

    for (int i = 0; i < num_threads; ++i) {
        warp->lanes[i].pc = pc;
        warp->lanes[i].active = (num_threads > i);
        //warp->lanes[i].fp_status = 0; // 这个怎么判断？
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

        uint32_t inst = *(uint32_t*)(s->vram_ptr + cmd);
        if (exec_one_inst(s, warp, inst) != 0){
            //报错？
            return -1;
        }

        for (int i = 0; i < GPGPU_WARP_SIZE; ++i) {
            if (warp->active_mask & (1 << i)) {
                warp->lanes[i].pc += 4;
            }
        }
        cycles++;
    }
    if (cycles >= max_cycles) {
        return -1;
    }
    return 0;
}


// 单条指令执行引擎 (译码与 SIMT 锁步执行)
int exec_one_inst(GPGPUState *s, GPGPUWarp *warp, uint32_t inst)
{

    uint32_t opcode = extract32(inst, 0, 7);
    uint32_t rd     = extract32(inst, 7, 5);
    uint32_t funct3 = extract32(inst, 12, 3);
    uint32_t rs1    = extract32(inst, 15, 5);
    uint32_t rs2    = extract32(inst, 20, 5);

    // funct7 主要用于区分 R 类型指令（如 ADD 和 SUB）
    uint32_t funct7 = extract32(inst, 25, 7); 
    // I-Type 立即数 (用于 lw, addi, slli)
    int32_t imm_i = sextract32(inst, 20, 12); 
    // S-Type 立即数 (专用于 sw 存储指令，它的物理位被斩断成了两截)
    int32_t imm_s = (sextract32(inst, 25, 7) << 5) | extract32(inst, 7, 5);
    // U-Type 立即数 (用于 lui，直接提取高 20 位)
    uint32_t imm_u = extract32(inst, 12, 20) << 12;

    switch (opcode) {

        // ---------------------------------------------------------
        // ALU 立即数运算 (OP-IMM): 例如 ANDI, SLLI, ADDI
        // ---------------------------------------------------------
        case 0x13: 
            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
                if (!(warp->active_mask & (1 << i))) continue; // 物理掩码过滤
                if (rd == 0) continue; // 架构规范：x0 寄存器硬连线到 0，不可写

                uint32_t val1 = warp->lanes[i].gpr[rs1];
                
                switch (funct3) {
                    case 0x0: // ADDI
                        warp->lanes[i].gpr[rd] = val1 + imm_i;
                        break;
                    case 0x1: // SLLI (逻辑左移，注意 shamt 只有低 5 位)
                        warp->lanes[i].gpr[rd] = val1 << (imm_i & 0x1F);
                        break;
                    case 0x7: // ANDI
                        warp->lanes[i].gpr[rd] = val1 & imm_i;
                        break;
                    // ... 其他的实验暂时用不到，可以先不写
                    default: return -1;
                }
            }
            break;

        // ---------------------------------------------------------
        // ALU 寄存器运算 (OP): 例如 ADD
        // ---------------------------------------------------------
        case 0x33:
            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
                if (!(warp->active_mask & (1 << i))) continue;
                if (rd == 0) continue;

                uint32_t val1 = warp->lanes[i].gpr[rs1];
                uint32_t val2 = warp->lanes[i].gpr[rs2];

                switch (funct3) {
                    case 0x0:
                        if (funct7 == 0x00) { // ADD
                            warp->lanes[i].gpr[rd] = val1 + val2;
                        } 
                        // SUB 是 funct7 == 0x20，目前不需要
                        break;
                    default: return -1;
                }
            }
            break;

        // ---------------------------------------------------------
        // U-Type 加载高位 (LUI): 将立即数直接塞入目标寄存器的高 20 位
        // ---------------------------------------------------------
        case 0x37:
            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
                if (!(warp->active_mask & (1 << i))) continue;
                if (rd == 0) continue;
                warp->lanes[i].gpr[rd] = imm_u;
            }
            break;

        // ---------------------------------------------------------
        // 系统指令 (SYSTEM): 这里藏着 EBREAK 和你的梯子 CSRRS
        // ---------------------------------------------------------
        case 0x73:
            if (funct3 == 0x0) { // EBREAK 等环境调用
                if (extract32(inst, 20, 12) == 1) { // EBREAK 的 funct12 是 1
                    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
                        if (warp->active_mask & (1 << i)) {
                            // 遇到 ebreak，当前线程执行完毕，物理休眠！
                            warp->lanes[i].active = false;
                            warp->active_mask &= ~(1 << i);
                        }
                    }
                }
            } else if (funct3 == 0x2) { // CSRRS (用于读取 mhartid)
                uint32_t csr_addr = extract32(inst, 20, 12);
                if (csr_addr == 0xF14) { // mhartid 的硬件地址
                    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
                        if (!(warp->active_mask & (1 << i))) continue;
                        if (rd == 0) continue;
                        
                        // 中英文注释：将初始化时填好的身份铭牌，写入目标寄存器
                        // Write the identity tag filled during initialization into the destination register
                        warp->lanes[i].gpr[rd] = warp->lanes[i].mhartid;
                    }
                }
            }
            break;

        // ---------------------------------------------------------
        // 还有 SW 和 LW 没写，这里就是处理寻址和内存路由的地方！
        // ---------------------------------------------------------
        case 0x23: // STORE (SW)
            // 检查 funct3 是否为 0x2 (Word，32位写入)
            if (funct3 != 0x2) return -1; 

            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {
                if (!(warp->active_mask & (1 << i))) continue; // 物理掩码过滤

                // 中英文注释：计算目标物理地址 (Base + Offset)
                // Calculate target physical address (Base + Offset)
                uint32_t base_addr = warp->lanes[i].gpr[rs1];
                uint32_t target_addr = base_addr + imm_s;
                
                // 中英文注释：获取需要写入的数据
                // Get the data to be written
                uint32_t store_val = warp->lanes[i].gpr[rs2];

                // 中英文注释：硬件级内存保护机制 (MMU 越界检查)
                // Hardware-level memory protection mechanism (MMU bounds check)
                if (target_addr + 4 > s->vram_size) {
                    s->error_status |= GPGPU_ERR_VRAM_FAULT;
                    return -1; // 触发硬件 Fault 停机
                }

                // 中英文注释：直接通过内部总线 (指针) 写入物理显存
                // Write directly to physical VRAM via internal bus (pointer)
                *(uint32_t*)(s->vram_ptr + target_addr) = store_val;
            }
            break;
        default:
            return -1; // 遇到完全不认识的 opcode，直接报非法指令
    }

    return 0;
}