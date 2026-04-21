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

// E2M1 绝对值到 FP32 查表 (LUT)
static const uint32_t e2m1_to_fp32_lut[8] = {
    0x00000000, // 000 -> 0.0
    0x3F000000, // 001 -> 0.5
    0x3F800000, // 010 -> 1.0
    0x3FC00000, // 011 -> 1.5
    0x40000000, // 100 -> 2.0
    0x40400000, // 101 -> 3.0
    0x40800000, // 110 -> 4.0
    0x40C00000  // 111 -> 6.0
};

// 纯粹比较绝对值的阈值判定电路
uint32_t float32_to_e2m1(uint32_t f32_val) {
    uint32_t sign = (f32_val >> 31) & 0x1;
    uint32_t exp  = (f32_val >> 23) & 0xFF;
    
    // 【雷区防线】中英文注释：拦截 NaN。规格书规定 E2M1 没有 NaN，遇到 NaN 必须饱和到 6.0
    if (exp == 0xFF && (f32_val & 0x7FFFFF) != 0) {
        return (sign << 3) | 0x7; // 返回带符号的 6.0 (二进制 111)
    }

    // 将 uint32_t 的二进制流安全地剥离符号，转为 float 以利用宿主机的 FPU 进行快速阈值比较
    uint32_t abs_f32_val = f32_val & 0x7FFFFFFF;
    float val;
    memcpy(&val, &abs_f32_val, sizeof(float)); // 绝对安全的类型双关 (Type Punning)

    uint8_t mag = 0;
    
    // 中英文注释：RNE (四舍六入五成双) 的中点判断梯子
    if (val >= 5.0f)       { mag = 0x7; } // 饱和到 6.0
    else if (val >= 3.5f)  { mag = 0x6; } // 归入 4.0
    else if (val >= 2.5f)  { mag = 0x5; } // 归入 3.0
    else if (val >= 1.75f) { mag = 0x4; } // 归入 2.0
    else if (val >= 1.25f) { mag = 0x3; } // 归入 1.5
    else if (val >= 0.75f) { mag = 0x2; } // 归入 1.0
    else if (val >= 0.25f) { mag = 0x1; } // 归入 0.5
    else                   { mag = 0x0; } // 归入 0.0

    return (sign << 3) | mag;
}

// 中英文注释：核心量化器 - 将 FP32 物理降维为 FP8 (E4M3/E5M2)
// Core Quantizer - Physical dimension reduction from FP32 to FP8
uint8_t float32_to_fp8(uint32_t f32_val, bool is_e4m3) {
    uint32_t sign = (f32_val >> 31) & 0x1;
    uint32_t exp  = (f32_val >> 23) & 0xFF;
    uint32_t mant = f32_val & 0x7FFFFF;

    // 架构参数硬件跳线 (Architecture Param Jumpers)
    int bias_diff = is_e4m3 ? (127 - 7) : (127 - 15);
    int mant_keep = is_e4m3 ? 3 : 2;
    int max_exp   = is_e4m3 ? 15 : 30;

    // 防线 1：拦截 Inf/NaN (极其重要，测例必考)
    if (exp == 0xFF) {
        // E4M3 规范：Inf/NaN 全局饱和到最大值 448 (0x7E)
        // E5M2 规范：保持 Inf 状态 (0x78)
        return is_e4m3 ? ((sign << 7) | 0x7E) : ((sign << 7) | 0x78 | (mant ? 1 : 0));
    }
    if (exp == 0 && mant == 0) return sign << 7; // 纯零短路

    // 补齐隐藏位，还原绝对物理阵型
    uint32_t m = mant | 0x800000; 
    int target_exp = exp - bias_diff;
    int shift_dist = 23 - mant_keep;

    // 防线 2 & 4：非规格化处理 (Subnormals)
    // 如果指数跌穿地板，就拿右移尾数来凑
    if (target_exp <= 0) {
        shift_dist += (1 - target_exp);
        target_exp = 0;
    }

    if (shift_dist >= 25) return sign << 7; // 跌穿了物理极限，彻底归零

    // 防线 5：硬件级 RNE 舍入逻辑 (无分支进位)
    uint32_t trunc_mask = (1 << shift_dist) - 1;
    uint32_t trunc_val = m & trunc_mask;
    uint32_t half_way = 1 << (shift_dist - 1);
    
    // 判决门电路：大于中点，或者正好等于中点且保留最低位为奇数
    bool round_up = (trunc_val > half_way) || (trunc_val == half_way && ((m >> shift_dist) & 1));
    
    m >>= shift_dist;    // 暴力斩断
    if (round_up) m += 1; // 注入舍入补偿

    // 进位溢出抢救 (例如尾数从 111 进位变成了 1000，必须进位到指数)
    if (m >= (1 << (mant_keep + 1))) {
        m >>= 1;
        target_exp += 1;
    }

    m &= (1 << mant_keep) - 1; // 掩码掉隐藏位，只留纯尾数

    // 防线 3：上限饱和钳位 (Saturation)
    if (target_exp > max_exp || (is_e4m3 && target_exp == 15 && m == 7)) {
        // 踩中 E4M3 保留的 1111_111 坑位，或者超出了最大物理指数
        if (is_e4m3) return (sign << 7) | 0x7E; // E4M3 钳位到 448
        else return (sign << 7) | 0x7B; // E5M2 钳位到最大有限值 (0x7B)
    }

    // 最终物理组装
    return (sign << 7) | (target_exp << mant_keep) | m;
}

// 将 FP8 的非标偏移量还原为 FP32 标准
uint32_t fp8_to_float32(uint8_t fp8_val, bool is_e4m3) {
    uint32_t sign = (fp8_val >> 7) & 1;
    int mant_keep = is_e4m3 ? 3 : 2;
    uint32_t exp = (fp8_val >> mant_keep) & (is_e4m3 ? 0xF : 0x1F);
    uint32_t mant = fp8_val & ((1 << mant_keep) - 1);

    if (exp == 0 && mant == 0) return sign << 31; // 纯 0
    
    // E5M2 的 Inf/NaN 还原
    if (!is_e4m3 && exp == 0x1F) {
        return (sign << 31) | 0x7F800000 | (mant << 21); 
    }

    int bias_diff = is_e4m3 ? (127 - 7) : (127 - 15);
    int target_exp = exp;
    
    // 还原 Subnormals (把向右挤掉的隐藏位移回来)
    if (target_exp == 0) {
        while ((mant & (1 << mant_keep)) == 0) {
            mant <<= 1;
            target_exp--;
        }
        mant &= ~(1 << mant_keep); // 剥离隐藏位
        target_exp++;
    }
    
    target_exp += bias_diff;
    uint32_t shifted_mant = mant << (23 - mant_keep);

    return (sign << 31) | (target_exp << 23) | shifted_mant;
}

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
        case 0x53: //53?
            switch(funct7) {
                case 0x68: // FCVT.S.W
                    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                        if (!(warp->active_mask & (1 << i))) continue;
                        warp->lanes[i].fpr[rd] = int32_to_float32((int32_t)warp->lanes[i].gpr[rs1], 
                                            &warp->lanes[i].fp_status);
                    }
                    break;

                case 0x08: // FMUL.S
                    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                        if (!(warp->active_mask & (1 << i))) continue;
                        warp->lanes[i].fpr[rd] = float32_mul(warp->lanes[i].fpr[rs1], 
                                                             warp->lanes[i].fpr[rs2], 
                                                             &warp->lanes[i].fp_status);
                    }
                    break;

                case 0x00: //fadd.s
                    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                        if (!(warp->active_mask & (1 << i))) continue;
                        warp->lanes[i].fpr[rd] = float32_add(warp->lanes[i].fpr[rs1], 
                                                             warp->lanes[i].fpr[rs2], 
                                                             &warp->lanes[i].fp_status);
                        }
                    break;

                case 0x60: //fcvt.w.s
                    for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                            if (!(warp->active_mask & (1 << i))) continue;
                            if(rd != 0) {
                                warp->lanes[i].gpr[rd] = float32_to_int32_round_to_zero(warp->lanes[i].fpr[rs1], 
                                            &warp->lanes[i].fp_status);
                            }
                        }
                    break;
                
                case 0x22:
                    switch(rs2) {
                        case 0:
                            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                if (!(warp->active_mask & (1 << i))) continue;
                                uint16_t bf16 = warp->lanes[i].fpr[rs1] & 0xFFFF;
                                warp->lanes[i].fpr[rd] = bf16 << 16; // BF16 和 fp32的特殊关系
                            }
                            break;
                        case 1: //fp32 BF16
                            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                if (!(warp->active_mask & (1 << i))) continue;
                                uint32_t tmp = warp->lanes[i].fpr[rs1];
                                uint32_t rounding_bias = ((tmp >> 16) & 1) + 0x7FFF;
                                uint16_t bf16_val = (tmp + rounding_bias) >> 16;
                                // 高 16 位 NaN-Boxing
                                warp->lanes[i].fpr[rd] = 0xFFFF0000 | bf16_val;
                            }
                            break;
                        default:
                            break;
                    }
                    break;
                case 0x24:
                    switch(rs2) {
                            case 0: // FCVT.S.E4M3 (E4M3 -> FP32)
                                for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                    if (!(warp->active_mask & (1 << i))) continue;
                                    warp->lanes[i].fpr[rd] = fp8_to_float32(warp->lanes[i].fpr[rs1] & 0xFF, true);
                                }
                                break;
                            case 1: // FCVT.E4M3.S (FP32 -> E4M3)
                                for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                    if (!(warp->active_mask & (1 << i))) continue;
                                    uint32_t src_val = warp->lanes[i].fpr[rs1];
                                    warp->lanes[i].fpr[rd] = 0xFFFFFF00 | float32_to_fp8(src_val, true);
                                }   
                                break;
                            case 2: // FCVT.S.E5M2 (E5M2 -> FP32)
                                for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                    if (!(warp->active_mask & (1 << i))) continue;
                                    uint32_t src_val = warp->lanes[i].fpr[rs1];
                                    warp->lanes[i].fpr[rd] = fp8_to_float32(src_val & 0xFF, false);
                                }
                                break;
                            case 3: // FCVT.E5M2.S (FP32 -> E5M2)
                                for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                    if (!(warp->active_mask & (1 << i))) continue;
                                    uint32_t src_val = warp->lanes[i].fpr[rs1];
                                    warp->lanes[i].fpr[rd] = 0xFFFFFF00 | float32_to_fp8(src_val, false);
                                }
                                break;
                            default:
                                break;  
                        }
                    break;
                case 0x26:
                    switch(rs2) {
                        case 0: //E2M1 → FP32
                            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                if (!(warp->active_mask & (1 << i))) continue;
                                // funct5 == 0x13, rs2 == 0 (假设的 E2M1 -> FP32 路由)
                                uint32_t e2m1_val = warp->lanes[i].fpr[rs1] & 0xF; // 截取低 4 位
                                uint32_t sign = (e2m1_val >> 3) & 0x1;             // 提取第 3 位符号
                                uint32_t mag = e2m1_val & 0x7;                     // 提取低 3 位绝对值索引
                                warp->lanes[i].fpr[rd] = e2m1_to_fp32_lut[mag] | (sign << 31);
                            }
                            break;
                        case 1:
                            for (int i = 0; i < GPGPU_WARP_SIZE; i++) {                
                                if (!(warp->active_mask & (1 << i))) continue;
                                warp->lanes[i].fpr[rd] = float32_to_e2m1(warp->lanes[i].fpr[rs1]);              
                            }
                            break;
                    }
            }
            break;
        default:
            return -1; // 遇到完全不认识的 opcode，直接报非法指令
    }
    return 0;
}