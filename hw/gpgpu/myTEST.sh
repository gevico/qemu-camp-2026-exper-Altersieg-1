#!/bin/bash

# --- 路径自动定位策略 ---
# 获取脚本所在目录的绝对路径，并推导出项目根目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="/workspace/qemu-camp-2026-exper-Altersieg-1"
BUILD_DIR="$REPO_ROOT/build"

# 检查构建目录是否存在
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    exit 1
fi

# 环境变量设置
export QTEST_QEMU_BINARY="$BUILD_DIR/qemu-system-riscv64"
QOS_TEST="$BUILD_DIR/tests/qtest/qos-test"
BASE_PATH="/riscv64/virt/generic-pcihost/pci-bus-generic/pci-bus/gpgpu/gpgpu-tests"

# 定义测试项列表
tests=(
    "device-id" "vram-size" "global-ctrl" "dispatch-regs"
    "vram-access" "dma-regs" "irq-regs" "simt-thread-id"
    "simt-block-id" "simt-warp-lane" "simt-thread-mask"
    "simt-reset" "kernel-exec" "fp-kernel-exec" "lp-convert"
    "lp-convert-e5m2-e2m1" "lp-convert-saturate"
)

declare -a results

echo ">>> Project Root: $REPO_ROOT"
echo ">>> Using Binary: $QTEST_QEMU_BINARY"
echo ">>> Execution in progress..."

for test_name in "${tests[@]}"; do
    printf "Running: %-25s " "$test_name"
    
    # 执行测试，屏蔽冗余的 vhost 报错
    $QOS_TEST -p "$BASE_PATH/$test_name" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        results+=("[$test_name] | PASS")
        printf "[\033[32mOK\033[0m]\n"
    else
        results+=("[$test_name] | FAIL / CRASH")
        printf "[\033[31mFAILED\033[0m]\n"
    fi
done

# --- 最终汇总 ---
echo -e "\n======================================================"
echo "           GPGPU FINAL TEST SUMMARY                  "
echo "======================================================"
printf "%-30s | %-10s\n" "TEST CASE" "RESULT"
echo "------------------------------------------------------"

for res in "${results[@]}"; do
    IFS='|' read -r name status <<< "$res"
    if [[ "$status" == *"PASS"* ]]; then
        printf "%-30s | \033[32m%s\033[0m\n" "$name" "$status"
    else
        printf "%-30s | \033[31m%s\033[0m\n" "$name" "$status"
    fi
done
echo "======================================================"
