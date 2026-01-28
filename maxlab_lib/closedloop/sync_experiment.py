#!/usr/bin/env python3

"""
SYNC EXPERIMENT SCRIPT

此脚本演示如何：
1. 启动C++闭环游戏程序并同步运行
2. 配置MaxLab硬件和数据采集
3. 在合适的时机同步启动数据流和游戏循环
"""

import subprocess
import os
import sys
import time
import threading
import maxlab as mx

# ============================================================================
# 配置部分
# ============================================================================

# C++可执行文件路径
CPP_EXECUTABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/maxone_with_filter"
)

# C++程序参数
CPP_ARGS = [
    "0",          # target_well
    "5",          # window_ms
    "8000",       # blanking_frames
    "1",          # show_gui (1=显示GUI)
    "20000",      # sample_rate_hz
    "5.0",        # threshold_multiplier
    "-20",        # min_threshold
    "1000",       # refractory_samples
    "1024",       # channel_count
    "1"           # wait_for_sync (1=等待同步)
]

# 实验配置
WELLS = [0]
RECORDING_DIR = "/tmp/experiments"
RECORDING_NAME = "closed_loop_test"
RECORDING_DURATION = 60  # 秒

# 电极配置（根据实际需求修改）
ELECTRODES = [i for i in range(1024)]  # 记录所有1024个通道

# ============================================================================
# C++进程管理
# ============================================================================

class CPPProcessManager:
    """管理C++游戏进程的类"""

    def __init__(self, executable, args):
        self.executable = executable
        self.args = args
        self.process = None
        self.output_thread = None
        self.running = False
        self.ready_event = threading.Event()

    def start(self):
        """启动C++进程，使用stdin/stdout管道"""
        print(f"[C++] Starting: {self.executable}")
        print(f"[C++] Arguments: {' '.join(self.args)}")

        # 创建进程，配置stdin/stdout/stderr
        self.process = subprocess.Popen(
            [self.executable] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 启动后台线程持续读取输出
        self.running = True
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()

        # 等待进程启动并等待同步信号
        self._wait_for_ready()

    def _read_output(self):
        """后台线程持续读取C++输出"""
        ready_marker = "[SYNC] Waiting for start signal"
        while self.running and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    print(f"[C++ OUT] {line.strip()}")
                    if ready_marker in line and not self.ready_event.is_set():
                        self.ready_event.set()
            except Exception as e:
                break

    def _wait_for_ready(self):
        """等待C++进程启动并进入等待状态"""
        print("[C++] Waiting for process to be ready...")

        ready_marker = "[SYNC] Waiting for start signal"
        timeout = 10  # 最多等待10秒
        start_time = time.time()

        while time.time() - start_time < timeout and not self.ready_event.is_set():
            time.sleep(0.1)

        if self.ready_event.is_set():
            print("[C++] Process is ready and waiting for sync signal")
        else:
            print(f"[C++] Warning: Did not see ready marker within {timeout}s")
            print("[C++] Continuing anyway, assuming process is ready")

    def send_start_signal(self):
        """向C++进程发送启动信号"""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("C++ process is not running")

        print("[C++] Sending 'start' signal...")
        try:
            # 写入"start\n"并刷新缓冲区
            self.process.stdin.write("start\n")
            self.process.stdin.flush()

            print("[C++] Start signal sent successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to send start signal: {e}")

    def stop(self):
        """停止C++进程"""
        if self.process is None:
            return

        print("[C++] Stopping process...")

        # 发送终止信号
        import signal
        self.process.send_signal(signal.SIGINT)  # Ctrl+C

        # 等待进程结束（最多5秒）
        try:
            self.process.wait(timeout=5)
            print(f"[C++] Process exited with code {self.process.returncode}")
        except subprocess.TimeoutExpired:
            print("[C++] Process did not exit gracefully, force killing...")
            self.process.kill()
            self.process.wait()
            print("[C++] Process was killed")

        self.running = False
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=1)

        self.process = None

# ============================================================================
# MaxLab采集配置
# ============================================================================

def configure_maxlab():
    """配置MaxLab硬件和采集参数"""
    print("\n[MX] Initializing MaxLab system...")

    # 初始化
    mx.initialize()
    mx.send(mx.Core().enable_stimulation_power(True))
    time.sleep(mx.Timing.waitInit)

    # 激活wells
    mx.activate(WELLS)
    print(f"[MX] Activated wells: {WELLS}")

    # 创建电极阵列
    array = mx.Array("experiment")
    array.select_electrodes(ELECTRODES)
    array.route()
    array.download()
    time.sleep(mx.Timing.waitAfterDownload)

    # Offset补偿
    mx.offset()
    time.sleep(mx.Timing.waitInMX2Offset)
    mx.clear_events()

    print("[MX] Configuration complete")

# ============================================================================
# 主实验流程
# ============================================================================

def run_experiment():
    """运行完整实验"""
    print("=" * 60)
    print("CLOSED LOOP SYNCHRONIZED EXPERIMENT")
    print("=" * 60)

    cpp_manager = None
    try:
        # 步骤1：启动C++游戏进程
        print("\n=== STEP 1: Starting C++ Game Process ===")
        cpp_manager = CPPProcessManager(CPP_EXECUTABLE, CPP_ARGS)
        cpp_manager.start()

        # 步骤2：配置MaxLab系统
        print("\n=== STEP 2: Configuring MaxLab ===")
        configure_maxlab()

        # 步骤3：准备数据采集
        print("\n=== STEP 3: Preparing Data Recording ===")
        s = mx.Saving()
        s.open_directory(RECORDING_DIR)
        s.start_file(RECORDING_NAME)
        s.group_define(0, "all_channels", list(range(1024)))
        print(f"[MX] Recording will be saved to: {RECORDING_DIR}/{RECORDING_NAME}")

        # 步骤4：同步启动
        print("\n=== STEP 4: Synchronized Start ===")
        print("[SYNC] Sending start signal to C++ game...")
        cpp_manager.send_start_signal()

        # 等待C++稳定
        print("[SYNC] Waiting 2 seconds for data stream to stabilize...")
        time.sleep(2)

        print("[MX] Starting data recording...")
        s.start_recording(WELLS)

        # 步骤5：运行实验
        print(f"\n=== STEP 5: Running Experiment ({RECORDING_DURATION}s) ===")
        print("[MX] Data recording and game loop are running...")
        print("[INFO] Press Ctrl+C to stop early")

        start_time = time.time()
        while time.time() - start_time < RECORDING_DURATION:
            # 可选：监控C++进程状态
            if cpp_manager.process.poll() is not None:
                print("[C++] Warning: C++ process exited early!")
                break

            time.sleep(1)
            elapsed = int(time.time() - start_time)
            remaining = RECORDING_DURATION - elapsed
            print(f"\r[INFO] Elapsed: {elapsed}s / {RECORDING_DURATION}s (remaining: {remaining}s)", end="")

        print("\n[INFO] Experiment duration completed")

        # 步骤6：停止采集
        print("\n=== STEP 6: Stopping Recording ===")
        s.stop_recording()
        time.sleep(mx.Timing.waitAfterRecording)
        s.stop_file()
        s.group_delete_all()
        print(f"[MX] Recording saved successfully")

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[INFO] Experiment interrupted by user")
        print("[INFO] Cleaning up...")

    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # 清理资源
        print("\n[INFO] Cleaning up resources...")

        # 停止C++进程
        if cpp_manager is not None:
            cpp_manager.stop()

        print("[INFO] Cleanup complete")

    return 0

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 检查C++可执行文件是否存在
    if not os.path.exists(CPP_EXECUTABLE):
        print(f"ERROR: C++ executable not found: {CPP_EXECUTABLE}")
        print("Please build the C++ program first:")
        print("  cd maxlab_lib && make maxone_with_filter")
        sys.exit(1)

    # 运行实验
    exit_code = run_experiment()
    sys.exit(exit_code)
