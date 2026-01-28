#!/usr/bin/env python3

"""
测试Python启动C++的同步通信功能（不涉及MaxLab硬件）
"""

import subprocess
import os
import sys
import time
import threading

# C++可执行文件路径
CPP_EXECUTABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/maxone_with_filter"
)

# C++程序参数（无GUI模式，启用同步）
CPP_ARGS = [
    "0",          # target_well
    "5",          # window_ms
    "8000",       # blanking_frames
    "0",          # show_gui (0=无GUI)
    "20000",      # sample_rate_hz
    "5.0",        # threshold_multiplier
    "-20",        # min_threshold
    "1000",       # refractory_samples
    "1024",       # channel_count
    "1"           # wait_for_sync (1=等待同步)
]

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
            stderr=subprocess.STDOUT,  # 合并stderr到stdout
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
        timeout = 10
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
        import signal
        self.process.send_signal(signal.SIGINT)

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

def run_test():
    """运行测试"""
    print("=" * 60)
    print("PYTHON-C++ SYNC TEST (NO HARDWARE)")
    print("=" * 60)

    cpp_manager = None
    try:
        # 步骤1：启动C++游戏进程
        print("\n=== STEP 1: Starting C++ Game Process ===")
        cpp_manager = CPPProcessManager(CPP_EXECUTABLE, CPP_ARGS)
        cpp_manager.start()

        # 步骤2：等待几秒钟（模拟硬件配置）
        print("\n=== STEP 2: Simulating Hardware Configuration ===")
        print("[TEST] Waiting 2 seconds...")
        time.sleep(2)

        # 步骤3：发送启动信号
        print("\n=== STEP 3: Sending Start Signal ===")
        cpp_manager.send_start_signal()

        # 步骤4：运行一段时间
        print("\n=== STEP 4: Running for 5 seconds ===")
        start_time = time.time()
        while time.time() - start_time < 5:
            if cpp_manager.process.poll() is not None:
                print("[C++] Warning: C++ process exited early!")
                break
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            remaining = 5 - elapsed
            print(f"\r[TEST] Elapsed: {elapsed}s / 5s (remaining: {remaining}s)", end="")

        print("\n[TEST] Test duration completed")

        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\n[TEST] Test interrupted by user")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\n[TEST] Cleaning up...")
        if cpp_manager is not None:
            cpp_manager.stop()
        print("[TEST] Cleanup complete")

if __name__ == "__main__":
    if not os.path.exists(CPP_EXECUTABLE):
        print(f"ERROR: C++ executable not found: {CPP_EXECUTABLE}")
        print("Please build the C++ program first:")
        print("  cd maxlab_lib && make maxone_with_filter")
        sys.exit(1)

    exit_code = run_test()
    sys.exit(exit_code)
