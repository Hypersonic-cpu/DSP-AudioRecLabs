import pyaudio
import wave
import time
import os
import threading
import re

class FluentRecorder:
    def __init__(self, base_folder="data/digits"):
        """
        初始化录音器。
        :param base_folder: 用于保存录音文件的根目录。
        """
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 1
        self.fs = 44100
        self.p = pyaudio.PyAudio()
        self.base_folder = base_folder
        
        # 创建保存目录
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
            print(f"已创建目录: {self.base_folder}")

    def _countdown_and_record(self, duration=3):
        """
        执行倒计时并返回录音数据流。
        """
        # This method is deprecated in favor of Enter-controlled recording.
        raise RuntimeError("Countdown recording is disabled. Use Enter-controlled recording.")

    def _record_until_enter(self):
        """
        Start recording immediately and stop when the user presses Enter.
        Returns the raw frames (bytes).
        """
        # 打开音频流
        stream = self.p.open(format=self.sample_format,
                             channels=self.channels,
                             rate=self.fs,
                             frames_per_buffer=self.chunk,
                             input=True)

        frames = []
        stop_event = threading.Event()

        def record_loop():
            while not stop_event.is_set():
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                except Exception:
                    # In case of occasional overflow, skip that chunk
                    continue
                frames.append(data)

        t = threading.Thread(target=record_loop, daemon=True)
        t.start()

        # Wait for user to press Enter to stop
        input()
        stop_event.set()
        t.join()

        stream.stop_stream()
        stream.close()

        return b''.join(frames)

    def _save_audio(self, filename, frames):
        """
        将音频帧保存到WAV文件。
        """
        filepath = os.path.join(self.base_folder, filename)
        folder = os.path.dirname(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(frames)
        wf.close()
        print(f"✓ 已保存: {filepath}\n" + "---")

    def _perform_recording_session(self, item_list, item_type, repeats, duration):
        """
        一个通用的执行录音会话的内部方法。
        :param item_list: 要录制的内容列表 (例如: [0, 1, 2] 或 ["张三", "李四"])
        :param item_type: 录制类型的名称 (例如: "数字" 或 "姓名")
        :param repeats: 每个内容的重复次数
        :param duration: 每次录音的时长
        """
        total_items = len(item_list)
        for i, item in enumerate(item_list):
            print(f"\n>>> 准备录制 {item_type}: {item} ({i+1}/{total_items}) <<<")
            
            
            # 如果是数字类型，查找已有文件以继续编号，避免覆盖
            if item_type == "数字":
                subdir = os.path.join(self.base_folder, str(item))
                try:
                    existing = os.listdir(subdir)
                except Exception:
                    existing = []

                max_idx = 0
                pattern = re.compile(rf"^{re.escape(str(item))}_(\d{{3}})\.wav$")
                for fn in existing:
                    m = pattern.match(fn)
                    if m:
                        try:
                            idx = int(m.group(1))
                            if idx > max_idx:
                                max_idx = idx
                        except ValueError:
                            continue

                start_index = max_idx + 1
                if start_index > 999:
                    print(f"已达到文件编号上限 999，对于数字 {item} 无法继续保存，跳过此数字。")
                    continue

                # 进行录制，编号从 start_index 开始
                # 按要求：用户按一次 Enter 启动该数字的连续录制（该组），之后每一遍结束后
                # 立即开始下一遍，无需再次按 Enter；两组之间仍需按 Enter 开始下一组。
                if start_index > 999:
                    print(f"已达到文件编号上限 999，对于数字 {item} 无法继续保存，跳过此数字。")
                    continue

                print(f"准备连续录制数字 {item}，共 {repeats} 遍。按 Enter 开始本组第一遍，之后每遍结束后自动开始下一遍。\n每遍录音时请按 Enter 停止当前一遍。")
                input()

                for r in range(repeats):
                    cur_idx = start_index + r
                    if cur_idx > 999:
                        print(f"达到最大编号 999，中止为数字 {item} 的后续录制。")
                        break

                    print(f"第 {r+1}/{repeats} 遍 (保存为 {item}_{cur_idx:03d}.wav) — 录音中，按 Enter 停止当前一遍。")

                    # 立即开始录音，按 Enter 停止当前一遍
                    recorded_frames = self._record_until_enter()

                    filename = os.path.join(str(item), f"{item}_{cur_idx:03d}.wav")
                    self._save_audio(filename, recorded_frames)

                    if r < repeats - 1:
                        print("即将开始下一遍...")
                        time.sleep(0.1)
            else:
                # 非数字类型保持原行为（不过姓名录制已在主流程被移除）
                for repeat in range(repeats):
                    print(f"第 {repeat+1}/{repeats} 遍")
                    filename = f"{item}_{repeat+1:03d}.wav"

                    print("按 Enter 开始录音...")
                    input()
                    print("录音中...按 Enter 停止。")

                    recorded_frames = self._record_until_enter()
                    self._save_audio(filename, recorded_frames)

                    if repeat < repeats - 1:
                        time.sleep(0.5)

            if i < total_items - 1:
                print(f"{item_type} '{item}' 录制完成！稍作休息...")
                time.sleep(2) # 在不同项目之间提供更长的休息时间

    def record_numbers(self, repeats=3, duration=3):
        """
        引导用户录制从0到9的数字。
        """
        print("\n" + "="*20 + " 开始录制数字 0-9 " + "="*20)
        numbers = list(range(10))
        self._perform_recording_session(numbers, "数字", repeats, duration)

    def record_names(self, names, repeats=3, duration=3):
        """
        引导用户录制一个姓名列表。
        """
        print("\n" + "="*22 + " 开始录制姓名 " + "="*22)
        self._perform_recording_session(names, "姓名", repeats, duration)

    def close(self):
        """
        终止PyAudio会话，释放资源。
        """
        self.p.terminate()
        print("录音设备已关闭。")

# --- 主程序 ---
if __name__ == "__main__":
    # --- 参数设置 ---
    REPEATS = 25     # 每个项目录制几遍
    DURATION = 2     # 每次录音时长（秒）
    NAMES = [] # 请修改为实际需要录制的姓名
    SAVE_FOLDER = "data/digits" # 所有录音保存的文件夹

    recorder = FluentRecorder(base_folder=SAVE_FOLDER)
    
    try:
        print("=" * 60)
        print("            欢迎使用自动录音程序")
        print(f"  设置: 每个项目录制 {REPEATS} 遍, 每遍时长 {DURATION} 秒")
        print(f"  文件将保存在 '{os.path.abspath(SAVE_FOLDER)}' 目录中")
        print("=" * 60)
        
        # --- 录制数字 ---
        input("准备好后，请按回车键开始录制【数字 0-9】...")
        recorder.record_numbers(repeats=REPEATS, duration=DURATION)
        
        print("\n" + "*"*60)
        print("恭喜！数字部分已全部录制完成！")
        
        # 不录制姓名，结束程序
        print("\n" + "*"*60)
        print("🎉 录音任务已完成（已跳过姓名录制）。")
        
    except KeyboardInterrupt:
        print("\n程序被用户手动中断。")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
    finally:
        recorder.close()
