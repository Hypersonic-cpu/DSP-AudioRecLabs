#!/usr/bin/env python3
"""
Simple demo: press Enter to start recording, press Enter to stop and save to `data/temp.wav`,
then run ML_all.py inference on the saved file.

Usage:
  python3 ML_demo.py

This script uses `scripts/record_audio.py`'s `FluentRecorder` for Enter-controlled recording.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
# Make sure scripts package is importable
sys.path.insert(0, os.path.join(HERE, 'scripts'))

try:
    from scripts.record_audio import FluentRecorder
except Exception as e:
    print("Failed to import FluentRecorder from scripts/record_audio.py:", e)
    print("Make sure you have required dependencies (pyaudio).")
    raise


def main():
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)

    recorder = FluentRecorder(base_folder=os.path.join(HERE, 'data'))

    print("Demo loop: press Enter to record, press Enter again to stop. Type 'q' to quit.")

    try:
        while True:
            cmd_start = input("\nReady? Press Enter to begin recording (or type 'q' then Enter to quit): ")
            if cmd_start.strip().lower() == 'q':
                print("Exiting demo.")
                break

            print("Recording... (press Enter to stop)")
            frames = recorder._record_until_enter()

            out_name = 'temp.wav'
            recorder._save_audio(out_name, frames)

            saved_path = os.path.join(HERE, 'data', out_name)
            print(f"Saved demo recording to: {saved_path}")

            # Call ML_all.py inference (use the current Python interpreter)
            model_path = 'results/mixed-5mini/ml_all_checkpoint.pt'
            cmd = [sys.executable, os.path.join(HERE, 'ML_all.py'),
                   '--mode', 'infer',
                   '--model', model_path,
                   '--file', saved_path,
                   '--n-mfcc', '15']

            print('\nRunning inference:')
            print(' '.join(cmd))

            # Run and stream output
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"Inference exited with code {proc.returncode}")

            cont = input("\nPress Enter to record again, or type 'q' then Enter to quit: ")
            if cont.strip().lower() == 'q':
                print("Exiting demo.")
                break

    except KeyboardInterrupt:
        print("\nRecording cancelled by user.")
    except Exception as e:
        print("Error during demo:", e)
    finally:
        try:
            recorder.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
