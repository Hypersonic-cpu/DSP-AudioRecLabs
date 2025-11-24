# import numpy as np
# import torch
# import os

# if __name__ == "__main__":
#     """ Env """
#     DSP_AUDIO_HOME = os.getenv('DSP_AUDIO_HOME', './dsp_project') # 默认当前目录下的dsp_project
#     DATA_ROOT = os.path.join(DSP_AUDIO_HOME, 'data')
#     OUTPUT_MFCC_DIR = os.path.join(DSP_AUDIO_HOME, 'data', 'mfcc')
#     MODEL_SAVE_PATH = os.path.join(OUTPUT_MFCC_DIR, 'templates.pt')

#     """ TRAINING """
#     from freq_domain.train import train_wrapper
#     # print(f"DSP_AUDIO_HOME: {DSP_AUDIO_HOME}")
    
#     # 检查数据源是否存在 (仅作提示)
#     # if not os.path.exists(DATA_ROOT):
#     #     print(f"Error: Data root directory does not exist: {DATA_ROOT}")
#         # 为了演示，创建一个假的目录结构
#         # os.makedirs(os.path.join(DATA_ROOT, '0'), exist_ok=True)
#         # open(os.path.join(DATA_ROOT, '0', 'demo.wav'), 'w').close()
#         # pass

#     # 调用训练 Wrapper
#     # batch_size 在此处传入，虽然内部是串行处理，但保留了接口
#     train_wrapper(
#         data_root_path=DATA_ROOT,
#         output_dir=OUTPUT_MFCC_DIR,
#         save_model_path=MODEL_SAVE_PATH,
#     )
    
#     # 下一步提示
#     print("-" * 40)
#     print("Next Step: 在 classifiers.py 中加载 templates.pt 并运行识别。")
#     print("例如: templates = torch.load('path/to/templates.pt')")

#     """ INFERENCE """
#     from freq_domain.classifiers import recognition_wrapper
#     # 简单的 dummy test
#     dummy_audio = np.random.uniform(-0.5, 0.5, 16000) # 1秒音频 (16kHz)
    
#     # 伪造一些模板 (实际应从文件加载)
#     dummy_templates = {
#         str(i): torch.randn(100, 13) for i in range(10)
#     }
    
#     print("Start Recognition Test...")
#     try:
#         result = recognition_wrapper(dummy_audio, 16000, dtw_distance, dummy_templates)
#         print(f"Recognition Result: {result}")
#     except Exception as e:
#         print(f"Error: {e}")