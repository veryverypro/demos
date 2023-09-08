import decord
import librosa
import numpy as np
import cv2
import argparse

# 从视频中随机选择一个批次的图像和音频数据，并将其转换为Mel Spectrogram
def process_video(video_path, batch_size):
    # 使用Decord从视频中读取图像和音频数据
    av_reader = decord.AVReader(video_path)
    num_frames = len(av_reader)

    # 从视频中随机选择一个批次的帧索引
    # frame_indices = np.random.choice(num_frames, size=batch_size, replace=False)
    
    # 确保 batch_size 不超过视频帧数
    batch_size = min(batch_size, num_frames)

    # 计算连续帧的起始索引
    start_index = np.random.randint(num_frames - batch_size + 1)

    # 获取连续帧的图像和音频数据
    audio_samples, frames = av_reader.get_batch(range(start_index, start_index + batch_size))

    # 获取选择的帧的图像和音频数据
    # audio_samples, frames = av_reader.get_batch(frame_indices)

    # 定义音频处理参数
    sample_rate = 44100  # 音频的采样率
    n_fft = 1024  # FFT点数
    hop_length = 512  # STFT的跳跃长度
    n_mels = 128  # Mel频道数

    # 将音频样本转换为Mel Spectrogram
    mel_specs = []
    for audio_sample in audio_samples:
        audio_sample = np.asarray(audio_sample.asnumpy(), dtype=np.float32)
        audio_sample = librosa.util.normalize(audio_sample)
        mel_spec = librosa.feature.melspectrogram(y=audio_sample, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_specs.append(mel_spec)

    # 将Mel Spectrogram转换为对数刻度
    mel_specs_db = [librosa.power_to_db(mel_spec, ref=np.max) for mel_spec in mel_specs]

    # 调整图像的大小以匹配Mel Spectrogram的大小
    resized_mel_specs = [cv2.resize(mel_spec, (frame.shape[1], frame.shape[0])) for frame, mel_spec in zip(frames.asnumpy(), mel_specs_db)]

    # 将图像和音频的Mel Spectrogram纵向拼接
    combined_data = [np.concatenate((frame.astype(np.uint8), mel_spec.astype(np.uint8)), axis=0) for frame, mel_spec in zip(frames.asnumpy(), resized_mel_specs)]

    return combined_data

import imageio

def create_output_video(combined_data, output_path):
    # 获取新视频的宽度和高度
    height, width, _ = combined_data[0].shape

    # 创建视频写入器
    writer = imageio.get_writer(output_path, fps=30)

    # 将纵向拼接的数据写入视频
    for data in combined_data:
        writer.append_data(data)

    # 关闭视频写入器
    writer.close()


# 主函数
def main(video_path, batch_size, output_path):
    
    # 处理视频并获取纵向拼接的数据
    combined_data = process_video(video_path, batch_size)

    # 创建新的视频并将数据写入其中
    create_output_video(combined_data, output_path)

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Process video and create output video with vertically stacked data.')
    parser.add_argument('--video_path', type=str, default='girl.mp4', help='Path to the video file')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for selecting frames')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to the output video file')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.video_path, args.batch_size, args.output_path)
