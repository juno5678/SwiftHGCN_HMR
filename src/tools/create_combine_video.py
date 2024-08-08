import cv2
import os

def create_side_by_side_video(orig_folder, full_folder, output_video, start_index, end_index, fps=30, image_format='jpg'):
    images = []
    
    # 构建文件名并收集所有图像路径
    for i in range(start_index, end_index + 1):
        image_name = f"downtown_rampAndStairs_00_image_{i:05d}.{image_format}"
        orig_path = os.path.join(orig_folder, image_name)
        full_path = os.path.join(full_folder, image_name)
        
        if os.path.exists(orig_path) and os.path.exists(full_path):
            images.append((orig_path, full_path))
        else:
            print(f"Warning: {orig_path} or {full_path} does not exist and will be skipped.")
    
    # 确定视频的帧大小
    if not images:
        print("No images found in the specified range.")
        return
    
    orig_frame = cv2.imread(images[0][0])
    full_frame = cv2.imread(images[0][1])
    height, width, layers = orig_frame.shape
    frame_size = (width * 2, height)
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    for orig_path, full_path in images:
        orig_frame = cv2.imread(orig_path)
        full_frame = cv2.imread(full_path)
        
        # 并排合并图像
        combined_frame = cv2.hconcat([orig_frame, full_frame])
        
        video_writer.write(combined_frame)
    
    video_writer.release()
    print(f"Video {output_video} has been created successfully.")

# 参数设置
orig_folder = 'orig'
full_folder = 'full'
output_video = 'combine_video.mp4'
start_index = 375
end_index = 975
fps = 30

# 创建视频
create_side_by_side_video(orig_folder, full_folder, output_video, start_index, end_index, fps)

