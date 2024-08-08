import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=30):
    # 获取所有 .jpg 图像文件的列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files.sort()  # 按名称排序

    # 确保文件夹中有图像
    if not image_files:
        raise ValueError(f"No .jpg files found in {image_folder}")

    # 读取第一张图像以获取尺寸
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # 也可以用 'XVID' 或其他编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 逐帧读取图像并写入视频
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        #print(image_path)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # 释放视频写入器
    video_writer.release()
    print(f"Video saved to {output_video_path}")

# 使用示例
image_folder = './mesh/'
output_video_path = 'mesh_video.mp4'
create_video_from_images(image_folder, output_video_path)
image_folder ='./full/'
output_video_path = 'full_video.mp4'
create_video_from_images(image_folder, output_video_path)
image_folder = './orig/'
output_video_path ='orig_video.mp4'
create_video_from_images(image_folder, output_video_path)
image_folder = './gt_full/'
output_video_path ='gt_full_video.mp4'
create_video_from_images(image_folder, output_video_path)
image_folder = './gt_mesh/'
output_video_path ='gt_mesh_video.mp4'
create_video_from_images(image_folder, output_video_path)
