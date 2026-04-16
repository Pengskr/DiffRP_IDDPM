import os
from PIL import Image

def convert_png_to_jpg(folder_path):
    # 如果不存在则创建输出文件夹（可选）
    output_folder = os.path.join(folder_path, "converted_jpgs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            # 构建完整的文件路径
            img_path = os.path.join(folder_path, filename)
            
            # 打开图片
            with Image.open(img_path) as img:
                # 注意：PNG 通常有透明通道 (RGBA)，转换成 JPG 前需要转为 RGB
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # 构建新的文件名（更换后缀）
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                save_path = os.path.join(output_folder, new_filename)
                
                # 保存为 JPG
                img.save(save_path, "JPEG", quality=100)
                print(f"成功转换: {filename} -> {new_filename}")

# 使用示例
if __name__ == "__main__":
    # 替换为你存放图片的实际路径
    your_path = "./PPD/test/UNSEEN/PATH_20PIXEL" 
    convert_png_to_jpg(your_path)