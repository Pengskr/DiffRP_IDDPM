import os
import numpy as np
import matplotlib.pyplot as plt
import einops
import torch as th
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, folder_a, folder_b, num_images, transform=None, threshold=0.99):
        self.dir_a = root_dir / folder_a
        self.dir_b = root_dir / folder_b
        self.transform = transform
        self.extension = ".jpg" 
        self.num_images = num_images
        self.threshold = threshold

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_name = f"{idx}{self.extension}"
        path_a = os.path.join(self.dir_a, img_name)
        path_b = os.path.join(self.dir_b, img_name)
        
        # 加载图片
        # .convert('RGB') 确保加载为3通道，如果是灰度图可改为 .convert('L')
        img_a = Image.open(path_a).convert('L')     # 0 是纯黑，255 是纯白
        img_b = Image.open(path_b).convert('L')

        # 应用预处理（如有）
        if self.transform:
            img_a = self.transform(img_a)   # ToTensor()会将L模式的PIL Image归一化
            img_b = self.transform(img_b)
        
        img_a = (img_a > self.threshold).float() * 2 - 1
        # img_b 反相：原本 > threshold 的部分变 -1，原本 <= threshold 的部分变 1
        img_b = (img_b <= self.threshold).float() * 2 - 1
        
        # 返回成对的张量
        return img_a, img_b, {}, img_name

def get_dataloader(root_folder_data, folder_Mo, folder_P, num_images, batch_size, image_size, shuffle = True):
    data_transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
    ])
    dataset = PairedImageDataset(
        root_folder_data,
        folder_Mo,
        folder_P,
        num_images=num_images,
        transform=data_transform,   # Path有反相
        threshold=0.9
    )
    if len(dataset) > 0:
        random_idx = np.random.randint(len(dataset))
        sample_a, _, _, _ = dataset[random_idx]

        img_shape = sample_a.shape
        print(f"成功加载 {len(dataset)} 对图片,单张图片张量尺寸 (C, H, W): {img_shape}")
    else:
        print("警告：数据集为空，请检查路径！")
    
    # 初始化 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader

def yield_dataloader(loader):
    while True:
        yield from loader

def show_dataloader(dataloader):
    sample_a, sample_b, _, _ = next(iter(dataloader))
    # 1. 提取第 1 对图像 (index 0)
    # 假设 sample_a 是 [batch_size, channels, height, width]
    img_a = sample_a[0].detach().cpu()
    img_b = sample_b[0].detach().cpu()

    # 2. 归一化并转换维度
    # 如果是单通道(灰度图)，squeeze 会去掉 C；如果是多通道，需要 transpose(1, 2, 0)
    def process_for_plot(tensor):
        # 归一化到 0-1 范围 (Matplotlib 对 float 类型的 0-1 支持更好)
        img = (tensor + 1) / 2
        img = img.clamp(0, 1).numpy()
        
        if img.shape[0] == 1: # 灰度图
            return img.squeeze(), 'gray'
        else: # RGB 
            return img.transpose(1, 2, 0), None

    data_a, cmap_a = process_for_plot(img_a)
    data_b, cmap_b = process_for_plot(img_b)

    # 3. 绘图
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(data_a, cmap=cmap_a)
    plt.title("MAP")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(data_b, cmap=cmap_b)
    plt.title("PATH")
    plt.axis('off')

    plt.show()

def show_samples(args, arr, M_o, P, img_name):
    # ------------------------------------ 显示推理结果 ------------------------------------
    # 1. 准备数据：将所有数据转回 torch Tensor 格式 [N, 1, H, W]
    n_display = args.num_samples
    M_o_disp = M_o[:n_display].detach().cpu()
    P_disp = P[:n_display].detach().cpu()

    # 将采样得到的 numpy 数组转回 tensor 并还原维度顺序为 [N, C, H, W]
    imgs_disp = th.from_numpy(arr[:n_display]).permute(0, 3, 1, 2).float()

    # 2. 归一化到 [0, 255]
    # M_o 和 P 原始通常在 [-1, 1]，imgs_disp 如果已经是 uint8 转回来的，就在 [0, 255]
    M_o_disp = (M_o_disp + 1) / 2 * 255
    P_disp = (P_disp + 1) / 2 * 255
    # 如果 imgs_disp 已经是 0-255 范围，则不需要再处理，否则按需归一化

    # --- 新增：给每张图加框线 (Padding) ---
    pad_width = 1  # 线条宽度（像素）
    pad_value = 127 # 线条颜色：0为黑，255为白，127为灰色

    # F.pad 参数顺序是 (左, 右, 上, 下)
    # 我们给右边和下边加 pad，这样拼接后就有分割线了
    def add_border(x):
        return th.nn.functional.pad(x, (0, pad_width, 0, pad_width), value=pad_value)

    M_o_disp = add_border(M_o_disp)
    P_disp = add_border(P_disp)
    imgs_disp = add_border(imgs_disp)

    # 3. 堆叠成三元组 [N, 3, 1, H, W]
    # 顺序：M_o (地图), P (真实路径), imgs (生成路径)
    combined = th.stack([M_o_disp, P_disp, imgs_disp], dim=1)

    # 4. 使用 einops 排布
    # 设置每行显示的样本组数 (每组包含 3 张图)
    n_groups_per_row = 4  # 每行显示 4 组路径对比
    b1 = n_display // n_groups_per_row
    b2 = n_groups_per_row

    # 排布逻辑：纵向堆叠样本 (b1*h)，横向排布 (b2组 * 每组3张 * 宽度w)
    res = einops.rearrange(combined, 
                        '(b1 b2) p c h w -> (b1 h) (b2 p w) c', 
                        b1=b1, b2=b2, p=3)

    # 5. 转换类型用于显示
    res_np = res.clamp(0, 255).numpy().astype(np.uint8)

    # 6. 直接在 Notebook 中显示
    plt.figure(figsize=(20, 10)) # 增加宽度以适应 1:3 的比例
    if res_np.shape[-1] == 1: # 灰度图处理
        plt.imshow(res_np.squeeze(), cmap='gray')
    else:
        plt.imshow(res_np)
    plt.axis('off')
    plt.title("Comparison: Map | Ground Truth | Sampled Path")
    plt.show()

    print(img_name)