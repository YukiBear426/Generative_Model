import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ==== 参数设置 ====
base_path = "/home/user6/public2/xyq/Cartoon_generation/output/Flow"  # 每个 epoch 的目录
output_gif = "./Flow_result.gif"  # 最终 GIF 输出路径
frame_duration = 200  # 每帧持续时间（ms）

# ==== 字体设置（Linux 默认字体）====
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=24)
except:
    font = ImageFont.load_default()

# ==== 图像加载与处理 ====
image_list = []

for epoch in tqdm(range(1, 101), desc="加载与标注图像"):
    epoch_dir = f"epoch_{epoch:03d}"
    image_path = os.path.join(base_path, epoch_dir, "samples.png")

    if not os.path.exists(image_path):
        print(f"⚠️ 跳过：{image_path}")
        continue

    # 打开图像并转换为RGB
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # 创建新图像（原图 + 底部文字条）
    new_img = Image.new("RGB", (width, height + 40), (0, 0, 0))
    new_img.paste(img, (0, 0))

    # 添加文字
    draw = ImageDraw.Draw(new_img)
    text = f"Epoch {epoch:03d}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_position = ((width - text_width) // 2, height + (40 - text_height) // 2)
    draw.text(text_position, text, fill=(255, 255, 255), font=font)

    image_list.append(new_img)

# ==== 保存为 GIF ====
if image_list:
    image_list[0].save(
        output_gif,
        save_all=True,
        append_images=image_list[1:],
        duration=frame_duration,
        loop=0,
        optimize=True
    )
    print(f"\n✅ GIF 动画已生成：{output_gif}（共 {len(image_list)} 帧）")
else:
    print("❌ 未找到任何图像，无法生成 GIF。")
