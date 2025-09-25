import os
import shutil

def collect_and_rename(src_root: str, dst_folder: str, src_filename: str = "3.png"):
    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)

    # 收集所有匹配的文件路径
    matches = []
    for root, dirs, files in os.walk(src_root):
        file_path = os.path.join(root, src_filename)
        if os.path.isfile(file_path):
            matches.append(file_path)

    # 可选：按子文件夹名或完整路径排序，确保命名顺序可控
    matches.sort()

    # 复制并重命名
    for idx, src_path in enumerate(matches, start=1):
        dst_path = os.path.join(dst_folder, f"{idx}.png")
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {src_path} → {dst_path}")

if __name__ == "__main__":
    # TODO: 修改为你的源目录和目标目录路径
    source_root = r"/data/llj/GenRobustify/picture/observe_2_hidden_state/suffix"
    destination = r"/data/llj/GenRobustify/picture/observe_2_hidden_state/hidden_state"

    collect_and_rename(source_root, destination)
