import os

def clean_directory(target_dir):
    if not os.path.isdir(target_dir):
        print(f"❌ 目录不存在: {target_dir}")
        return

    deleted = 0
    total = 0

    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        if os.path.isfile(file_path):
            total += 1
            # 检查文件名是否以 cat 或 dog 开头（不区分大小写）
            if not (filename.lower().startswith('cat') or filename.lower().startswith('dog')):
                os.remove(file_path)
                deleted += 1

    print(f"\n✅ 完成清理：共扫描 {total} 个文件，删除 {deleted} 个无效文件。")

if __name__ == "__main__":
    # 修改为你想清理的目标路径
    target_folder = "data/test"  # 替换成你的路径，比如 "./images"
    clean_directory(target_folder)
