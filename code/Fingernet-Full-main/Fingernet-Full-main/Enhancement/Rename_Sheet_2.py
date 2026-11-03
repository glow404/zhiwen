import os
import glob
import re
import shutil

def rename_remove_enh(my_folder):
    """
    等效于 MATLAB Rename_Sheet_2.m:
    - 在 my_folder 下查找 *.bmp
    - 将文件名中匹配 '_enh\\W*' 的部分移除，保存为新文件名（同目录）
    - 若目标文件名已存在，自动添加序号避免覆盖
    """
    if not os.path.isdir(my_folder):
        raise FileNotFoundError(f"Folder not found: {my_folder}")

    pattern = os.path.join(my_folder, '*.bmp')
    files = glob.glob(pattern)
    if not files:
        print("No .bmp files found in", my_folder)
        return

    for fullpath in files:
        base = os.path.basename(fullpath)
        name, ext = os.path.splitext(base)
        # 移除 '_enh' 后跟任意非单词字符的部分（MATLAB 中为 '_enh(\W*)'）
        new_name_core = re.sub(r'(?i)_enh\W*', '', name)  # (?i) 不区分大小写
        new_base = new_name_core + ext.lower()

        dest = os.path.join(my_folder, new_base)
        # 如果目标和源相同则跳过
        if os.path.abspath(dest) == os.path.abspath(fullpath):
            print(f"Skipped (same name): {base}")
            continue

        # 若目标存在则添加序号避免覆盖
        if os.path.exists(dest):
            i = 1
            while True:
                candidate = f"{new_name_core}_{i}{ext.lower()}"
                dest = os.path.join(my_folder, candidate)
                if not os.path.exists(dest):
                    break
                i += 1

        # 复制文件（保持原文件不变），若需移动可用 os.rename
        try:
            shutil.copy2(fullpath, dest)
            print(f"Copied: {base} -> {os.path.basename(dest)}")
        except Exception as e:
            print(f"Failed to copy {base} -> {os.path.basename(dest)}: {e}")

if __name__ == '__main__':
    # 修改为你的目录
    myFolder = '/path/to/your/folder'  # <-- 替换为实际路径
    rename_remove_enh(myFolder)