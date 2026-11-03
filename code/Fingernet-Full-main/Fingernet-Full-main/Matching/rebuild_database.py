#!/usr/bin/env python3
"""
数据库重建脚本
用于删除现有数据库并重建新的数据库

使用方法:
python rebuild_database.py [ICount] [JCount]

参数:
ICount - 指纹数量 (默认: 10)
JCount - 每个指纹的图片数量 (默认: 8)

示例:
python rebuild_database.py           # 重建完整数据库 (10x8=80张图片)
python rebuild_database.py 2 1      # 重建测试数据库 (2x1=2张图片)
python rebuild_database.py 5 4      # 重建中等数据库 (5x4=20张图片)
"""

import os
import sys
import time
from build_db import build_db

def rebuild_database(ICount=5, JCount=4):
    """
    删除现有数据库并重建新数据库
    
    Args:
        ICount: 指纹数量
        JCount: 每个指纹的图片数量
    """
    print("=" * 60)
    print("数据库重建工具")
    print("=" * 60)
    
    # 检查现有数据库
    if os.path.exists('db.pkl'):
        file_size = os.path.getsize('db.pkl') / 1024  # KB
        print(f"发现现有数据库: db.pkl ({file_size:.1f} KB)")
        
        # 删除现有数据库
        try:
            os.remove('db.pkl')
            print("✓ 现有数据库已删除")
        except Exception as e:
            print(f"✗ 删除数据库失败: {e}")
            return False
    else:
        print("未发现现有数据库文件")
    
    # 重建数据库
    print(f"\n开始重建数据库...")
    print(f"参数: {ICount} 个指纹, 每个 {JCount} 张图片")
    print(f"预计处理图片数量: {ICount * JCount} 张")
    
    start_time = time.time()
    
    try:
        build_db(ICount, JCount)
        end_time = time.time()
        
        # 检查新数据库
        if os.path.exists('db.pkl'):
            new_file_size = os.path.getsize('db.pkl') / 1024  # KB
            print(f"\n✓ 数据库重建成功!")
            print(f"新数据库大小: {new_file_size:.1f} KB")
            print(f"构建时间: {end_time - start_time:.2f} 秒")
            return True
        else:
            print("\n✗ 数据库重建失败: 未生成数据库文件")
            return False
            
    except Exception as e:
        print(f"\n✗ 数据库重建失败: {e}")
        return False

def main():
    """主函数"""
    # 解析命令行参数
    ICount = 5  # 默认值
    JCount = 4   # 默认值
    
    if len(sys.argv) >= 2:
        try:
            ICount = int(sys.argv[1])
        except ValueError:
            print("错误: ICount 必须是整数")
            sys.exit(1)
    
    if len(sys.argv) >= 3:
        try:
            JCount = int(sys.argv[2])
        except ValueError:
            print("错误: JCount 必须是整数")
            sys.exit(1)
    
    # 参数验证
    if ICount <= 0 or JCount <= 0:
        print("错误: ICount 和 JCount 必须大于 0")
        sys.exit(1)
    
    if ICount > 10:
        print("警告: ICount > 10 可能超出可用图片范围")
    
    if JCount > 8:
        print("警告: JCount > 8 可能超出可用图片范围")
    
    # 执行重建
    success = rebuild_database(ICount, JCount)
    
    if success:
        print("\n数据库重建完成! 可以开始进行匹配测试。")
        sys.exit(0)
    else:
        print("\n数据库重建失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()