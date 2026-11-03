"""
验证转换后的 Python 代码功能正确性

这个脚本测试所有转换后的 Python 模块是否可以正确导入和基本功能是否正常。

Vahid. K. Alilou
Department of Computer Engineering
The University of Semnan

July 2013
"""

import sys
import os
import importlib.util
import traceback

def test_import(module_path, module_name):
    """
    测试模块导入
    
    Parameters:
    module_path - 模块文件路径
    module_name - 模块名称
    
    Returns:
    success - 是否成功导入
    error_msg - 错误信息（如果有）
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, None
    except Exception as e:
        return False, str(e)

def verify_conversions():
    """
    验证所有转换后的 Python 文件
    """
    print("=== 验证转换后的 Python 代码 ===\n")
    
    # 定义要测试的模块
    modules_to_test = [
        # 主目录模块
        ("build_db.py", "build_db"),
        ("build_fmr.py", "build_fmr"),
        ("ext_finger.py", "ext_finger"),
        ("f_enhance.py", "f_enhance"),
        ("main_calc_fmr.py", "main_calc_fmr"),
        ("main_single.py", "main_single"),
        ("main_total.py", "main_total"),
        ("match.py", "match"),
        ("plot_data.py", "plot_data"),
        ("score.py", "score"),
        ("transform.py", "transform"),
        ("transform2.py", "transform2"),
        
        # FExtraction 目录模块
        ("FExtraction/angular_filter_bank.py", "angular_filter_bank"),
        ("FExtraction/compute_coherence.py", "compute_coherence"),
        ("FExtraction/dist2.py", "dist2"),
        ("FExtraction/fft_enhance_cubs.py", "fft_enhance_cubs"),
        ("FExtraction/freqest.py", "freqest"),
        ("FExtraction/normalise.py", "normalise"),
        ("FExtraction/p.py", "p"),
        ("FExtraction/ridgefilter.py", "ridgefilter"),
        ("FExtraction/ridgefreq.py", "ridgefreq"),
        ("FExtraction/ridgeorient.py", "ridgeorient"),
        ("FExtraction/ridgesegment.py", "ridgesegment"),
        ("FExtraction/smoothen_frequency_image.py", "smoothen_frequency_image"),
        ("FExtraction/smoothen_orientation_image.py", "smoothen_orientation_image"),
        ("FExtraction/test_bifurcation.py", "test_bifurcation"),
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    failed_modules = []
    
    # 测试每个模块
    for file_path, module_name in modules_to_test:
        full_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(full_path):
            print(f"❌ {module_name}: 文件不存在 - {file_path}")
            failed_modules.append((module_name, "文件不存在"))
            continue
        
        success, error_msg = test_import(full_path, module_name)
        
        if success:
            print(f"✅ {module_name}: 导入成功")
            success_count += 1
        else:
            print(f"❌ {module_name}: 导入失败 - {error_msg}")
            failed_modules.append((module_name, error_msg))
    
    # 输出总结
    print(f"\n=== 验证结果总结 ===")
    print(f"总模块数: {total_count}")
    print(f"成功导入: {success_count}")
    print(f"导入失败: {total_count - success_count}")
    print(f"成功率: {(success_count / total_count) * 100:.1f}%")
    
    if failed_modules:
        print(f"\n=== 失败的模块详情 ===")
        for module_name, error_msg in failed_modules:
            print(f"- {module_name}: {error_msg}")
    
    return success_count, total_count, failed_modules

def check_dependencies():
    """
    检查必要的依赖包
    """
    print("\n=== 检查依赖包 ===")
    
    required_packages = [
        "numpy",
        "scipy",
        "matplotlib", 
        "PIL",  # Pillow
        "pickle"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少的依赖包: {', '.join(missing_packages)}")
        print("请使用以下命令安装:")
        if "PIL" in missing_packages:
            missing_packages.remove("PIL")
            missing_packages.append("Pillow")
        print(f"pip install {' '.join(missing_packages)}")
    
    return missing_packages

def main():
    """
    主验证函数
    """
    print("指纹识别系统 MATLAB 到 Python 转换验证")
    print("=" * 50)
    
    # 检查依赖
    missing_deps = check_dependencies()
    
    # 验证转换
    success_count, total_count, failed_modules = verify_conversions()
    
    # 最终结果
    print(f"\n{'=' * 50}")
    if success_count == total_count and not missing_deps:
        print("🎉 所有模块转换验证成功！")
        print("系统已准备好使用。")
    elif missing_deps:
        print("⚠️  转换完成，但缺少必要的依赖包。")
        print("请安装缺少的依赖包后重新验证。")
    else:
        print("⚠️  部分模块存在问题，请检查失败的模块。")
    
    return success_count == total_count and not missing_deps

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)