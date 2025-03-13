import os
import shutil

def copy_ham_file(id1, id2, source_base_path, dest_path):
    """
    复制指定的ham文件到目标路径
    
    Args:
        id1 (str): 第一级目录ID
        id2 (str): 目标文件名(不含扩展名)
        source_base_path (str): 源文件基础路径
        dest_path (str): 目标路径
    """
    try:
        # 构建完整的源文件路径
        source_file = os.path.join(source_base_path, str(id1), 'hadms', f'{id2}.jsonl')
        
        # 确保源文件存在
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"源文件不存在: {source_file}")
            
        # 确保目标目录存在
        os.makedirs(dest_path, exist_ok=True)
        
        # 构建目标文件路径
        dest_file = os.path.join(dest_path, f'{id2}.jsonl')
        
        # 复制文件
        shutil.copy2(source_file, dest_file)
    except Exception as e:
        print(f"复制文件失败: {e}")

path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/my_MIMIC_IV/patients_sorted"
dest_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/bohao_mimic_project/SFT/find"

copy_ham_file(100000, 100000, path, dest_path)