# 解压 MIMIC-IV 数据集中的 .gz 文件
import os
import gzip
import shutil

# 数据路径
root_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/download-mimc/physionet.org/files/mimiciv/2.2/"
# root_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/physionet.org/mimic-iv-ed-2.2/"
# root_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/physionet.org/mimic-iv-note-deidentified-free-text-clinical-notes"
output_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/decompressed_MIMIC_IV/"

def decompress_gz_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".gz"):
                gz_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_dir_path = os.path.join(output_dir, relative_path)
                
                # 创建对应的子目录
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)
                
                csv_file_path = os.path.join(output_dir_path, file[:-3])  # 去掉 ".gz"
                
                # 解压文件
                with gzip.open(gz_file_path, 'rb') as f_in:
                    with open(csv_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f"解压完成: {gz_file_path} -> {csv_file_path}")

# 解压 hosp 和 icu 文件夹
decompress_gz_files(os.path.join(root_path, "hosp"), os.path.join(output_path, "hosp"))
decompress_gz_files(os.path.join(root_path, "icu"), os.path.join(output_path, "icu"))
# decompress_gz_files(os.path.join(root_path, "ed"), os.path.join(output_path, "ed"))
# decompress_gz_files(os.path.join(root_path, "note"), os.path.join(output_path, "note"))
