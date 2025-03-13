import pandas as pd

# # Step 1: 读取第一个CSV文件 (没有表头)
# table1_path = '/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/process_code/item_set/procedures_icd.csv'  # 第一个CSV文件路径
# table1 = pd.read_csv(table1_path, header=None, names=['icd_code', 'icd_version', 'stat_value'])

# # Step 2: 读取第二个CSV文件 (有表头)
# table2_path = '/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/supplementary_files/hosp/d_icd_procedures.csv'  # 第二个CSV文件路径
# table2 = pd.read_csv(table2_path)

# # Step 3: 合并两个表以匹配 ICD 代码和版本
# merged_table = pd.merge(table1, table2[['icd_code', 'icd_version', 'long_title']], 
#                         on=['icd_code', 'icd_version'], 
#                         how='left')

# # Step 4: 将合并结果保存回第一个CSV文件的位置，并写入表头
# merged_table.to_csv(table1_path, index=False, header=True, encoding='utf-8-sig')


# import pandas as pd

# # Step 1: 读取第一个CSV文件（没有表头）
# table1_path = '/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/process_code/item_set/hcpcsevents.csv'  # 第一个CSV文件路径
# table1 = pd.read_csv(table1_path, header=None, names=['code', 'stat_value'])

# # Step 2: 读取第二个CSV文件（有表头）
# table2_path = '/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/supplementary_files/hosp/d_hcpcs.csv'  # 第二个CSV文件路径
# table2 = pd.read_csv(table2_path)

# # Step 3: 合并两个表以匹配 item_id
# merged_table = pd.merge(table1, table2[['code', 'short_description']], 
#                         on='code', 
#                         how='left')

# # Step 4: 将合并结果保存回第一个CSV文件的位置，并写入表头
# merged_table.to_csv(table1_path, index=False, header=True, encoding='utf-8-sig')

import pandas as pd

# Step 1: 读取第一个CSV文件（没有表头）
table1_path = '/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/process_code/item_set/icu_events.csv'  # 第一个CSV文件路径
table1 = pd.read_csv(table1_path, header=None, names=['itemid', 'stat_value'])

# Step 2: 读取第二个CSV文件（有表头）
table2_path = '/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/supplementary_files/icu/d_items.csv'  # 第二个CSV文件路径
table2 = pd.read_csv(table2_path)

# Step 3: 合并两个表以匹配 item_id
merged_table = pd.merge(table1, table2[list(table2.keys())], 
                        on='itemid', 
                        how='left')

# Step 4: 将合并结果保存回第一个CSV文件的位置，并写入表头
merged_table.to_csv(table1_path, index=False, header=True, encoding='utf-8-sig')