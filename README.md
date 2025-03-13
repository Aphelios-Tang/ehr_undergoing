# compress 文件夹
## IC-Former：压缩重建部分的代码（文件名含有NUMBER为数字encoder和decoder的方法，含有number/num为另一种数字编码的方法）
- icformer目录：icformer压缩模型的核心代码，没修改。只修改了configuration.py hidden_size部分
- data_utils.py：数据集构建，加入了MIMICDataset
- DS.py: MIMIC-IV-note中有关discharge的处理代码
- finetune.py, finetune.sh, generate.py, generate.sh: 原作者代码，未修改
- modules_num.py: 
- modules.py: 原作者代码的基础上，加入evaluation和log
- my_mimic_dataset_for_reconstruction.py: 读取目录下的mimic数据集，生成free text用于压缩重建任务
- NUMBER_encoder_decoder_train.py/.sh: 数字encoder和decoder单独训练的代码，验证能够重构数字
- number_extract.py: freetext中提取数字替换占位符
- NUMBER_modules.py: 基于modules.py修改，引入有关数字encoder和decoder的模型，训练和评估
- NUMBER_pretrain_ehr.py/.sh: 训练压缩模型，含有数字encoder和decoder
- number.py: 用于数字编码的一些函数
- pretrain_ehr_num.py/.sh: 数字编码方法的训练
- pretrain_ehr.py/.sh: 用icformer在ehr数据集上压缩的训练
- pretrain.py/.sh: 原作者的代码，未修改
- qwen_vocabulary.jsonl: qwen2.5-7b-instruct模型的词表
- qwen.py: 验证qwen模型能不能理解reconstruction
- test....的py文件: 测试
- token_id.py: 输出qwen词表
- utils_ehr.py: 处理ehr数据集需要的函数
- utils.py: icformer的参数设定

## IC-Former：原作者代码未修改

# ehr_freetext 文件夹
创建不同大小的数据集

# process_code 文件夹，原始MIMIC-IV的数据处理
- single目录: 原始MIMIC-IV数据集中对应类型文件(ed. hosp, icu, note)，按照subject_id整理，合并
- decompress_MIMIC.py: 解压
- filename_combination.py: 合并四种类型的文件，按照subject_id整理
- patient_split.py: 划分数据集
- hadm_id_jsonl_combine.py: 按照hamd_id整理

# SFT
- IC目录：用于icformer和sft串起来的代码，包括icformer的模型和参数设定（未对数字处理的版本）
- item_similarity目录: chaoyi学长原先的代码和文件，未修改
- make_dataset目录: 从格式化json文件的数据集生成free_text形式的数据集的代码，for_reconstruction是构建压缩重建的数据，for_json是构建sft的freetext数据
- test_code: 测试代码，含有旧版本内容，这里面不重要
- tongji目录
  - balance_dataset目录：用于平衡数据集，确保yes和no的个数相等，针对4个单选任务
  - count_length_and_draw: 数据集统计绘图等代码
  - DS_detail.py: 把discharge的test提取出清晰的格式
  - find_long_history/py: 找history长的数据
  - find_sub_ham: 找具体的hamd_id文件
- DS.py: 涉及提取discharge所需的函数
- filter_24.py: 涉及提取入院24小时item的函数
- lab.py: 实验室结果处理，chaoyi学长原来的代码，未修改
- my_mimic_dataset_from_free_text_addcompress.py: icformer + sft的数据集代码，读取的是free_text的jsonl文件，文件从make_dataset目录下的for_json生成。在preprocess函数中修改了很多
- my_mimic_dataset_from_free_text.py: 数据集代码，读取的是free_text的jsonl文件（不含有压缩）
- my_mimic_dataset_hadm_id_cutlen.py: 数据集代码，读取的是各个目录下hadm_id的jsonl文件，先处理后生成数据
- train和test文件为对应的训练和测试

# supplementary_files 目录(文件较大，不方便上传)
chaoyi学长的原始文件，未修改

