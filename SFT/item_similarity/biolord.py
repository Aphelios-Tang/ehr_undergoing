import os
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#Mean Pooling - Take attention mask into account for correct averaging
# 计算句子的平均池化嵌入向量
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# 判断文件的第一行是否包含数据
def is_first_row_data(file_path):
    # 读取文件的前两行
    first_row = pd.read_csv(file_path, nrows=1, header=None).iloc[0]
    # 检查第一行是否包含数字
    return first_row.apply(lambda x: any(char.isdigit() for char in str(x))).any()

# 根据第一行是否为数据决定是否有header，返回pandas DataFrame对象
def load_file(file_path):
    if is_first_row_data(file_path):
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path)

    # df = df.head(10)
    # print(df)
    return df


# Sentences we want sentence embeddings for
# item, str, id
DATA_LOG = [
    ["procedures_icd", "long_title", "icd_code"],
    ["diagnosis_icd", "long_title", "icd_code"],
    ["labevents", "label", "itemid"],
    ["microbiologyevents", 0, 1],
    ["emar", 0, 1]
]

input_folder = "/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/process_code/item_set"
output_folder = "/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/SFT/item_similarity"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('/mnt/hwfile/medai/liaoyusheng/projects/MIMIC-IV-LLM/models/BioLORD-2023')
model = AutoModel.from_pretrained('/mnt/hwfile/medai/liaoyusheng/projects/MIMIC-IV-LLM/models/BioLORD-2023').cuda()

for data_name, col_name_or_index, id_col_name_or_index in DATA_LOG:
    if os.path.exists(f'{output_folder}/{data_name}.csv'):
        continue

    df = load_file(f"{input_folder}/{data_name}.csv")
    df = df.drop_duplicates(subset=col_name_or_index, keep='first')
    df = df.dropna()
    item_set = df[col_name_or_index].tolist()


    # Tokenize sentences
    total_embeedings = []
    for i in range(0, len(item_set), 128):
        sub_item_set = item_set[i:i+128]
        encoded_input = tokenizer(sub_item_set, padding="max_length", truncation=True, return_tensors='pt')
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].cuda()

        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling
        item_embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu()
        total_embeedings.append(item_embedding)

    # Normalize embeddings
    total_embeedings = torch.cat(total_embeedings, 0)
    similarity_matrix = cosine_similarity(total_embeedings)
    similarity_df = pd.DataFrame(similarity_matrix, index=df[col_name_or_index], columns=df[col_name_or_index])

    # 创建一个字典来保存每个 item 的 top 100 相似项
    top_n = 100
    top_similarities = {"item": [], "item_id": df[id_col_name_or_index]}

    for item in similarity_df.index:
        # 获取当前 item 的相似度并排序
        similar_items = similarity_df[item].nlargest(top_n + 1)  # 加1是因为包含了自身
        similar_items = similar_items[similar_items.index != item]  # 去掉自身
        top_similarities["item"].append(item)
        similar_items = similar_items.index.tolist()
        for i in range(len(similar_items)):
            if f"top_{i+1}" not in top_similarities:
                top_similarities[f"top_{i+1}"] = [similar_items[i]]
            else:
                top_similarities[f"top_{i+1}"].append(similar_items[i])

    top_similarities_df = pd.DataFrame(top_similarities)
    top_similarities_df.to_csv(f'{output_folder}/{data_name}.csv', index=False)


