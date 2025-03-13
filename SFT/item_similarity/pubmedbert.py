from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Mean Pooling - Take attention mask into account for correct averaging
def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("/mnt/hwfile/medai/liaoyusheng/projects/MIMIC-IV-LLM/models/pubmedbert-base-embeddings")
model = AutoModel.from_pretrained("/mnt/hwfile/medai/liaoyusheng/projects/MIMIC-IV-LLM/models/pubmedbert-base-embeddings").cuda()

df = pd.read_csv("/mnt/petrelfs/wuchaoyi/my_MIMIC_IV_2.2/process_code/item_set/diagnosis_icd.csv")
df = df.drop_duplicates(subset='long_title', keep='first')
# df = df.head(5)
df = df.dropna()
long_title_list = df['long_title'].tolist()

inputs = tokenizer(long_title_list, padding=True, truncation=True, return_tensors='pt')
for key in inputs:
    inputs[key] = inputs[key].cuda()

# for span_end_id in range(10, len(long_title_list), 10):
#     try:
#         long_titles = long_title_list[span_end_id-10:span_end_id]
#         inputs = tokenizer(long_titles, padding=True, truncation=True, return_tensors='pt')
#     except:
#         import pdb
#         pdb.set_trace()

# Compute token embeddings
with torch.no_grad():
    output = model(**inputs)

# Perform pooling. In this case, mean pooling.
embeddings = meanpooling(output, inputs['attention_mask']).cpu()
similarity_matrix = cosine_similarity(embeddings)

similarity_df = pd.DataFrame(similarity_matrix, index=df['long_title'], columns=df['long_title'])

# 创建一个字典来保存每个 item 的 top 100 相似项
top_n = 100
top_similarities = {"item": []}

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
top_similarities_df.to_csv('top_similar_items.csv', index=False)
