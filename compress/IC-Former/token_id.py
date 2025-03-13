import json
import os
from transformers import AutoTokenizer

def save_vocab_to_jsonl(model_path, output_file="qwen_vocabulary.jsonl"):
    """
    从Qwen模型中提取词表并保存为jsonl文件
    
    Args:
        model_path: Qwen模型路径
        output_file: 输出文件路径
    """
    print(f"加载模型分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 获取词表
    vocab = tokenizer.get_vocab()
    print(f"词表大小: {len(vocab)}")
    
    # 将词表保存为jsonl文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, token_id in vocab.items():
            # 将每个词元和ID作为一个JSON对象写入文件
            f.write(json.dumps({"token": token, "id": token_id}, ensure_ascii=False) + '\n')
    
    print(f"词表已保存到: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    model_path = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/Qwen2.5-7B-Instruct"
    save_vocab_to_jsonl(model_path)