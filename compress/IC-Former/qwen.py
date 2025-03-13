import transformers
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

text = "Event_type: Labevent, Time: 2152-01-27 20:35:00 -> 2152-01-27 20:35:00\n\nHematocrit: 30.4 %  (40.0, 52.0)--------abnormal ()\nHemoglobin: 9.3 g/dL  (14.0, 18.0)--------abnormal ()\nMCH: 31.5 pg  (27.0, 32.0)--------normal ()\nMCHC: 30.6 %  (31.0, 35.0)--------abnormal ()\nMCV: 103.0 fL  (82.0, 98.0)--------abnormal ()\nPlatelet Count: 98.0 K/uL  (150.0, 440.0)--------abnormal ()\nRDW: 26.8 %  (10.5, 15.5)--------abnormal ()\nRed Blood Cells: 2.95 m/uL  (4.6, 6.2)--------abnormal ()\nWhite Blood Cells: 3.7 K/uL  (4.0, 11.0)--------abnormal ()\nAlanine Aminotransferase (ALT): 27.0 IU/L  (0.0, 40.0)--------normal ()\nAlkaline Phosphatase: 70.0 IU/L  (40.0, 130.0)--------normal ()\nAnion Gap: 14.0 mEq/L  (8.0, 20.0)--------normal ()\nAsparate Aminotransferase (AST): 33.0 IU/L  (0.0, 40.0)--------normal ()\nBicarbonate: 24.0 mEq/L  (22.0, 32.0)--------normal ()\nBilirubin, Total: 0.9 mg/dL  (0.0, 1.5)--------normal ()\nCalcium, Total: 8.3 mg/dL  (8.4, 10.3)--------abnormal ()\nChloride: 100.0 mEq/L  (96.0, 108.0)--------normal ()\nCreatinine: 0.9 mg/dL  (0.5, 1.2)--------normal ()\nGlucose: 134.0 mg/dL  (70.0, 100.0)--------abnormal (IF FASTING, 70-100 NORMAL, >125 PROVISIONAL DIABETES.)\nHaptoglobin: 90.0 mg/dL  (30.0, 200.0)--------normal ()\nLactate Dehydrogenase (LD): 393.0 IU/L  (94.0, 250.0)--------abnormal ()\nMagnesium: 1.7 mg/dL  (1.6, 2.6)--------normal ()\nPhosphate: 3.5 mg/dL  (2.7, 4.5)--------normal ()\nPotassium: 3.7 mEq/L  (3.3, 5.1)--------normal ()\nSodium: 134.0 mEq/L  (133.0, 145.0)--------normal ()\nUrea Nitrogen: 16.0 mg/dL  (6.0, 20.0)--------normal ()\nINR(PT): 2.0   (0.9, 1.1)--------abnormal ()\nPT: 21.7 sec  (9.4, 12.5)--------abnormal ()\nPTT: 41.4 sec  (25.0, 36.5)--------abnormal ()"

model_dir = "/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/tangbohao-240108100043/mmedlm_model/Qwen2.5-7B-Instruct"
language_model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, device_map='cuda', torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir,
        model_max_length=8192,
        add_bos_token=True
    )
embedding_layer = language_model.get_input_embeddings()

context_ids = tokenizer(text, return_tensors="pt")['input_ids'][0].to("cuda")
context_embeds = embedding_layer(context_ids).unsqueeze(0)

prompt = "Output the contents word by word and make sure your output is totally same with it. Do not output any other words. \n"
prompt_ids = tokenizer(prompt, return_tensors="pt")['input_ids'][0].to("cuda")

eos = torch.tensor(tokenizer.eos_token_id).to("cuda")
eos_embedding = embedding_layer(eos).unsqueeze(0).unsqueeze(0)

prompt_embeds = embedding_layer(prompt_ids).unsqueeze(0)


inputs_embeds = torch.cat([prompt_embeds, context_embeds, eos_embedding], dim=1)
# inputs_embeds = torch.cat([pre_embeds, soft_prompt, model.AE], dim=1)

outputs = language_model.generate(
    inputs_embeds=inputs_embeds, 
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id,
    top_k=50,
)

prediction = tokenizer.decode(outputs[0], skip_special_tokens = True)

print("original text: \n", text)

print("prediction text : \n", prediction)
    