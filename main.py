import torch
from transformers import  AutoTokenizer, PreTrainedTokenizerFast, AdamW, AutoModelForCausalLM, BitsAndBytesConfig,HfArgumentParser, get_scheduler, set_seed

import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torch import cuda
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

from tqdm import tqdm

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import bitsandbytes as bnb
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

config = {'mode_ID':"microsoft/Phi-3-mini-4k-instruct",
          'seed': 1 ,
          'max_seq_len' : 4096,
          'epochs': 3,
          'lr': 2e-4,
          'batch': 4,
          'lora_r':8,
          'lora_alpha':32,
          'target_module':["q_proj", "up_proj", "o_proj", "k_proj", "down_proj","gate_proj", "v_proj"],
          'lora_dropout':0.05,
          'lora_tasktype' :'CAUSAL_LM',
          'lora_bias' : 'none',
          'optimizer': 'paged_adamw_8bit',
          'scheduler':'cosine'}


from peft import (
    get_peft_config,  # PEFT 설정을 가져오기 위한 함수
    get_peft_model,  # PEFT 모델을 가져오기 위한 함수
    get_peft_model_state_dict,  # PEFT 모델 상태 사전을 가져오기 위한 함수
    set_peft_model_state_dict,  # PEFT 모델 상태 사전을 설정하기 위한 함수
    LoraConfig,  # LoRA 모델 구성을 정의하는 클래스
    PeftType,  # PEFT 모델의 타입을 정의
    PrefixTuningConfig,  # PrefixTuning 모델 구성을 정의하는 클래스
    PromptEncoderConfig,  # PromptEncoder 모델 구성을 정의하는 클래스
    PeftModel,  # PEFT 모델을 정의하는 클래스
    PeftConfig,  # PEFT 모델의 구성을 정의하는 클래스
)

# PEFT 모델의 타입 설정 (LoRA로 설정)
peft_type = PeftType.LORA

# LoRA 모델을 위한 설정
peft_config = LoraConfig(
    r=config['lora_r'],  # LoRA 모델의 r 값
    lora_alpha=config['lora_alpha'],  # LoRA 모델의 alpha 값
    target_modules=config['target_module'],  # LoRA 모델의 타겟 모듈 리스트
    lora_dropout=config['lora_dropout'],  # LoRA 모델의 드롭아웃 비율
    bias=config['lora_bias'],  # LoRA 모델의 편향 설정
    task_type=config['lora_tasktype']  # LoRA 모델의 태스크 유형
)

# AutoTokenizer를 사용하여 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(config['mode_ID'], trust_remote_code=True, eos_token='</s>')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
	config['mode_ID'],
	device_map="cuda",
	torch_dtype=torch.float16,
	trust_remote_code=True, 
	use_cache=False,
	# quantization_config=bnb_config,
)

model.gradient_checkpointing_enable() # 모델에서 그래디언트 체크포인팅 활성화 (메모리 효율 향상)
from peft import prepare_model_for_kbit_training # peft 라이브러리에서 k 비트 학습 준비 함수 임포트

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") # CUDA 사용 가능 여부 확인

model = prepare_model_for_kbit_training(model)# k 비트 학습을 위해 모델 준비 - prepare_model_for_kbit_training 함수 사용
model = get_peft_model(model, peft_config) # PEFT 적용 
model = model.to(device) # 모델을 학습 장치 (GPU 등)로 이동

def make_prompt(user_request, answer):
    
    conversation = [ {'role': 'user', 'content': user_request},
                  {'role': 'assistant', 'content': answer}]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    return prompt


from datasets import load_dataset
dataset = load_dataset('qiaojin/PubMedQA', 'pqa_artificial')
q = dataset['train']['question']
c = dataset['train']['context']
label = dataset['train']['final_decision']

# 'q'와 'c'를 페어링하여 input 열 만들기
input_list = [f"Question: {q_} Context: {c_}" for q_, c_ in zip(q, c)]


df_all = pd.DataFrame({'input': input_list,'label':label})
# 'no'인 값 10000개 추출
no_df = df_all[df_all['label'] == 'no'].sample(n=10000, random_state=42)

# 'yes'인 값 10000개 추출
yes_df = df_all[df_all['label'] == 'yes'].sample(n=10000, random_state=42)

# 두 데이터 프레임 합치기
combined_df = pd.concat([no_df, yes_df])


# train, test 데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(combined_df['input'], combined_df['label'], test_size=0.2, random_state=42)

# train, val 데이터셋 나누기
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


del dataset, q, c, label, input_list, df_all, no_df, yes_df, combined_df

train_data_prompt_list = []
for x,y in zip(X_train, y_train):
    train_data_prompt_list.append(make_prompt(x,y))

valid_data_prompt_list = []
for x2,y2 in zip(X_val, y_val):
    valid_data_prompt_list.append(make_prompt(x2,y2))

test_data_prompt_list = []
for x3,y3 in zip(X_test, y_test):
    test_data_prompt_list.append(make_prompt(x3,y3))

del x, y, x2, y2, x3, y3

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



train_dataset = Dataset(train_data_prompt_list)
valid_dataset = Dataset(valid_data_prompt_list)

def train(epoch, loader):
    model.train()
    loss_avg = 0
    for i, prompt in enumerate(loader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        #loss.backward()
        #optimizer.step()
        scaler.update()
        print(f"epoch : {epoch} - step : {i}/{len(loader)} - loss: {loss.item()}")
        loss_avg += loss.item()
        
        del inputs
        del outputs
        del loss
        
    print(f'Epoch: {epoch}, train_Loss:  {loss_avg/len(loader)}')
    loss_dic['Train'].append(loss_avg/len(loader))

        
def validate(epoch,loader):  
    model.eval()
    loss_avg = 0
    with torch.no_grad():       
        for i, prompt in enumerate(loader):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss_avg += loss.item()
            
            del inputs
            del outputs
            del loss
            
    print(f'Epoch: {epoch}, Valid_Loss:  {loss_avg/len(loader)}')
    loss_dic['Val'].append(loss_avg/len(loader))



train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

scaler = GradScaler()
optimizer = AdamW(model.parameters(), lr = 3e-4)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=227,
    num_training_steps=15000
)

from tqdm import tqdm
import time

loss_dic = {"epoch":[],"Train":[], "Val":[]}
best_loss = 100
early_stop_count = 0

for epoch in tqdm(range(1, 10)):
    
    loss_dic['epoch'].append(epoch)
    train(epoch, train_loader)
    validate(epoch, valid_loader)
    lr_scheduler.step()
    
    if loss_dic['Val'][epoch - 1] > best_loss:
        early_stop_count += 1       
        if early_stop_count >= 2:
            loss_dic_df = pd.DataFrame(loss_dic)
            # loss_dic_df.to_excel('./loss.xlsx', index=False)
            # torch.save(model.state_dict(), f'./bestmodel_{epoch}.pth')
            break
    else:
        best_loss = loss_dic['Val'][epoch - 1]
        early_stop_count = 0

if __name__ == '__main__':
    print('Train Start')