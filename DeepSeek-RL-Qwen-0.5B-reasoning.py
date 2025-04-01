import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# 加载数据集

SYSTEM_PROMPT = '''
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
'''

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>"""


def extract_xml_answer(text: str) -> str:
    '''提取生成的目标答案'''
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str| None:
    '''提取数据集label'''
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    '''提取训练集
    数据集预处理后格式：
    {prompt:PROMPT,
    answer:ANSWER}'''
    data = load_dataset('openai/gsm8k','main')[split]
    data = data.map( lambda x :{
        'prompt':[
            {'role':'system','content':SYSTEM_PROMPT},
            {'role': 'user','content':x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data
# 准确性奖励
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions] #提取模型的回答
    q = prompts[0][-1]['content'] # 提取问题
    extracted_responses = [extract_xml_answer(r) for r in responses] # 从模型回答中，提取XML格式的ans
    print('-' * 20 ,f"Question:\n{q}",f"\nAnswer:\n{answer[0]}",f"\nResponse:\n{responses[0]}",f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r,a in zip(extracted_responses, answer)] # 如果答案与正确答案相同，返回奖励2，否则0

# 纯数字奖励 --> 让模型的输出只包含数字
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extract_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extract_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    '''严格格式奖励, 符合XML格式返回奖励'''
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    '''软格式奖励'''
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    '''符合xml要求格式就会产生奖励，不需要完全'''
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) ->list[float]:
    contents = [completion[0]['content'] for completion in completions]
    return [count_xml(c) for c in contents]

# 获取训练集
dataset = get_gsm8k_questions()

# 加载模型
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

output_dir = "outputs/Qwen-0.5B-GRPO"
run_name = "Qwen-0.5B-GRPO-gsm8k"

training_config = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=5,
    bf16=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=400,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    report_to="wandb"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.bfloat16,
    device_map = None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model = model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func
    ],
    args = training_config,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(output_dir)