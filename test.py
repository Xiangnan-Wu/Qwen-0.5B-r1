import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

class ChatModel:
    def __init__(self, model_path):
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        ).to("cuda:0")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 系统提示
        self.system_prompt = '''You are a helpful assistant. Always respond in the following XML format:
<reasoning>
Provide a clear step-by-step reasoning process.
</reasoning>
<answer>
Give a concise and accurate answer.
</answer>'''
        
        # 初始化对话历史
        self.conversation = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def generate_response(self, user_message):
        # 添加用户消息到对话历史
        self.conversation.append({"role": "user", "content": user_message})
        
        # 使用chat template构造输入
        inputs = self.tokenizer.apply_chat_template(
            self.conversation, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # 生成响应
        outputs = self.model.generate(
            inputs,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
        )
        
        # 解码响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 添加模型响应到对话历史
        self.conversation.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_conversation(self):
        # 重置对话历史
        self.conversation = [
            {"role": "system", "content": self.system_prompt}
        ]
        return []

def create_gradio_interface(model_path):
    # 初始化聊天模型
    chat_model = ChatModel(model_path)
    
    # 创建Gradio界面
    with gr.Blocks() as demo:
        # 聊天界面
        chatbot = gr.Chatbot(
            label="GRPO Model Chat",
            height=500
        )
        
        # 用户输入框
        msg = gr.Textbox(
            label="Your Message", 
            placeholder="Enter your message here..."
        )
        
        # 清除对话按钮
        clear = gr.ClearButton([msg, chatbot])
        
        # 提交消息的处理函数
        def respond(message, chat_history):
            # 生成响应
            response = chat_model.generate_response(message)
            
            # 更新聊天历史
            chat_history.append((message, response))
            
            return "", chat_history
        
        # 绑定消息提交事件
        msg.submit(
            respond, 
            [msg, chatbot], 
            [msg, chatbot]
        )
        
        # 清除对话按钮事件
        clear.click(
            chat_model.clear_conversation, 
            None, 
            chatbot
        )
    
    return demo

# 运行Gradio应用
def main():
    model_path = "your_model_path"  # 替换为你的模型路径
    
    # 创建并启动Gradio界面
    demo = create_gradio_interface(model_path)
    demo.launch(share=True)  # share=True可以创建公网链接

if __name__ == "__main__":
    main()