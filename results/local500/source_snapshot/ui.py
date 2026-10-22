from functools import lru_cache
import os
for key in ('NO_PROXY','no_proxy'):
    os.environ[key]=','.join(filter(None,[os.environ.get(key,''),'127.0.0.1','localhost','::1']))
import gradio as gr
from .experiment import load_model,predict

@lru_cache(maxsize=1)
def resources():return load_model()

def answer(image,question,beams):
    if image is None:raise gr.Error('请先选择一张图片')
    if not question or len(question)>500:raise gr.Error('请输入1–500字符的英文问题')
    model,processor,device,_=resources()
    predictions,elapsed=predict(model,processor,[image.convert('RGB')],[question],int(beams),device)
    return predictions[0],f'设备：{device} ｜模型生成耗时：{elapsed:.3f}s ｜束宽：{int(beams)}'

def launch(port=7865):
    with gr.Blocks(title='实验5 · 本地视觉问答',theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 实验5 · 本地视觉问答\nBLIP · 选择图片，用英文提问，获得简短答案')
        with gr.Row():
            image=gr.Image(type='pil',label='输入图片',height=380)
            with gr.Column():
                question=gr.Textbox(label='英文问题',value='How many people are in the picture?')
                beams=gr.Radio([1,3],value=1,label='解码束宽')
                button=gr.Button('回答问题',variant='primary')
                result=gr.Textbox(label='模型回答');status=gr.Textbox(label='运行信息')
        button.click(answer,[image,question,beams],[result,status],concurrency_limit=1)
        gr.Markdown('模型可能出现计数、颜色和关系判断错误。批量评测与案例见 results/；VQAv2 validation 结果有训练重叠风险。')
    demo.queue().launch(server_name='127.0.0.1',server_port=port,share=False)
