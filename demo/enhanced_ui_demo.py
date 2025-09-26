# coding=gbk 
# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
import argparse
import os.path as osp
from functools import partial
import time
import datetime
import pathlib
# 必须在 import gradio 前设置
home_dir = os.path.expanduser("~")
gradio_tmp = os.path.join(home_dir, "gradio_tmp")
pathlib.Path(gradio_tmp).mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = gradio_tmp

import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.config import Config, DictAction
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

# 设置标注器
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
MASK_ANNOTATOR = sv.MaskAnnotator()

class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h

LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.6, text_thickness=2)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Enhanced UI Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='work directory', default='output')
    parser.add_argument('--port', type=int, default=7860, help='port for gradio interface')
    parser.add_argument('--share', action='store_true', help='share gradio interface')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    return parser.parse_args()

def run_detection(runner, image, text, max_num_boxes, score_thr, nms_thr):
    """运行目标检测"""
    if image is None:
        return None, "请先上传图片！"
    
    if not text.strip():
        return None, "请输入要检测的物体类别！"
    
    try:
        # 处理文本输入
        texts = [[t.strip()] for t in text.split(',')] + [[' ']]
        
        # 准备数据
        data_info = dict(img_id=0, img=np.array(image), texts=texts)
        data_info = runner.pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])

        # 推理
        with autocast(enabled=False), torch.no_grad():
            output = runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        # NMS处理
        keep = nms(pred_instances.bboxes,
                   pred_instances.scores,
                   iou_threshold=nms_thr)
        pred_instances = pred_instances[keep]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        # 限制检测框数量
        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        
        # 处理masks
        masks = pred_instances.get('masks', None)
        
        # 创建检测结果
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores'],
            mask=masks
        )
        
        # 创建标签
        labels = [
            f"{texts[class_id][0]} {confidence:0.2f}" 
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # 绘制结果
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        if masks is not None:
            image = MASK_ANNOTATOR.annotate(image, detections)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(image)
        
        # 生成结果信息
        num_detections = len(detections.confidence)
        info = f"检测完成！共发现 {num_detections} 个目标"
        
        return result_image, info
        
    except Exception as e:
        return None, f"检测失败：{str(e)}"

def set_preset_categories(category_type):
    """设置预设类别"""
    presets = {
        "常见物体": "person, car, truck, bus, motorcycle, bicycle, dog, cat, bird, horse",
        "动物": "dog, cat, bird, horse, cow, sheep, elephant, bear, zebra, giraffe",
        "交通工具": "car, truck, bus, motorcycle, bicycle, train, boat, airplane",
        "日常用品": "bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, pizza",
        "家具": "chair, couch, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard",
        "体育用品": "frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket",
        "安全帽检测": "helmet, head, person, worker, construction worker"
    }
    return presets.get(category_type, "")

def clear_all():
    """清空所有内容"""
    return None, "", None, "已清空所有内容"

def save_result(image):
    """保存结果图片"""
    if image is None:
        return None, "没有结果图片可保存"
    
    # 创建输出目录
    os.makedirs("output_images", exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_result_{timestamp}.jpg"
    filepath = os.path.join("output_images", filename)
    
    # 保存图片
    image.save(filepath)
    return filepath, f"图片已保存到: {filepath}"

def create_enhanced_ui(runner):
    """创建增强版UI界面"""
    
    # 自定义CSS样式
    css = """
    .gradio-container {
        font-family: 'Microsoft YaHei', Arial, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .control-panel {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .result-panel {
        background: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    """
    
    with gr.Blocks(title="YOLO-World 智能目标检测", css=css, theme=gr.themes.Soft()) as demo:
        
        # 标题
        gr.HTML("""
        <div class="main-header">
            <h1> YOLO-World 智能目标检测系统</h1>
            <p>上传图片，输入想要检测的物体类别，即可获得检测结果</p>
        </div>
        """)
        
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=2, elem_classes="control-panel"):
                gr.Markdown("###  图片上传")
                input_image = gr.Image(
                    type='pil', 
                    label="上传图片", 
                    height=300,
                    interactive=True
                )
                
                gr.Markdown("###  检测类别设置")
                
                # 预设类别选择
                category_type = gr.Dropdown(
                    choices=["常见物体", "动物", "交通工具", "日常用品", "家具", "体育用品", "安全帽检测"],
                    label="选择预设类别",
                    value="常见物体"
                )
                
                # 文本输入框
                input_text = gr.Textbox(
                    lines=4,
                    label="检测类别 (用逗号分隔)",
                    placeholder="例如: person, car, dog, cat",
                    value="person, car, truck, bus, motorcycle, bicycle"
                )
                
                # 预设类别按钮
                category_type.change(
                    fn=set_preset_categories,
                    inputs=[category_type],
                    outputs=[input_text]
                )
                
                gr.Markdown("###  检测参数")
                
                with gr.Row():
                    max_num_boxes = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1,
                        label="最大检测框数量"
                    )
                
                with gr.Row():
                    score_thr = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.3, step=0.01,
                        label="置信度阈值"
                    )
                    
                with gr.Row():
                    nms_thr = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.7, step=0.01,
                        label="NMS阈值"
                    )
                
                gr.Markdown("###  操作按钮")
                with gr.Row():
                    detect_btn = gr.Button(" 开始检测", variant="primary", size="lg")
                    clear_btn = gr.Button(" 清空", variant="secondary")
            
            # 右侧结果面板
            with gr.Column(scale=3, elem_classes="result-panel"):
                gr.Markdown("###  检测结果")
                
                output_image = gr.Image(
                    type='pil', 
                    label="检测结果", 
                    height=400,
                    interactive=False
                )
                
                # 状态信息
                status_info = gr.Textbox(
                    label="状态信息",
                    value="请上传图片并设置检测类别",
                    interactive=False
                )
                
                # 保存按钮和下载
                with gr.Row():
                    save_btn = gr.Button(" 保存结果", variant="secondary")
                    download_file = gr.File(label="下载结果", visible=False)
        
        # 使用说明
        gr.Markdown("""
        ###  使用说明
        1. **上传图片**: 点击图片上传区域，选择要检测的图片
        2. **选择类别**: 可以从预设类别中选择，或者手动输入想要检测的物体类别（用逗号分隔）
        3. **调整参数**: 根据需要调整检测参数
           - **最大检测框数量**: 限制显示的检测框数量
           - **置信度阈值**: 只显示置信度高于此值的检测结果
           - **NMS阈值**: 用于去除重叠的检测框
        4. **开始检测**: 点击"开始检测"按钮获得结果
        5. **保存结果**: 可以保存检测结果图片
        
        ###  提示
        - 支持检测任意类别的物体，只需输入对应的英文名称
        - 可以同时检测多个类别，用逗号分隔
        - 建议上传清晰度较高的图片以获得更好的检测效果
        """)
        
        # 绑定事件
        detect_btn.click(
            fn=partial(run_detection, runner),
            inputs=[input_image, input_text, max_num_boxes, score_thr, nms_thr],
            outputs=[output_image, status_info]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[input_image, input_text, output_image, status_info]
        )
        
        save_btn.click(
            fn=save_result,
            inputs=[output_image],
            outputs=[download_file, status_info]
        )
    
    return demo

def main():
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # 初始化模型
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    print("? 模型加载完成！")
    print(f" 启动UI界面，端口: {args.port}")

    # 创建并启动UI
    demo = create_enhanced_ui(runner)
    demo.launch(
        server_name='0.0.0.0',
        server_port=args.port,
        share=args.share,
        show_api=False
    )

if __name__ == '__main__':
    main()
