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
# ������ import gradio ǰ����
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

# ���ñ�ע��
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
    """����Ŀ����"""
    if image is None:
        return None, "�����ϴ�ͼƬ��"
    
    if not text.strip():
        return None, "������Ҫ�����������"
    
    try:
        # �����ı�����
        texts = [[t.strip()] for t in text.split(',')] + [[' ']]
        
        # ׼������
        data_info = dict(img_id=0, img=np.array(image), texts=texts)
        data_info = runner.pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                          data_samples=[data_info['data_samples']])

        # ����
        with autocast(enabled=False), torch.no_grad():
            output = runner.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        # NMS����
        keep = nms(pred_instances.bboxes,
                   pred_instances.scores,
                   iou_threshold=nms_thr)
        pred_instances = pred_instances[keep]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        # ���Ƽ�������
        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        
        # ����masks
        masks = pred_instances.get('masks', None)
        
        # ���������
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores'],
            mask=masks
        )
        
        # ������ǩ
        labels = [
            f"{texts[class_id][0]} {confidence:0.2f}" 
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # ���ƽ��
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
        if masks is not None:
            image = MASK_ANNOTATOR.annotate(image, detections)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(image)
        
        # ���ɽ����Ϣ
        num_detections = len(detections.confidence)
        info = f"�����ɣ������� {num_detections} ��Ŀ��"
        
        return result_image, info
        
    except Exception as e:
        return None, f"���ʧ�ܣ�{str(e)}"

def set_preset_categories(category_type):
    """����Ԥ�����"""
    presets = {
        "��������": "person, car, truck, bus, motorcycle, bicycle, dog, cat, bird, horse",
        "����": "dog, cat, bird, horse, cow, sheep, elephant, bear, zebra, giraffe",
        "��ͨ����": "car, truck, bus, motorcycle, bicycle, train, boat, airplane",
        "�ճ���Ʒ": "bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, pizza",
        "�Ҿ�": "chair, couch, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard",
        "������Ʒ": "frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket",
        "��ȫñ���": "helmet, head, person, worker, construction worker"
    }
    return presets.get(category_type, "")

def clear_all():
    """�����������"""
    return None, "", None, "�������������"

def save_result(image):
    """������ͼƬ"""
    if image is None:
        return None, "û�н��ͼƬ�ɱ���"
    
    # �������Ŀ¼
    os.makedirs("output_images", exist_ok=True)
    
    # �����ļ���
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_result_{timestamp}.jpg"
    filepath = os.path.join("output_images", filename)
    
    # ����ͼƬ
    image.save(filepath)
    return filepath, f"ͼƬ�ѱ��浽: {filepath}"

def create_enhanced_ui(runner):
    """������ǿ��UI����"""
    
    # �Զ���CSS��ʽ
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
    
    with gr.Blocks(title="YOLO-World ����Ŀ����", css=css, theme=gr.themes.Soft()) as demo:
        
        # ����
        gr.HTML("""
        <div class="main-header">
            <h1>?? YOLO-World ����Ŀ����ϵͳ</h1>
            <p>�ϴ�ͼƬ��������Ҫ����������𣬼��ɻ�ü����</p>
        </div>
        """)
        
        with gr.Row():
            # ���������
            with gr.Column(scale=2, elem_classes="control-panel"):
                gr.Markdown("### ?? ͼƬ�ϴ�")
                input_image = gr.Image(
                    type='pil', 
                    label="�ϴ�ͼƬ", 
                    height=300,
                    interactive=True
                )
                
                gr.Markdown("### ??? ����������")
                
                # Ԥ�����ѡ��
                category_type = gr.Dropdown(
                    choices=["��������", "����", "��ͨ����", "�ճ���Ʒ", "�Ҿ�", "������Ʒ", "��ȫñ���"],
                    label="ѡ��Ԥ�����",
                    value="��������"
                )
                
                # �ı������
                input_text = gr.Textbox(
                    lines=4,
                    label="������ (�ö��ŷָ�)",
                    placeholder="����: person, car, dog, cat",
                    value="person, car, truck, bus, motorcycle, bicycle"
                )
                
                # Ԥ�����ť
                category_type.change(
                    fn=set_preset_categories,
                    inputs=[category_type],
                    outputs=[input_text]
                )
                
                gr.Markdown("### ?? ������")
                
                with gr.Row():
                    max_num_boxes = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1,
                        label="����������"
                    )
                
                with gr.Row():
                    score_thr = gr.Slider(
                        minimum=0.01, maximum=1.0, value=0.3, step=0.01,
                        label="���Ŷ���ֵ"
                    )
                    
                with gr.Row():
                    nms_thr = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.7, step=0.01,
                        label="NMS��ֵ"
                    )
                
                gr.Markdown("### ?? ������ť")
                with gr.Row():
                    detect_btn = gr.Button("?? ��ʼ���", variant="primary", size="lg")
                    clear_btn = gr.Button("??? ���", variant="secondary")
            
            # �Ҳ������
            with gr.Column(scale=3, elem_classes="result-panel"):
                gr.Markdown("### ?? �����")
                
                output_image = gr.Image(
                    type='pil', 
                    label="�����", 
                    height=400,
                    interactive=False
                )
                
                # ״̬��Ϣ
                status_info = gr.Textbox(
                    label="״̬��Ϣ",
                    value="���ϴ�ͼƬ�����ü�����",
                    interactive=False
                )
                
                # ���水ť������
                with gr.Row():
                    save_btn = gr.Button("?? ������", variant="secondary")
                    download_file = gr.File(label="���ؽ��", visible=False)
        
        # ʹ��˵��
        gr.Markdown("""
        ### ?? ʹ��˵��
        1. **�ϴ�ͼƬ**: ���ͼƬ�ϴ�����ѡ��Ҫ����ͼƬ
        2. **ѡ�����**: ���Դ�Ԥ�������ѡ�񣬻����ֶ�������Ҫ������������ö��ŷָ���
        3. **��������**: ������Ҫ����������
           - **����������**: ������ʾ�ļ�������
           - **���Ŷ���ֵ**: ֻ��ʾ���Ŷȸ��ڴ�ֵ�ļ����
           - **NMS��ֵ**: ����ȥ���ص��ļ���
        4. **��ʼ���**: ���"��ʼ���"��ť��ý��
        5. **������**: ���Ա�������ͼƬ
        
        ### ?? ��ʾ
        - ֧�ּ�������������壬ֻ�������Ӧ��Ӣ������
        - ����ͬʱ���������ö��ŷָ�
        - �����ϴ������Ƚϸߵ�ͼƬ�Ի�ø��õļ��Ч��
        """)
        
        # ���¼�
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

    # ��������
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # ��ʼ��ģ��
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

    print("? ģ�ͼ�����ɣ�")
    print(f"?? ����UI���棬�˿�: {args.port}")

    # ����������UI
    demo = create_enhanced_ui(runner)
    demo.launch(
        server_name='0.0.0.0',
        server_port=args.port,
        share=args.share,
        show_api=False
    )

if __name__ == '__main__':
    main()