#!/usr/bin/env python3
"""
Zero-shot helmet detection demo with enhanced prompts.
This script demonstrates how to use YOLO-World for helmet detection without training.
"""

import os
import cv2
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


def load_helmet_prompts(prompt_path):
    """Load helmet detection prompts from JSON file."""
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    return prompts


def create_enhanced_prompts():
    """Create enhanced prompts for better zero-shot performance."""
    return [
        # Class 0: Wearing helmet
        [
            "a person wearing a motorcycle helmet",
            "motorcyclist with a proper safety helmet",
            "biker wearing a protective helmet",
            "person on motorcycle with helmet on head",
            "scooter rider wearing a safety helmet",
            "cyclist with a protective helmet",
            "rider wearing a motorcycle helmet",
            "person with bike helmet on head",
            "motorcyclist with full face helmet",
            "biker with DOT approved helmet",
            "person wearing a hard shell helmet",
            "rider with a certified safety helmet"
        ],
        # Class 1: Not wearing helmet
        [
            "a person without a helmet",
            "motorcyclist without any head protection",
            "biker riding without helmet",
            "person on motorcycle without helmet",
            "scooter rider without helmet",
            "cyclist without helmet",
            "rider without helmet",
            "person without head protection",
            "motorcyclist with bare head",
            "biker with exposed head",
            "person riding without safety gear",
            "rider with unprotected head"
        ],
        # Class 2: Wearing fake helmet
        [
            "a person wearing a fake helmet",
            "rider with a fake or improvised helmet",
            "person wearing a bucket on head",
            "rider with a bowl on head",
            "person wearing a pot as helmet",
            "rider with a basket on head",
            "person wearing a cap as helmet",
            "rider with cardboard on head",
            "person wearing a hood as helmet",
            "rider with makeshift helmet",
            "person wearing non-protective headgear",
            "rider with decorative head covering"
        ],
        # Class 3: Construction hard hat
        [
            "a construction worker wearing a hard hat",
            "person wearing a construction helmet",
            "worker with a safety hard hat",
            "person on construction site with helmet",
            "engineer wearing a hard hat",
            "worker wearing a safety helmet",
            "person wearing a hard hat at work",
            "construction worker with hard hat",
            "worker with protective hard hat",
            "person wearing work safety helmet",
            "construction site worker with helmet",
            "person wearing OSHA approved hard hat"
        ]
    ]


def inference_zeroshot(model, image_path, prompts, test_pipeline, 
                      score_thr=0.3, max_dets=100, device='cuda:0'):
    """Perform zero-shot inference on helmet detection."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = image[:, :, [2, 1, 0]]  # BGR to RGB
    
    # Prepare data
    data_info = dict(img=image, img_id=0, texts=prompts)
    data_info = test_pipeline(data_info)
    data_batch = dict(
        inputs=data_info['inputs'].unsqueeze(0),
        data_samples=[data_info['data_samples']]
    )
    
    # Run inference
    with torch.no_grad():
        output = model.test_step(data_batch)[0]
    
    # Process results
    pred_instances = output.pred_instances
    
    # Apply score threshold
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    
    # Limit max detections
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]
    
    # Convert to numpy
    pred_instances = pred_instances.cpu().numpy()
    boxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']
    
    # Map labels to text descriptions
    class_names = ['wearing_helmet', 'no_helmet', 'fake_helmet', 'hard_hat']
    label_texts = [class_names[label] for label in labels]
    
    return boxes, labels, label_texts, scores


def visualize_results(image_path, boxes, labels, scores, class_names, output_path=None):
    """Visualize detection results."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Color mapping for different classes
    colors = {
        0: (0, 255, 0),    # Green for wearing helmet
        1: (255, 0, 0),    # Red for no helmet
        2: (255, 255, 0),  # Yellow for fake helmet
        3: (0, 0, 255)     # Blue for hard hat
    }
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = colors.get(label, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label and score
        label_text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Zero-shot helmet detection demo')
    parser.add_argument('--config', type=str, 
                       default='configs/zeroshot/yolo_world_v2_s_helmet_zeroshot.py',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       default='pretrained_models/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--prompts', type=str,
                       default='data/texts/helmet_zeroshot_prompts.json',
                       help='Path to prompts JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image')
    parser.add_argument('--score-thr', type=float, default=0.3,
                       help='Score threshold for detections')
    parser.add_argument('--max-dets', type=int, default=100,
                       help='Maximum number of detections')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for inference')
    parser.add_argument('--use-enhanced', action='store_true',
                       help='Use enhanced prompts instead of file')
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    
    # Initialize model
    print(f"Loading model from: {args.checkpoint}")
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    
    # Setup test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)
    
    # Load prompts
    if args.use_enhanced:
        print("Using enhanced prompts...")
        prompts = create_enhanced_prompts()
    else:
        print(f"Loading prompts from: {args.prompts}")
        prompts = load_helmet_prompts(args.prompts)
    
    print(f"Number of classes: {len(prompts)}")
    for i, class_prompts in enumerate(prompts):
        print(f"Class {i}: {len(class_prompts)} prompts")
    
    # Run inference
    print(f"Running inference on: {args.image}")
    try:
        boxes, labels, label_texts, scores = inference_zeroshot(
            model, args.image, prompts, test_pipeline, 
            args.score_thr, args.max_dets, args.device
        )
        
        # Print results
        print(f"\nDetection results ({len(boxes)} objects found):")
        class_names = ['wearing_helmet', 'no_helmet', 'fake_helmet', 'hard_hat']
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box
            print(f"  {i+1}: {class_names[label]} (score: {score:.3f}) "
                  f"at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # Visualize results
        if args.output:
            print(f"Saving visualization to: {args.output}")
            visualize_results(args.image, boxes, labels, scores, 
                            class_names, args.output)
        else:
            print("Use --output to save visualization")
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
