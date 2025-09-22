# YOLO-World Open-World Detection System
个人练手项目

---

## 1. 环境配置
原项目的配置方法较为复杂，建议参考知乎回答：
[知乎教程](https://zhuanlan.zhihu.com/p/1908833699748877163)  
> ⚠️ 注意 `numpy` 版本

也可以直接按照主目录中验证过的 `requirements.txt` 配置环境。

---

## 2. 模型文件配置
同 YOLO-World 项目，将模型文件放在 `./weights` 文件夹下，包括：
- CLIP 模型
- YOLO-World 模型

---

## 3. UI 界面
文件：`enhanced_ui_demo.py`  
添加了可视化界面设计，主要功能包括：

🎯 **主要功能**
- 中文界面：完全中文化的用户界面  
- 预设类别：提供常见物体、动物、交通工具等预设类别选择  
- 智能检测：支持检测任意类别的物体  
- 结果保存：可以保存检测结果图片  
- 参数调节：可视化的参数调节界面  

🚀 **使用方法**
1. 安装依赖（如果尚未安装）：  
   ```bash
   pip install -r requirements.txt
2.运行界面：
   ```bash
    python demo/enhanced_ui_demo.py \
    configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth --share
3.添加图片进行检测，界面会显示检测结果并支持保存。