import io
import os
import xml.etree.ElementTree as ET  # 新增：用于生成 XML
from typing import List             # 新增：用于类型提示
from fastapi import FastAPI, File, UploadFile, Body, BackgroundTasks # 新增：用于后台运行训练任务
from fastapi.middleware.cors import CORSMiddleware # 1. 导入中间件
from fastapi.responses import HTMLResponse # 新增
from fastapi.staticfiles import StaticFiles # 新增
from pydantic import BaseModel      # 新增：用于定义接收的数据格式
from PIL import Image
import torch
import shutil # 新增：用于复制文件
import subprocess
import yaml

app = FastAPI()

# 2. 配置允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境请指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 新增：访问根路径时返回 index.html ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# 加载模型
# 使用本地的 yolov5s.pt 权重
model = torch.hub.load('./yolov5', 'custom', path='yolov5/yolov5s.pt', source='local')

# --- 新增：定义数据模型 (Pydantic) ---
# 这定义了前端传给我们的 JSON 数据长什么样
class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str  # 类别名称

class AnnotationData(BaseModel):
    filename: str       # 图片文件名
    width: int          # 图片宽度
    height: int         # 图片高度
    boxes: List[BoundingBox] # 包含的检测框列表
    save_image: bool = False # 新增：是否保存图片

# --- 新增：训练参数模型 ---
class TrainConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    data_yaml: str = "coco128.yaml" # 数据集配置文件路径
    weights: str = "yolov5s.pt"     # 预训练权重

# 全局变量记录训练状态 (简单实现，生产环境建议用数据库或 Redis)
training_status = {
    "is_training": False,
    "progress": 0,
    "message": "Idle"
}

def run_training_task(config: TrainConfig):
    """
    后台执行训练任务的函数
    """
    global training_status
    training_status["is_training"] = True
    training_status["message"] = "Training started..."
    
    # 构造命令
    # 注意：这里假设 python 环境就是当前环境
    cmd = [
        "python", "yolov5/train.py",
        "--img", str(config.img_size),
        "--batch", str(config.batch_size),
        "--epochs", str(config.epochs),
        "--data", config.data_yaml,
        "--weights", config.weights,
        "--project", "runs/train", # 训练结果保存路径
        "--name", "custom_exp",    # 实验名称
        "--exist-ok"               # 覆盖同名实验
    ]

    try:
        # 使用 subprocess 调用，并实时捕获输出（这里简化为等待结束）
        # 实际项目中可以使用 Popen 实时读取 stdout 更新进度
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            training_status["message"] = "Training completed successfully!"
        else:
            training_status["message"] = f"Training failed: {process.stderr}"
            
    except Exception as e:
        training_status["message"] = f"Error: {str(e)}"
    finally:
        training_status["is_training"] = False

@app.post("/train")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    """
    启动模型训练接口
    """
    if training_status["is_training"]:
        return {"status": "error", "message": "A training task is already running."}
    
    # 将训练任务加入后台队列
    background_tasks.add_task(run_training_task, config)
    
    return {"status": "success", "message": "Training task started in background."}

@app.get("/train/status")
async def get_training_status():
    """
    查询训练状态接口
    """
    return training_status

# --- 新增：模型评估接口 ---
class ValConfig(BaseModel):
    weights: str = "runs/train/custom_exp/weights/best.pt" # 默认使用刚才训练好的模型
    data_yaml: str = "coco128.yaml"
import re  # 新增：用于正则解析

@app.post("/evaluate")
async def evaluate_model(config: ValConfig):
    """
    运行模型评估，返回 mAP 等指标
    """
    if not os.path.exists(config.weights):
        return {"status": "error", "message": "Weights file not found. Please train first."}

    cmd = [
        "python", "yolov5/val.py",
        "--weights", config.weights,
        "--data", config.data_yaml,
        "--task", "val",
        "--project", "runs/val",
        "--name", "eval_exp",
        "--exist-ok"
    ]
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        output = process.stdout
        
        # --- 新增：解析关键指标 ---
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map95": 0.0
        }
        
        # 尝试更灵活的正则匹配
        # 匹配逻辑：找 "all" -> 跳过两个整数(Images, Instances) -> 抓取后面4个浮点数
        match = re.search(r'all\s+\d+\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', output)
        
        # 如果上面的没匹配到，尝试另一种格式（有时候 P R mAP 之前没有 Images Instances 列）
        if not match:
             match = re.search(r'all\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', output)

        if match:
            metrics["precision"] = float(match.group(1))
            metrics["recall"] = float(match.group(2))
            metrics["map50"] = float(match.group(3))
            metrics["map95"] = float(match.group(4))
        # -------------------------

        return {
            "status": "success",
            "output": output,
            "metrics": metrics, # 返回解析后的指标
            "error": process.stderr
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取上传的图片
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 进行推理
    results = model(image)

    # --- 修改保存逻辑 ---
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 先保存一份干净的原图 (用于后续标注保存)
    image.save(os.path.join(save_dir, file.filename))

    # 2. 再绘制框并保存一份可视化的图 (可选，为了区分可以加个前缀)
    results.render() 
    img_with_boxes = Image.fromarray(results.ims[0])
    img_with_boxes.save(os.path.join(save_dir, "labeled_" + file.filename))
    
    # 返回给前端显示的路径（前端其实用的是自己本地读取的流，这个路径主要是给后端自己用的）
    save_path = os.path.join(save_dir, file.filename)

    # 返回 JSON 结果，同时返回保存路径告知用户
    return {
        "message": "Success",
        "save_path": save_path,
        "detections": results.pandas().xyxy[0].to_dict(orient="records")
    }

# --- 新增：保存标注接口 ---
@app.post("/save_annotation")
async def save_annotation(data: AnnotationData):
    """
    接收修正后的标注数据，保存为 Pascal VOC 格式的 XML 文件，并保存图片
    """
    # 1. 准备保存目录
    xml_dir = "annotations"
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    # --- 新增：保存图片逻辑 ---
    if data.save_image:
        # 我们假设 predict 接口已经把干净的原图保存在了 results 目录下
        source_image_path = os.path.join("results", data.filename)
        target_image_path = os.path.join(xml_dir, data.filename)
        
        # 如果原图存在，就复制过去
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, target_image_path)
        else:
            print(f"警告: 找不到原图 {source_image_path}，仅保存 XML")
    # -------------------------

    # 2. 构建 XML 树结构 (Pascal VOC 标准格式)
    root = ET.Element("annotation")
    
    ET.SubElement(root, "folder").text = "annotations" # 修改文件夹名为 annotations
    ET.SubElement(root, "filename").text = data.filename
    
    # 尺寸信息
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(data.width)
    ET.SubElement(size, "height").text = str(data.height)
    ET.SubElement(size, "depth").text = "3"

    # 遍历所有框，添加 object 节点
    for box in data.boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box.label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        # 确保坐标是整数
        ET.SubElement(bndbox, "xmin").text = str(int(box.xmin))
        ET.SubElement(bndbox, "ymin").text = str(int(box.ymin))
        ET.SubElement(bndbox, "xmax").text = str(int(box.xmax))
        ET.SubElement(bndbox, "ymax").text = str(int(box.ymax))

    # 3. 生成 XML 文件
    xml_filename = os.path.splitext(data.filename)[0] + ".xml"
    save_path = os.path.join(xml_dir, xml_filename)
    
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="\t", level=0)
        
    tree.write(save_path, encoding="utf-8", xml_declaration=True)

    return {
        "message": "Annotation saved successfully",
        "xml_path": save_path,
        "image_saved": data.save_image
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)