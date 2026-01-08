# PDF批量OCR识别系统

一个基于PaddleOCR的PDF批量识别系统，**特别适合多页PDF识别同一绝对位置的批量处理**，支持鼠标画框选择识别区域，智能多进程处理，输出Excel结果。

## ✨ 功能特性

- **批量处理**：支持同时处理多个PDF文件
- **同一位置识别**：**核心功能** - 多页PDF中识别同一绝对位置的内容
- **可视化画框**：鼠标拖拽选择识别区域，一次设置，多页复用
- **智能多进程**：根据系统资源自动计算最佳核心数
- **手动核心设置**：支持用户手动指定使用的核心数
- **图像处理**：包含灰度化、二值化、去噪功能
- **Excel输出**：识别结果自动输出为Excel文件，每页一行，方便数据统计
- **进度显示**：实时显示处理进度
- **配置保存**：自动保存识别区域配置，下次直接使用

## 📋 环境要求

- Python 3.10+ 
- Windows/Linux/macOS

## 🛠️ 安装步骤

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装系统依赖

#### Windows
1. 下载 [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
2. 解压到项目根目录，确保目录结构为 `./poppler/Library/bin`
3. 或添加 poppler 的 bin 目录到系统环境变量

#### Linux
```bash
sudo apt-get install poppler-utils
```

#### macOS
```bash
brew install poppler
```

## 🚀 使用指南

### 1. 准备PDF文件

将需要处理的PDF文件放入项目根目录。

### 2. 运行程序

```bash
python paddle_ocr_regions.py
```

### 3. 操作流程

#### 步骤1：扫描PDF文件
程序会自动扫描当前目录下的所有PDF文件。

#### 步骤2：绘制识别区域
- 如果是首次运行，会弹出窗口让您绘制识别区域
- 拖动鼠标在PDF预览图上绘制矩形框
- 按 `Enter` 键保存并退出，按 `ESC` 键跳过
- 配置会自动保存到 `selected_boxes.json` 文件

#### 步骤3：选择核心模式
程序会显示核心选择菜单：
```
请选择核心选择模式:
1. 智能选择核心数 (根据系统资源自动调节)
2. 手动设置核心数
请输入选择 (1/2，默认1): 
```

- 选择 `1`：程序会根据系统资源自动计算最佳核心数
- 选择 `2`：手动输入要使用的核心数

#### 步骤4：等待处理完成
程序会开始批量处理PDF文件，显示实时进度：
```
开始处理: 图纸2.pdf
  系统分析:
    - CPU核心数: 4
    - 可用内存: 7.8GB
    - PDF大小: 0.03GB
    - 估算每进程内存: 0.07GB
    - 预留系统内存: 1.0GB
    - PDF因子: 1.0
    - 推荐核心数: 4

总页数: 39
使用 4 个CPU核心并发处理...
开始处理...
图纸2.pdf: 100%|██████████| 39/39 [01:23<00:00,  2.14s/页]
```

#### 步骤5：查看结果
处理完成后，会在当前目录生成Excel文件，文件名格式为 `原文件名_鼠标OCR.xlsx`。

## 📁 项目结构

```
.
├── paddle_ocr_regions.py    # 主程序文件
├── requirements.txt         # 依赖清单
├── selected_boxes.json      # 识别区域配置（自动生成）
├── poppler/                 # Poppler工具（Windows需要手动下载）
│   └── Library/
│       └── bin/
└── README.md                # 项目文档
```

## 📦 依赖说明

| 依赖库 | 版本要求 | 用途 |
| --- | --- | --- |
| opencv-python | >=4.7.0 | 图像处理 |
| Pillow | >=9.0.0 | 图像处理 |
| numpy | >=1.21.0 | 数值计算 |
| paddleocr | >=2.7.0 | OCR识别 |
| pdf2image | >=1.16.0 | PDF转图像 |
| pandas | >=1.5.0 | 数据处理和Excel输出 |
| openpyxl | >=3.0.0 | Excel输出依赖 |
| psutil | >=5.9.0 | 系统资源监控 |
| tqdm | >=4.64.0 | 进度显示 |

## ⚠️ 注意事项

1. **PaddleOCR安装**：首次安装paddleocr可能需要较长时间，建议使用国内镜像源加速安装：
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **内存占用**：处理大型PDF文件时，建议根据系统内存情况调整核心数，避免内存溢出。

3. **识别区域配置**：
   - 配置文件 `selected_boxes.json` 会自动保存
   - 如果需要重新绘制区域，可删除该文件后重新运行程序
   - 不同DPI设置会导致配置不兼容

4. **图像处理**：
   - 保留了灰度化、二值化、去噪功能
   - 移除了图像矫正纠偏功能，如有需要可自行添加

5. **多进程处理**：
   - 智能核心数计算会考虑系统内存和PDF大小
   - 手动设置核心数时，建议留至少1个核心给系统

## 📝 更新日志

- 移除了图像矫正纠偏功能
- 保留了灰度化、二值化、去噪功能
- 添加了智能/手动核心选择功能
- 优化了配置保存逻辑

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License
