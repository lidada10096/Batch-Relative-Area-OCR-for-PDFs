# mouse_ocr_allinone.py
import os
import sys
import json
import time
import threading
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import multiprocessing
import psutil

# ---------------- 参数配置 ----------------
DPI = 400
# 指向当前目录下的poppler文件夹，方便移植
# 检查poppler文件夹是否存在，如果不存在则使用系统路径
poppler_dir = os.path.join(os.path.dirname(__file__), 'poppler')
# 确保指向poppler的bin目录，因为convert_from_path需要找到pdftocairo等可执行文件
# 对于Windows系统，poppler的bin目录通常在Library子目录下
if os.path.exists(poppler_dir):
    # 检查是否有Library/bin结构（Windows poppler）
    library_bin_path = os.path.join(poppler_dir, 'Library', 'bin')
    # 检查是否有直接的bin目录（Linux/Mac poppler）
    direct_bin_path = os.path.join(poppler_dir, 'bin')
    
    if os.path.exists(library_bin_path):
        POPPLER_PATH = library_bin_path
    elif os.path.exists(direct_bin_path):
        POPPLER_PATH = direct_bin_path
    else:
        POPPLER_PATH = None
else:
    POPPLER_PATH = None


# ---------------- 图像预处理与纠偏函数 ----------------
def preprocess_image(image):
    """
    图像预处理：灰度化、二值化、去噪
    """
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 自适应二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # 去噪
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised

# ---------------- 智能核心数计算函数 ----------------
def calculate_optimal_workers(pdf_file, max_allowed_workers=None):
    """
    根据系统资源和PDF大小智能计算最佳核心数
    max_allowed_workers: 手动设置的最大核心数（可选）
    """
    # 获取CPU核心数
    cpu_count = multiprocessing.cpu_count()
    
    # 获取系统可用内存（GB）
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    # 获取PDF文件大小（GB）
    pdf_size = os.path.getsize(pdf_file) / (1024**3)
    
    # 估算每个进程需要的内存（PDF大小的2倍，用于页面加载和OCR）
    estimated_memory_per_worker = pdf_size * 2
    
    # 预留1GB给系统和其他程序
    usable_memory = max(1, available_memory - 1)
    
    # 基于内存计算最大安全核心数
    max_workers_by_memory = max(1, int(usable_memory / estimated_memory_per_worker))
    
    # 基于PDF大小调整：
    # - 小文件（<100MB）：可以使用全部核心
    # - 中等文件（100-200MB）：使用75%核心
    # - 大文件（>200MB）：使用50%核心（但至少2核）
    if pdf_size < 0.1:  # <100MB
        pdf_factor = 1.0
    elif pdf_size < 0.2:  # 100-200MB
        pdf_factor = 0.75
    else:  # >200MB
        pdf_factor = 0.5
    
    # 计算最终核心数
    optimal_workers = min(cpu_count, max_workers_by_memory)
    optimal_workers = max(2, int(optimal_workers * pdf_factor))  # 至少2核
    
    # 限制范围：2-4核
    optimal_workers = min(4, max(2, optimal_workers))
    
    # 如果手动设置了核心数，使用手动设置的值
    if max_allowed_workers is not None:
        optimal_workers = min(optimal_workers, max_allowed_workers)
        optimal_workers = max(1, optimal_workers)  # 确保至少1核
    
    print(f"  系统分析:")
    print(f"    - CPU核心数: {cpu_count}")
    print(f"    - 可用内存: {available_memory:.1f}GB")
    print(f"    - PDF大小: {pdf_size:.2f}GB")
    print(f"    - 估算每进程内存: {estimated_memory_per_worker:.2f}GB")
    print(f"    - 预留系统内存: 1.0GB")
    print(f"    - PDF因子: {pdf_factor}")
    print(f"    - 推荐核心数: {optimal_workers}")
    
    return optimal_workers

# ---------------- 鼠标画框部分 ----------------
def load_boxes_from_config():
    """尝试从配置文件加载图框坐标"""
    config_file = "selected_boxes.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                boxes = config.get("boxes", [])
                saved_dpi = config.get("dpi", DPI)
                if boxes and saved_dpi == DPI:
                    print(f"✓ 检测到已存在的配置文件")
                    print(f"  - 文件: {config_file}")
                    print(f"  - 已保存 {len(boxes)} 个图框")
                    print(f"  - DPI设置: {saved_dpi}")
                    return boxes
                elif boxes:
                    print(f"⚠ 配置文件DPI不匹配 ({saved_dpi} != {DPI})，将重新设置")
        except Exception as e:
            print(f"⚠ 读取配置文件失败: {str(e)}，将重新设置")
    return None

def draw_boxes():
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("错误: 当前目录未找到PDF文件")
        sys.exit(1)
    
    # 先尝试从配置文件加载
    boxes = load_boxes_from_config()
    if boxes is not None:
        return boxes
    
    # 如果没有配置文件，需要用户画框
    sample_pdf = pdf_files[0]
    print(f"\n未检测到配置文件，请绘制识别区域")
    print(f"使用示例文件: {sample_pdf}")
    print("提示: 在窗口中拖动鼠标绘制矩形框，按回车键保存并退出")
    
    # 加载示例PDF页面
    img_pil = convert_from_path(sample_pdf, dpi=DPI, first_page=1, last_page=1, poppler_path=POPPLER_PATH)[0]
    img_np  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_height, img_width = img_np.shape[:2]
    
    drawing, ix, iy = False, -1, -1
    boxes = []
    
    def draw_hint_text(image):
        """在图片上绘制操作提示"""




        # 使用粗体字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 增大字体尺寸
        font_scale = 1.2
        # 加粗线条
        thickness = 3
        
        # 使用英文提示，避免OpenCV中文乱码问题
        text1 = "Hint: Drag=Draw Rectangle | Enter=Save | ESC=Skip"
        text2 = f"Boxes: {len(boxes)}"
        
        # 大红色文字，无背景
        cv2.putText(image, text1, (20, 40), font, font_scale, (0, 0, 255), thickness)
        cv2.putText(image, text2, (20, 80), font, font_scale, (0, 0, 255), thickness)
        
        return image
    
    def mouse_rect(event, x, y, flags, param):
        nonlocal drawing, ix, iy, img_np, boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, ix, iy = True, x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_copy = img_np.copy()
            # 绘制深红色粗线条矩形框
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 0, 139), 3)
            # 添加提示文本
            img_copy = draw_hint_text(img_copy)
            cv2.imshow('draw', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            boxes.append((ix, iy, x, y))
            print(f"框{len(boxes)}: 左上({ix},{iy}) 右下({x},{y})")
            # 绘制所有已画的框和提示文字
            img_copy = img_np.copy()
            # 重绘所有已保存的框
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 139), 3)
            # 添加提示文本
            img_copy = draw_hint_text(img_copy)
            cv2.imshow('draw', img_copy)
    
    cv2.namedWindow('draw', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('draw', mouse_rect)
    
    # 初始显示图片并添加提示
    display_img = draw_hint_text(img_np.copy())
    cv2.imshow('draw', display_img)
    
    # 等待用户操作
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # 回车键
            break
        elif key == 27:  # ESC键
            boxes = []
            break
    
    cv2.destroyAllWindows()
    
    if not boxes:
        print("跳过画框，使用默认设置")
        return []
    
    with open("selected_boxes.json", "w", encoding="utf-8") as f:
        json.dump({"dpi": DPI, "boxes": boxes}, f, ensure_ascii=False)
    print("坐标已保存 → selected_boxes.json")
    return boxes

def pil2nd(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ---------------- 线程安全的进度管理类 ----------------
class ThreadSafeProgress:
    def __init__(self, total_pages, desc):
        self.total_pages = total_pages
        self.processed = 0
        self.lock = threading.Lock()
        self.pbar = tqdm(total=total_pages, desc=desc, unit="页", leave=True)
        self.start_time = time.time()
    
    def update(self, n=1):
        with self.lock:
            self.processed += n
            self.pbar.update(n)
    
    def close(self):
        self.pbar.close()

# ---------------- 单进程OCR处理函数（用于多进程池） ----------------
def ocr_page_worker(page_idx, pdf_file, boxes):
    # 每个进程独立初始化OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
    
    try:
        # 获取单页PDF
        pages = convert_from_path(pdf_file, dpi=DPI, first_page=page_idx, last_page=page_idx, poppler_path=POPPLER_PATH)
        if not pages:
            raise ValueError(f"页面 {page_idx} 转换为空")
        
        page = pages[0]
        if page is None or page.size == (0, 0):
            raise ValueError(f"页面 {page_idx} 为空图像")
        
        row = {"页码": page_idx}
        for idx, (x1, y1, x2, y2) in enumerate(boxes, 1):
            # 确保裁剪区域有效
            page_width, page_height = page.size
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(page_width, x2)
            y2 = min(page_height, y2)
            
            if x1 >= x2 or y1 >= y2:
                # 无效的裁剪区域
                row[f"Box{idx}"] = ""
                continue
            
            # 裁剪区域
            cropped = page.crop((x1, y1, x2, y2))
            if cropped is None or cropped.size == (0, 0):
                row[f"Box{idx}"] = ""
                continue
            
            # 转换为OpenCV格式
            roi = np.array(cropped)
            if roi.size == 0:
                row[f"Box{idx}"] = ""
                continue
            
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            ocr_res = ocr.ocr(roi, cls=True)
            text = "\n".join([line[1][0] for line in ocr_res[0]]) if ocr_res and ocr_res[0] else ""
            row[f"Box{idx}"] = text
        return row
    except Exception as e:
        # 捕获所有异常，返回错误信息
        print(f"  详细错误: {str(e)}")
        raise

# ---------------- 多进程优化的单文件处理函数 ----------------
def process_single_pdf_multiprocess(pdf_file, boxes, manual_cores=None):
    try:
        pdf_name = os.path.basename(pdf_file)
        print(f"\n{'='*60}")
        print(f"开始处理: {pdf_name}")
        print(f"{'='*60}")
        
        # 智能计算最佳核心数
        max_workers = calculate_optimal_workers(pdf_file, manual_cores)
        
        # 先获取总页数
        page_cnt = len(convert_from_path(pdf_file, dpi=1, poppler_path=POPPLER_PATH))
        print(f"\n总页数: {page_cnt}")
        print(f"使用 {max_workers} 个CPU核心并发处理...")
        print("开始处理...")
        
        records = [None] * page_cnt
        start_time = time.time()
        
        # 创建进度管理
        progress = ThreadSafeProgress(page_cnt, pdf_name)
        
        # 使用进程池并发处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有页面处理任务
            futures = {}
            for page_idx in range(1, page_cnt + 1):
                future = executor.submit(ocr_page_worker, page_idx, pdf_file, boxes)
                futures[future] = page_idx
            
            # 收集结果
            for future in as_completed(futures):
                page_idx = futures[future]
                try:
                    result = future.result()
                    records[page_idx - 1] = result
                    progress.update(1)
                except Exception as e:
                    print(f"  警告: 处理第 {page_idx} 页时出错: {str(e)}")
                    records[page_idx - 1] = {"页码": page_idx}
        
        progress.close()
        
        elapsed_time = time.time() - start_time
        out_file = pdf_file.replace(".pdf", "_鼠标OCR.xlsx")
        
        # 过滤None值
        valid_records = [r for r in records if r is not None]
        if valid_records:
            pd.DataFrame(valid_records).to_excel(out_file, index=False)
        
        print(f"\n{'─'*60}")
        print(f"✓ {pdf_name} 处理完成")
        print(f"  页数: {page_cnt}")
        print(f"  耗时: {elapsed_time:.1f}秒")
        print(f"  速度: {page_cnt/elapsed_time:.2f}页/秒")
        print(f"  输出: {out_file}")
        print(f"{'─'*60}")
        
        return {
            'file': pdf_name,
            'pages': page_cnt,
            'time': elapsed_time,
            'output': out_file,
            'success': True
        }
        
    except Exception as e:
        print(f"\n✗ 处理 {pdf_file} 时出错: {str(e)}")
        return {
            'file': pdf_file,
            'pages': 0,
            'time': 0,
            'output': None,
            'success': False,
            'error': str(e)
        }

# ---------------- 主程序入口 ----------------
if __name__ == "__main__":
    print("="*60)
    print("PDF批量OCR识别系统")
    print("="*60)
    
    # 1. 文件扫描
    print("\n[1/4] 扫描PDF文件...")
    pdf_files = sorted(glob.glob("*.pdf"))
    if not pdf_files:
        print("错误: 当前目录未找到PDF文件")
        sys.exit(1)
    
    print(f"找到 {len(pdf_files)} 个PDF文件:")
    for i, pdf in enumerate(pdf_files, 1):
        file_size = os.path.getsize(pdf) / (1024 * 1024)
        print(f"  {i}. {pdf} ({file_size:.2f} MB)")
    
    # 2. 鼠标画框
    print("\n[2/4] 请在弹出的窗口中绘制识别区域...")
    boxes = draw_boxes()
    
    # 3. 批量处理（使用智能多进程或手动设置）
    print(f"\n[3/4] 开始批量处理...")
    print("="*60)
    
    # 让用户选择核心模式
    while True:
        print("请选择核心选择模式:")
        print("1. 智能选择核心数 (根据系统资源自动调节)")
        print("2. 手动设置核心数")
        choice = input("请输入选择 (1/2，默认1): ").strip()
        
        if choice == "" or choice == "1":
            cores_to_use = None
            print("处理模式: 智能多进程 (根据系统资源自动调节)")
            break
        elif choice == "2":
            while True:
                try:
                    manual_cores = int(input("请输入要使用的核心数: ").strip())
                    if manual_cores > 0:
                        cores_to_use = manual_cores
                        print(f"处理模式: 手动设置核心数 - {manual_cores}核")
                        break
                    else:
                        print("错误: 核心数必须大于0")
                except ValueError:
                    print("错误: 请输入有效的数字")
            break
        else:
            print("错误: 请输入有效的选择 (1/2)")
    print("="*60)
    
    total_start_time = time.time()
    results = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        result = process_single_pdf_multiprocess(pdf_file, boxes, cores_to_use)
        results.append(result)
    
    # 结果汇总
    total_time = time.time() - total_start_time
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print(f"\n{'='*60}")
    print("处理汇总")
    print(f"{'='*60}")
    print(f"总文件数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {total_time:.1f}秒")
    if cores_to_use is not None:
        print(f"处理模式: 手动设置核心数 - {cores_to_use}核")
    else:
        print(f"处理模式: 智能多进程 (根据系统资源自动调节)")
    print(f"\n详细结果:")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        if result['success']:
            print(f"  {i}. {status} {result['file']} - {result['pages']}页, {result['time']:.1f}秒")
        else:
            print(f"  {i}. {status} {result['file']} - 错误: {result.get('error', '未知')}")
    
    print(f"{'='*60}")
    print("全部完成！")