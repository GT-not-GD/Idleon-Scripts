# main_server.py
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, SkipValidation
from contextlib import asynccontextmanager
import threading
import time
import json
import os
import subprocess
import cv2
import sys
import numpy as np
import win32gui
import win32con
import win32api
import pyautogui
import keyboard
import pytesseract
from PIL import Image
from pynput import mouse 
import base64 

# --- 导入您的 GameBot 和 GameFunctions 类 ---
# GameBot 和 GameFunctions 类定义在 'main.py' 文件中
try:
    from main import GameBot, GameFunctions, KEY_CODES, TESSERACT_PATH
except ImportError:
    print("错误: 无法导入 'main2.py'。请确保文件存在，并且 GameBot/GameFunctions 类已正确定义。")
    # 如果导入失败，我们在这里也抛出异常，避免后续代码因为找不到类而报错
    raise

# --- 全局变量 ---
game_bot_instance: GameBot | None = None 
game_functions_instance: GameFunctions | None = None
running_threads: dict[str, threading.Thread] = {} # 存储正在运行的自动化任务线程

# --- FastAPI 应用设置 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global game_bot_instance, game_functions_instance, running_threads # 确保所有全局变量都被声明
    try:
        window_configs = {'default': 'Legends Of Idleon'}
        print(f"Initializing GameBot with config: {window_configs}")
        game_bot_instance = GameBot(window_configs=window_configs) 
        
        if not game_bot_instance.hwnds:
            raise Exception("GameBot initialization failed: No window handles were found. Ensure the game is running and the window title in the config is correct.")
        
        if 'default' not in game_bot_instance.hwnds:
             print("Warning: Default window 'Legends Of Idleon' not found in configuration.")
        
        # 尝试访问 active_hwnd，确保它不因 'hwnd' 的缺失而出错
        _ = game_bot_instance.active_hwnd 

        game_functions_instance = GameFunctions(game_bot_instance)
        game_bot_instance.start_exit_listener()
        print("GameBot and GameFunctions initialized successfully.")
    except Exception as e:
        # ... (异常处理代码，与之前相同) ...
        print(f"--------------------------------------------------")
        print(f"FATAL ERROR during GameBot/GameFunctions initialization:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        if 'window_configs' in locals(): print(f"Initial window_configs: {window_configs}")
        if 'window_title' in locals(): print(f"Problematic Window Title: '{window_title}'")
        if game_bot_instance:
             print(f"GameBot instance created, but failed during further setup.")
             if hasattr(game_bot_instance, 'hwnds'): print(f"GameBot hwnds found: {game_bot_instance.hwnds}")
             if hasattr(game_bot_instance, 'active_window_key'):
                 print(f"GameBot active_window_key: {game_bot_instance.active_window_key}")
                 try:
                     active_hwnd_val = game_bot_instance.active_hwnd
                     print(f"GameBot active_hwnd: {active_hwnd_val}")
                 except AttributeError:
                     print(f"GameBot active_hwnd: AttributeError - Attribute 'hwnd' does not exist. Did you mean 'hwnds' or 'active_hwnd' property?")
        print(f"--------------------------------------------------")
        raise RuntimeError(f"Failed to initialize core components: {e}") from e
    
    yield # 应用程序在此处运行
    
    # Shutdown logic (if any)
    print("Application shutdown.")

app = FastAPI(title="Idleon Automation API", lifespan=lifespan)

# 定义允许的前端源列表
origins = [
    "http://localhost:5173",  # 你的 Vite React 应用地址
    "http://127.0.0.1:5173", # 有时浏览器会用 127.0.0.1
    # 如果你有其他前端地址，也在这里添加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源
    allow_credentials=True, # 是否支持 cookie
    allow_methods=["*"],    # 允许所有方法 (GET, POST, etc.)
    allow_headers=["*"],    # 允许所有请求头
)
# --- Pydantic 模型 ---
class ConfigUpdateModel(BaseModel):
    key: str
    value: SkipValidation 

# --- API 端点 ---
@app.get("/")
async def read_root():
    return {"message": "Idleon Automation API is running!"}

@app.get("/status")
async def get_status():
    """获取所有功能的运行状态"""
    global running_threads, game_functions_instance, game_bot_instance # !!! 再次声明全局变量 !!!
    
    status = {}
    if not game_functions_instance or not game_bot_instance: 
        # 如果初始化失败了，返回错误
        return {"error": "Core components not initialized"}
        
    # 动态获取所有可公开调用的方法
    all_function_names = [func_name for func_name in dir(game_functions_instance) 
                          if callable(getattr(game_functions_instance, func_name)) 
                          and not func_name.startswith('_') 
                          and func_name not in ['bot', '__init__', '__doc__', '__module__']] 
    
    # 使用全局的 running_threads 来检查状态
    for func_name in all_function_names:
        is_running = False
        # 访问全局的 running_threads
        if func_name in running_threads and running_threads[func_name].is_alive():
            is_running = True
        status[func_name] = {"running": is_running}

    return status

@app.post("/start/{function_name}")
async def start_function(function_name: str):
    """启动指定功能的自动化"""
    if not game_functions_instance:
        raise HTTPException(status_code=503, detail="Game functions not initialized")

    func_to_start = getattr(game_functions_instance, function_name, None)
    if func_to_start and callable(func_to_start):
        if function_name in running_threads and running_threads[function_name].is_alive():
            return {"message": f"Function '{function_name}' is already running."}

        # 为每个功能创建一个新的线程来运行
        thread = threading.Thread(target=func_to_start, daemon=True)
        running_threads[function_name] = thread
        thread.start()
        print(f"Started function '{function_name}' in thread {thread.ident}")
        return {"message": f"Started function '{function_name}'"}
    else:
        raise HTTPException(status_code=404, detail=f"Function '{function_name}' not found or not callable.")

@app.post("/stop/{function_name}")
async def stop_function(function_name: str):
    """停止指定功能的自动化"""
    if not game_bot_instance:
        raise HTTPException(status_code=503, detail="Game bot not initialized")

    if function_name == "all":
        game_bot_instance.running = False # 通知所有循环停止
        running_threads.clear() # 清理线程记录
        print("Signaled all functions to stop.")
        return {"message": "All functions signaled to stop."}
    
    elif function_name in running_threads and running_threads[function_name].is_alive():
        game_bot_instance.running = False # 仅设置为 False，让循环自行退出
        print(f"Signaled function '{function_name}' to stop.")
        # 不在此处 join，让线程自行退出
        return {"message": f"Signaled function '{function_name}' to stop."}
    else:
        return {"message": f"Function '{function_name}' is not running or not found."}

@app.post("/config/update")
async def update_config(config_update: ConfigUpdateModel):
    """更新配置文件 (仅模拟，实际更新逻辑需要实现)"""
    print(f"Received config update request: key='{config_update.key}', value='{config_update.value}', type='{type(config_update.value).__name__}'")
    # TODO: 实现实际的配置更新逻辑
    return {"message": f"Config update for '{config_update.key}' received (value type: {type(config_update.value).__name__}). Actual update logic needs implementation."}

@app.get("/screenshot/{window_key}")
async def get_screenshot(window_key: str = 'default', region: str = "full"):
    """
    获取指定窗口的截图，支持指定区域 (region)。
    region 格式: "x,y,w,h" (像素坐标)
    返回PNG格式的图片。
    """
    if not game_bot_instance:
        raise HTTPException(status_code=503, detail="Game bot not initialized")

    if window_key not in game_bot_instance.hwnds:
         raise HTTPException(status_code=404, detail=f"Window key '{window_key}' not found.")

    current_active_key = game_bot_instance.active_window_key
    try:
        game_bot_instance.active_window_key = window_key
        
        screenshot = None
        if region == "full":
            screenshot = game_bot_instance.get_screenshot(refresh=True) 
        else:
            try:
                x, y, w, h = map(int, region.split(','))
                full_screenshot = game_bot_instance.get_screenshot(refresh=True)
                if full_screenshot is not None:
                    img_height, img_width = full_screenshot.shape[:2]
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img_width - x)
                    h = min(h, img_height - y)
                    if w > 0 and h > 0:
                        screenshot = full_screenshot[y:y+h, x:x+w]
                    else:
                        raise ValueError("Invalid region dimensions or position.")
                else:
                    raise RuntimeError("Failed to capture full screenshot.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid region format or error during cropping: {e}")

        if screenshot is None:
            raise HTTPException(status_code=500, detail="Failed to capture screenshot.")

        # --- 2. 核心修正部分 ---
        # 将截图编码为 PNG 格式
        _, encoded_image = cv2.imencode('.png', screenshot)
        
        # 将编码后的图像字节转换为 bytes 对象
        image_bytes = encoded_image.tobytes()
        
        #直接返回一个 Response 对象，并指定媒体类型
        return Response(content=image_bytes, media_type="image/png")
        # -------------------------

    finally:
        game_bot_instance.active_window_key = current_active_key

def run_frontend_dev_server():
    """进入前端目录并执行 'npm run dev'"""
    # 获取当前脚本所在的目录
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建前端项目的路径 (假设它和后端脚本在同一个父目录下)
    frontend_dir = os.path.join(backend_dir, "my-idleon-ui")

    if not os.path.isdir(frontend_dir):
        print(f"!!! 错误：找不到前端目录: {frontend_dir}")
        print("请确保前端项目 'my-idleon-ui' 文件夹与 main_server.py 在同一目录下。")
        return

    print(f"\n--- 正在启动前端开发服务器 (Vite) ---")
    print(f"目录: {frontend_dir}")
    print("命令: npm run dev")
    print("-----------------------------------------\n")

    # 使用 subprocess.Popen 来执行命令
    # shell=True 在 Windows 上对于执行 npm/npx 命令更可靠
    # stdout=subprocess.PIPE 和 stderr=subprocess.PIPE 可以捕获输出，但为了简单，我们直接让它打印到控制台
    try:
        # 在 Windows 上，需要使用 shell=True 来正确找到 npm
        proc = subprocess.Popen(
            "npm run dev",
            cwd=frontend_dir, # 在指定目录下执行命令
            shell=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE # 可选：在一个新的控制台窗口中打开前端服务
        )
        proc.wait() # 等待进程结束
    except FileNotFoundError:
        print("!!! 严重错误：找不到 'npm' 命令。请确保你已经安装了 Node.js 并且 npm 在系统的 PATH 环境变量中。")
    except Exception as e:
        print(f"启动前端服务器时发生未知错误: {e}")


# --- 运行 FastAPI 应用 ---
if __name__ == "__main__":
    import uvicorn
    import webbrowser

    # 1. 创建并启动前端服务器线程
    frontend_thread = threading.Thread(target=run_frontend_dev_server, daemon=True)
    frontend_thread.start()
    
    # 2. 启动后端服务器 (在主线程中)
    print("\n--- 正在启动后端服务器 (FastAPI) ---")
    print("地址: http://127.0.0.1:8000")
    print("---------------------------------------\n")
    
    # 等待一小段时间，确保前端服务有时间启动
    print("等待前端服务启动...")
    time.sleep(5) # 等待 5 秒

    # 自动在浏览器中打开前端页面
    try:
        webbrowser.open("http://localhost:5173/")
    except Exception as e:
        print(f"自动打开浏览器失败: {e}")

    # uvicorn.run 会阻塞主线程，直到你按 Ctrl+C
    # 当主线程结束时，因为前端线程是守护线程(daemon=True)，它也会被自动终止
    uvicorn.run(app, host="127.0.0.1", port=8000)