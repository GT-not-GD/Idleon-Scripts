"""
Idleon 游戏自动化脚本
--------------------
本脚本用于自动化 Legends Of Idleon 游戏的多种日常操作。
依赖：OpenCV, pytesseract, pywin32, pyautogui, numpy, PIL, keyboard, msvcrt, json, threading, math, re, os, time

作者建议：
- 配置文件 config.json 需与本脚本同目录
- Tesseract OCR 路径需正确设置
- 仅支持 Windows 平台
"""

# 导入所需的库
import cv2  # 用于图像处理
import numpy as np  # 用于数值计算
import keyboard  # 用于键盘事件监听
import threading  # 用于多线程操作
import time  # 用于时间操作
import win32api  # 用于 Windows API 调用
import win32con  # 包含 Windows 常量
import win32gui  # 用于窗口操作
import win32ui  # 用于窗口设备上下文操作
import pyautogui  # 用于模拟鼠标和键盘操作
import re  # 用于正则表达式处理
import pytesseract  # 用于 OCR 文字识别
import os  # 用于文件和目录操作
import json  # 用于处理 JSON 配置文件
import msvcrt  # 用于控制台输入
import math  # 用于数学计算
from PIL import Image  # 用于图像处理
from pynput import mouse  # 用于鼠标事件监听
mouse_controller = mouse.Controller()  # 初始化鼠标控制器

# --- 常量定义 (Constants) ---
CONFIG_FILE_PATH = "config.json"  # 配置文件路径
TESSERACT_PATH = r'C:\All\Other\Tesseract-OCR\tesseract.exe'  # Tesseract OCR 引擎路径
THRESHOLD = 0.8  # 图像匹配阈值
SCREENSHOT_FOLDER = "screenshots"  # 截图保存文件夹

# --- 虚拟键码 (Virtual Key Codes) ---
# 定义键盘按键的虚拟键码，用于模拟按键操作
KEY_CODES = {
    'A': 0x41, 'B': 0x42, 'C': 0x43, 'D': 0x44, 'E': 0x45,
    'F': 0x46, 'G': 0x47, 'H': 0x48, 'I': 0x49, 'J': 0x4A,
    'K': 0x4B, 'L': 0x4C, 'M': 0x4D, 'N': 0x4E, 'O': 0x4F,
    'P': 0x50, 'Q': 0x51, 'R': 0x52, 'S': 0x53, 'T': 0x54,
    'U': 0x55, 'V': 0x56, 'W': 0x57, 'X': 0x58, 'Y': 0x59,
    'Z': 0x5A,
    '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
    '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
    'ESC': win32con.VK_ESCAPE,
    'SPACE': win32con.VK_SPACE,
    'F1': win32con.VK_F1,
    'F2': win32con.VK_F2,
    'F3': win32con.VK_F3,
    'F4': win32con.VK_F4,
    'F5': win32con.VK_F5,
    'F6': win32con.VK_F6,
    'F7': win32con.VK_F7,
    'F8': win32con.VK_F8,
    'F9': win32con.VK_F9,
    'F10': win32con.VK_F10,
    'F11': win32con.VK_F11,
    'F12': win32con.VK_F12,
    'LEFT': win32con.VK_LEFT,
    'RIGHT': win32con.VK_RIGHT,
    'UP': win32con.VK_UP,
    'DOWN': win32con.VK_DOWN
}

# --- GameBot: 游戏自动化核心类 ---
class GameBot:
    """
    游戏自动化脚本核心类，负责底层操作和配置管理
    """

    def __init__(self, game_window_name="Legends Of Idleon"):
        """
        初始化 GameBot 实例
        Args:
            game_window_name (str): 游戏窗口名称
        """
        # 查找游戏窗口句柄
        self.hwnd = win32gui.FindWindow(None, game_window_name)
        if not self.hwnd:
            raise Exception(f"未找到窗口: {game_window_name}")

        # 加载配置文件
        self.config = self._load_config(CONFIG_FILE_PATH)
        if not self.config:
            raise Exception("配置加载失败，程序退出。")

        # 设置 Tesseract OCR 路径
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

        # 确保截图文件夹存在
        os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

        # 初始化运行标志和缓存
        self.running = True
        self.image_cache = {}  # 缓存已加载的图像
        self.screenshot_cache = None  # 缓存最近的截图

    def _load_config(self, config_path):
        """
        加载配置文件
        Args:
            config_path (str): 配置文件路径
        Returns:
            dict: 配置文件内容
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_path} 未找到！")
            return None
        except json.JSONDecodeError:
            print(f"配置文件 {config_path} 格式错误！")
            return None

    def _load_image(self, image_path):
        """
        加载图像并缓存，避免重复加载
        Args:
            image_path (str): 图像文件路径
        Returns:
            numpy.ndarray: 加载的图像
        """
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"无法加载图像: {image_path}")
                return None
            self.image_cache[image_path] = image
            return image
        except Exception as e:
            print(f"加载图像失败: {e}")
            return None

    def get_screenshot(self, max_loc=None, x_offset=0, y_offset=0, width=None, height=None, refresh=False):
        """
        获取指定窗口区域的截图，支持缓存
        Args:
            max_loc (tuple): 参考位置 (x, y)
            x_offset (int): x 偏移量
            y_offset (int): y 偏移量
            width (int): 截图宽度
            height (int): 截图高度
            refresh (bool): 是否刷新截图缓存
        Returns:
            numpy.ndarray: 截图图像
        """
        # 如果不需要刷新且有缓存，直接返回缓存的截图
        if not refresh and self.screenshot_cache is not None:
            screenshot = self.screenshot_cache
        else:
            screenshot = self._capture_window()
            self.screenshot_cache = screenshot  # 缓存截图

        if screenshot is None:
            return None

        try:
            # 获取窗口矩形坐标
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            if max_loc:
                # 计算目标截图区域
                target_left = max_loc[0] + x_offset
                target_top = max_loc[1] + y_offset
                if width is None:
                    width = right - left - max_loc[0]
                if height is None:
                    height = bottom - top - max_loc[1]
            else:
                # 截取整个窗口
                target_left = 0
                target_top = 0
                width = right - left
                height = bottom - top

            # 提取截图区域
            cropped_image = screenshot[target_top:target_top + height, target_left:target_left + width]
            return cropped_image
        except Exception as e:
            print(f"获取截图出错: {e}")
            return None

    def _capture_window(self):
        """
        捕获整个窗口的截图
        Returns:
            numpy.ndarray: 截图图像
        """
        try:
            # 获取窗口矩形坐标
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top

            # 创建设备上下文和位图对象
            hdc = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hdc)
            saveDC = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

            # 获取位图数据并转换为图像
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            image = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

            # 释放资源
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hdc)

            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"窗口截图失败: {e}")
            return None

    def find_image_location(self, image_path, refresh_screenshot=False):
        """
        在游戏窗口截图中查找指定图像
        Args:
            image_path (str): 图像文件路径
            refresh_screenshot (bool): 是否刷新截图缓存
        Returns:
            tuple: 图像在窗口中的位置 (x, y)，如果未找到则返回 None
        """
        # 获取窗口截图
        screenshot = self.get_screenshot(refresh=refresh_screenshot)
        if screenshot is None:
            return None

        # 加载模板图像
        template = self._load_image(image_path)
        if template is None:
            return None

        # 转换截图为灰度图像并进行模板匹配
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 如果匹配值超过阈值，返回匹配位置
        if max_val >= THRESHOLD:
            return max_loc
        return None

    def interact_with_image(self, image_path, duration=0, refresh=False, click_times=1, x_offset=0, y_offset=0, delay=0, swipe=False):
        """
        与指定图像进行交互（点击或滑动）
        Args:
            image_path (str): 图像文件路径
            duration (float): 持续时间
            refresh (bool): 是否刷新截图
            click_times (int): 点击次数
            x_offset (int): x 坐标偏移量
            y_offset (int): y 坐标偏移量
            delay (float): 点击间隔
            swipe (bool): 是否为滑动操作
        Returns:
            bool: 是否成功交互
        """
        if image_path == self.config["breeding"]["breed_image"]:
            return self.interact_with_image(self.config["breeding"]["breed_01_image"])
        max_loc = self.find_image_location(image_path) # 查找图像位置
        if max_loc: # 如果找到图像
            template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 重新加载模板获取形状
            if template is None: # 再次检查模板加载
                print(f"模板图片 {image_path} 加载失败！无法交互。", flush=True)
                return False
            center_x = max_loc[0] + template.shape[1] // 2 + x_offset
            center_y = max_loc[1] + template.shape[0] // 2 + y_offset

            if swipe: # 滑动操作
                loot_y_offset = -50
                loot_swipe_width = 300
                swipe_steps = 10
                self._swipe_loot(center_x, center_y + loot_y_offset, loot_swipe_width, swipe_steps)
            elif refresh: # 刷新操作 (邮差)
                self._click_and_hold(*self.config["postman"]["refresh_coordinates"], duration=0.3)
                for _ in range(click_times):
                    self._click_and_hold(center_x, center_y, duration, x_offset=x_offset, y_offset=y_offset)
                    time.sleep(delay)
            else: # 普通点击操作
                for _ in range(click_times):
                    self._click_and_hold(center_x, center_y, duration, x_offset=x_offset, y_offset=y_offset)
                    time.sleep(delay)
            return True
        return False

    def _click_and_hold(self, x, y, duration=0, click_times=1, x_offset=0, y_offset=0):
        """
        模拟鼠标点击并按住指定时间
        Simulate mouse click and hold.
        Args:
            x: 点击的 x 坐标。
            y: 点击的 y 坐标。
            duration: 按住持续时间 (秒)。
            click_times: 点击次数。
            x_offset: x 坐标偏移量。
            y_offset: y 坐标偏移量。
        """
        target_x = int(x + x_offset) # 计算目标点击 x 坐标
        target_y = int(y + y_offset) # 计算目标点击 y 坐标
        lParam = win32api.MAKELONG(target_x, target_y) # 合并 x, y 坐标为 lParam
        for _ in range(click_times): # 重复点击次数
            start_time = time.time()
            while time.time() - start_time < duration: # 按住持续时间
                win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam) # 发送鼠标左键按下消息
                time.sleep(0.01)
            win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, lParam) # 发送鼠标左键释放消息
            if click_times > 1: # 多次点击的延迟
                time.sleep(0.1)

    def _swipe_loot(self, center_x, center_y, width, steps):
        """
        模拟鼠标滑动 (用于物品拾取)
        Simulate mouse swipe for loot.
        Args:
            center_x: 中心 x 坐标。
            center_y: 中心 y 坐标。
            width: 滑动宽度。
            steps: 滑动步骤数。
        """
        start_x = center_x - width // 2 # 计算滑动起始 x 坐标
        end_x = center_x + width // 2 # 计算滑动结束 x 坐标
        x_step = (end_x - start_x) / steps # 计算每步 x 轴移动距离

        lParam_start = win32api.MAKELONG(start_x, center_y) # 滑动起始位置 lParam
        win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam_start) # 鼠标左键按下

        for i in range(1, steps + 1): # 平滑滑动 steps 步
            x = int(start_x + i * x_step) # 计算当前步 x 坐标
            lParam = win32api.MAKELONG(x, center_y) # 当前位置 lParam
            win32api.PostMessage(self.hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, lParam) # 发送鼠标移动消息
            time.sleep(0.01)

        lParam_end = win32api.MAKELONG(end_x, center_y) # 滑动结束位置 lParam
        win32api.PostMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, lParam_end) # 鼠标左键释放

    def preprocess_image(self, image):
        """
        预处理图像以增强白色数字识别效果
        Preprocess image for white number OCR.
        Args:
            image: 输入图像，BGR 格式。
        Returns:
            numpy.ndarray: 处理后的图像，适合 OCR 识别。
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 转换为 HSV 色彩空间
        lower_white = np.array([0, 0, 200]) # 白色下限
        upper_white = np.array([180, 25, 255]) # 白色上限
        mask = cv2.inRange(hsv, lower_white, upper_white) # 创建白色掩模
        result = cv2.bitwise_and(image, image, mask=mask) # 应用掩模，提取白色区域
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # 转换为灰度图像
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 使用 Otsu 阈值法进行二值化
        return thresh

    def read_game_number(self, image_path, x_offset, y_offset, width, height, save_screenshot=False):
        """
        使用 OCR 读取游戏画面指定区域的数字
        OCR read number from game region.
        Args:
            image_path: 参考图像文件路径 (用于定位区域)。
            x_offset:  OCR 区域的 x 偏移量。
            y_offset:  OCR 区域的 y 偏移量。
            width:     OCR 区域的宽度。
            height:    OCR 区域的高度。
            save_screenshot: 是否保存截图用于调试。
        Returns:
            int: 读取到的数字，如果读取失败返回 None。
        """
        try:
            max_loc = self.find_image_location(image_path) # 查找参考图像位置
            if not max_loc:
                print(f"无法找到参考图像 {image_path}，OCR 读取失败。", flush=True)
                return None
            screenshot = self.get_screenshot(max_loc, x_offset, y_offset, width, height) # 截取 OCR 区域截图
            if screenshot is None:
                return None

            im = cv2.resize(screenshot, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC) # 放大图像
            im = self.preprocess_image(im) # 预处理图像，增强数字识别
            pil_image = Image.fromarray(im) # 转换为 PIL Image for OCR

            if save_screenshot: # 保存截图
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_filename = os.path.join(SCREENSHOT_FOLDER, f"screenshot_{timestamp}")
                pil_original_screenshot = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)) # 原始截图 for save
                pil_original_screenshot.save(f"{base_filename}_original.png")
                pil_image.save(f"{base_filename}_processed.png")

            text = pytesseract.image_to_string(pil_image, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,KMB') # 使用 Tesseract OCR 识别数字
            match = re.search(r"([\d\.,]+)([KMB])?", text.replace(" ", ""), re.IGNORECASE) # 使用正则表达式提取数字
            if match: # 如果找到匹配的数字
                number_str = match.group(1).replace(",", "") # 去除千位分隔符逗号
                unit = match.group(2) # 这将是 K, M, B, 或 None
                try:
                    number = float(number_str) # 转换为浮点数
                    if unit:
                        unit = unit.upper()
                        if unit == 'K':
                            number *= 1000
                        elif unit == 'M':
                            number *= 1000000
                        elif unit == 'B':
                            number *= 1000000000
                    return int(number) # 返回整数
                except ValueError:
                    print(f"OCR 结果无法转换为数字: {text}", flush=True)
                    return None
            else:
                print(f"OCR 未找到数字: {text}", flush=True)
                return None

        except Exception as e:
            print(f"OCR 出错: {e}", flush=True)
            return None

    def press_key(self, key, duration=0):
        """
        模拟按键按下和释放
        Simulate key press and release.
        Args:
            key: 按键名称或虚拟键码。
            duration: 按住时间 (秒)，默认为 0。
        """
        vk_code = KEY_CODES.get(key) if isinstance(key, str) else key # 允许传入键名字符串或虚拟键码
        if vk_code is None:
            print(f"未知的按键: {key}", flush=True)
            return

        if duration > 0: # 按住一段时间
            start_time = time.time()
            while time.time() - start_time < duration:
                win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, 0) # 按下按键
                time.sleep(0.05)
                win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, vk_code, 0) # 释放按键
                time.sleep(0.05)
        else: # 短按一下
            win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, 0) # 按下按键
            time.sleep(0.2) # 保持短暂延迟
            win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, vk_code, 0) # 释放按键

    def timed_input(self, prompt, timeout=2, timeout_msg=None):
        """
        带超时的用户输入函数
        """
        print(prompt, end='', flush=True)
        input_str = ''
        start_time = time.time()
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char == '\r':
                    print()
                    break
                elif char == '\x08':
                    if len(input_str) > 0:
                        input_str = input_str[:-1]
                        print('\b \b', end='', flush=True)
                elif char == '\x03':
                    raise KeyboardInterrupt
                else:
                    input_str += char
            if time.time() - start_time > timeout:
                if timeout_msg:
                    print(timeout_msg, flush=True)
                else:
                    print(f"\n在{timeout}秒内没有输入，使用默认值。", flush=True)
                return None
            time.sleep(0.01)
        return input_str.strip()

    def exit_handler(self):
        """
        监听 'delete' 键退出脚本的线程函数
        Listen for 'delete' key to exit script.
        """
        while True:
            if keyboard.is_pressed('delete'): # 检测是否按下 'delete' 键
                print("检测到 'delete' 键，退出脚本。", flush=True)
                self.running = False # 设置脚本运行标志为 False，停止循环
                break
            time.sleep(0.1) # 短暂休眠

    def start_exit_listener(self):
        """
        启动退出监听线程
        Start exit listener thread.
        """
        exit_thread = threading.Thread(target=self.exit_handler) # 创建退出监听线程
        exit_thread.daemon = True # 设置为守护线程，主线程退出时自动退出
        exit_thread.start() # 启动线程

    def get_client_mouse_position(self):
        """
        获取鼠标在游戏窗口内的相对位置。
        Returns:
            tuple: (x, y) 鼠标在窗口内的相对坐标，如果失败则返回 (None, None)。
        """
        try:
            # 获取鼠标的屏幕坐标
            cursor_x, cursor_y = win32api.GetCursorPos()

            # 获取游戏窗口的矩形坐标
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)

            # 检查鼠标是否在窗口范围内
            if left <= cursor_x <= right and top <= cursor_y <= bottom:
                # 计算鼠标在窗口内的相对坐标
                relative_x = cursor_x - left
                relative_y = cursor_y - top
                return relative_x, relative_y
            else:
                return None, None
        except Exception as e:
            print(f"获取鼠标位置失败: {e}", flush=True)
            return None, None

    def calculate_fishing_power_bar_percent(self, fishbar_percent):
        """
        根据鱼的位置百分比计算力量条的百分比。
        Args:
            fishbar_percent (float): 鱼的位置百分比 (0.0 到 1.0)。
        Returns:
            float: 力量条的百分比 (0.0 到 1.0)。
        """
        power_percent = (0.4664 * fishbar_percent**3 -
                         1.0693 * fishbar_percent**2 +
                         1.603 * fishbar_percent +
                         0.0036)
        return max(min(power_percent, 1.0), 0.0)  # 限制在 [0, 1] 区间

    def calculate_fishing_power_bar_click_duration(self, powerbar_bars):
        """
        根据力量条的格数计算鼠标按住的持续时间。
        Args:
            powerbar_bars (float): 力量条的格数。
        Returns:
            int: 鼠标按住的持续时间 (毫秒)。
        """
        hold_duration = (2.268 * powerbar_bars**3 -
                         32.207 * powerbar_bars**2 +
                         232.423 * powerbar_bars +
                         82.220)
        return min(max(hold_duration, 20.0), 900.0)  # 限制在 [20ms, 900ms] 区间

    def hold_left_mouse_ms_at_current_pos(self, duration_ms):
        """
        按住鼠标左键在当前鼠标位置指定毫秒数。
        Args:
            duration_ms (int): 按住时间（毫秒）。
        """
        mouse_controller.press(mouse.Button.left)
        time.sleep(duration_ms / 1000.0)  # 将毫秒转换为秒
        mouse_controller.release(mouse.Button.left)

# --- GameFunctions: 游戏功能函数集合 ---
class GameFunctions:
    """
    游戏功能函数集合，依赖 GameBot 核心类
    Collection of game automation functions.
    """
    def __init__(self, game_bot):
        """
        初始化 GameFunctions 实例，关联 GameBot 实例
        Args:
            game_bot (GameBot): GameBot 实例
        """
        self.bot = game_bot # 关联 GameBot 实例
        self.mister_bribe_quest_status = 0 #  假设Mister Bribe quest 状态(未激活) - 用于测试
        self.dismiss_item_available_status = True # 假设dismiss item 数量 - 用于测试 
        self.current_coins = 80 # 假设金币数量 - 用于测试
        self.current_case_number = 61 # 假设当前案子 - 用于测试
        self._fishing_listener = None
        self.fishing_config = self.bot.config.get("fishing", {})

    # --- AFK 快进 ---
    def afk_fast_forward_loop(self):
        """
        AFK 快进循环
        AFK fast forward loop.
        """
        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.2)
                self.bot.interact_with_image(self.bot.config["afk"]["candy_image"], duration=0.3) # 使用糖果
                time.sleep(0.7)
                self.bot.interact_with_image(self.bot.config["afk"]["storage_image"]) # 领取奖励
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["afk"]["gt_leg_image"], swipe=True) # 滑动 GT 腿
                time.sleep(0.5)
        except Exception as e:
            print(f"afk_fast_forward_loop 异常: {e}", flush=True)

    # --- AFK 基因萃取 ---
    def afk_gene_extract_loop(self):
        """
        AFK 基因萃取循环
        AFK gene extract loop.
        """
        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.2)
                self.bot.interact_with_image(self.bot.config["afk"]["candy_image"], duration=0.3) # 使用糖果
                time.sleep(0.7)
                self.bot.interact_with_image(self.bot.config["afk"]["gain_image"], x_offset=-50, y_offset=30, click_times=2) # 点击增益按钮
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["afk"]["claim_image"]) # 领取奖励
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["afk"]["gt_leg_image"], swipe=True) # 滑动 GT 腿
                time.sleep(0.2)
        except Exception as e:
            print(f"afk_gene_extract_loop 异常: {e}", flush=True)

    # --- 航海 ---
    def sailing_loop(self):
        """
        航海循环
        Sailing loop.
        """
        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.2)
                self.bot.interact_with_image(self.bot.config["sailing"]["boost_image"], duration=0.3) # 使用航海加速
                time.sleep(0.2)
                self.bot.press_key('ESC') # ESC 退出
                time.sleep(0.2)
                self.bot.press_key('W') # 按 W 键
                time.sleep(0.2)
                self.bot.press_key('SPACE') # 按 SPACE 键
                time.sleep(4.5)
                self.bot.press_key('ESC') # ESC 退出
                time.sleep(0.5)
                self.bot._click_and_hold(*self.bot.config["sailing"]["treasure_position"]) # 点击宝箱位置
                time.sleep(0.1)
                for _ in range(6):
                    self.bot._click_and_hold(*self.bot.config["sailing"]["chest_position"], duration=0.2) # 连续点击宝箱
        except Exception as e:
            print(f"sailing_loop 异常: {e}", flush=True)

    # --- 邮差 ---
    def postman_loop(self):
        """
        邮差循环
        Postman loop.
        """
        try:
            while self.bot.running:
                self.bot.interact_with_image(self.bot.config["postman"]["postman_image"], refresh=True) # 刷新邮差任务
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["postman"]["sign_image"]) # 签署邮差任务
                time.sleep(0.5)
        except Exception as e:
            print(f"postman_loop 异常: {e}", flush=True)

    # --- 地牢骰子 ---
    def dungeon_lootroll_loop(self):
        """
        地牢骰子循环
        Dungeon loot roll loop.
        """
        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["dungeon"]["dice_image"], duration=0.3) # 使用骰子
                time.sleep(0.5)
                if self.bot.interact_with_image(self.bot.config["dungeon"]["lootroll_image"]): # 尝试 Loot Roll
                    continue
        except Exception as e:
            print(f"dungeon_lootroll_loop 异常: {e}", flush=True)

    # --- 养殖 ---
    def breeding_loop(self):
        """
        养殖循环
        Breeding loop.
        """
        print("请选择要养殖的宠物：")
        choice = input("请输入选项: ")
        if choice not in self.bot.config["breeding"]["pets"]:
            print("无效选项！", flush=True)
            return

        pet_image_path = self.bot.config["breeding"]["pets"][choice] # 获取宠物图片路径

        print("是否孵化闪亮宠物？\n1. 是\n2. 否")
        shiny_choice = input("请输入选项 (1-2): ")
        hatch_shiny = shiny_choice == "1" # 是否孵化闪亮宠物

        print("请选择操作：\n1. KEEP\n2. TRASH")
        action_choice = input("请输入选项 (1-2): ")
        if action_choice == "1":
            action_image = self.bot.config["breeding"]["keep_image"] # 选择 KEEP 图片
            try:
                min_ability = int(input("请输入能力最小值: ")) # 输入最小能力值
            except ValueError:
                print("请输入有效的数字！", flush=True)
                return
        elif action_choice == "2":
            action_image = self.bot.config["breeding"]["trash_image"] # 选择 TRASH 图片
        else:
            print("无效选项！", flush=True)
            return

        try:
            while self.bot.running:
                self.bot.press_key('W') # 按 W 键
                time.sleep(0.2)
                self.bot.press_key('SPACE') # 按 SPACE 键
                time.sleep(0.5)
                self.bot.interact_with_image(pet_image_path) # 选择宠物
                time.sleep(0.2)
                if hatch_shiny:
                    self.bot.interact_with_image(self.bot.config["breeding"]["shiny_form_image"], click_times=2) # 选择闪亮形态

                while self.bot.interact_with_image(self.bot.config["breeding"]["breed_image"]): # 持续点击 BREED
                    time.sleep(0.2)
                    if hatch_shiny:
                        self.bot.interact_with_image(self.bot.config["breeding"]["keep_image"]) # 闪亮宠物始终 KEEP
                        continue

                    if action_image == self.bot.config["breeding"]["keep_image"]: # 如果选择 KEEP
                        image_path = self.bot.config["breeding"]["breed_01_image"] # 能力值图片路径
                        ability_value = self.bot.read_game_number(
                            image_path, 266, 60, 83, 22, save_screenshot=False) # 读取能力值
                        if ability_value is None:
                            self.bot.interact_with_image(self.bot.config["breeding"]["trash_image"]) # 读取失败 TRASH
                        elif ability_value >= min_ability:
                            print(f"能力值 {ability_value} 达标，执行 KEEP", flush=True)
                            self.bot.interact_with_image(self.bot.config["breeding"]["keep_image"]) # 能力达标 KEEP
                        else:
                            print(f"能力值 {ability_value} 不达标，执行 TRASH", flush=True)
                            self.bot.interact_with_image(self.bot.config["breeding"]["trash_image"]) # 能力不达标 TRASH
                    else:
                        self.bot.interact_with_image(self.bot.config["breeding"]["trash_image"]) # 否则 TRASH
                    time.sleep(0.2)

                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.2)
                self.bot.interact_with_image(self.bot.config["breeding"]["egg_image"], duration=0.3, click_times=30, delay=0.2) # 使用蛋
                time.sleep(0.2)
                self.bot.press_key('ESC') # ESC 退出
                time.sleep(0.2)
        except Exception as e:
            print(f"breeding_loop 异常: {e}", flush=True)

    # --- 游戏 ---
    def gaming_loop(self):
        """
        游戏循环
        Gaming loop.
        """
        config = self.bot.config["gaming"] # 获取游戏配置
        base_x, base_y = config["evo_base_position"] # 起始位置
        end_x, end_y = config["evo_end_position"] # 结束位置
        num_cols, num_rows = 10, 8 # 固定行列数，配置中可调整

        x_offset = (end_x - base_x) / (num_cols - 1) if num_cols > 1 else 0 # 计算 x 轴偏移量
        y_offset = (end_y - base_y) / (num_rows - 1) if num_rows > 1 else 0 # 计算 y 轴偏移量

        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.5)
                self.bot.interact_with_image(config["boost_image"], duration=0.3) # 使用加速
                time.sleep(0.2)
                self.bot._click_and_hold(*self.bot.config["default"]["codex_position"]) # 点击 Codex
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["default"]["quik_ref"]) # 点击 Quik Ref
                time.sleep(0.5)
                self.bot.interact_with_image(config["pc"]) # 点击 PC
                time.sleep(0.5)
                
                if self.bot.find_image_location(config["harvest"]): # 验证是否发现 ["harvest"] 图片
                    self.bot._click_and_hold(*config["harvest_position"]) # 点击 Harvest
                    time.sleep(0.5)
                    self.bot._click_and_hold(*config["shovel_position"]) # 点击 Shovel
                    time.sleep(0.5)

                    for row in range(num_rows): # 遍历行
                        for col in range(num_cols): # 遍历列
                            calculated_x = int(base_x + col * x_offset) # 计算点击 x 坐标
                            calculated_y = int(base_y + row * y_offset) # 计算点击 y 坐标
                            self.bot._click_and_hold(calculated_x, calculated_y) # 点击格子
                            time.sleep(0.02)
                else:
                    print("Harvest image not found, skipping harvest and shovel operations.", flush=True) # 如果没有发现 harvest 图片，打印信息并跳过操作
        except Exception as e:
            print(f"gaming_loop 异常: {e}", flush=True)

    # --- Boss ---
    def boss_loop(self):
        """
        Boss 循环
        Boss loop.
        """
        n = 0
        try:
            while self.bot.running:
                max_loc = self.bot.find_image_location(self.bot.config["boss"]["boss_image"]) # 查找 Boss 图片
                if max_loc: # 如果找到 Boss
                    template = cv2.imread(self.bot.config["boss"]["boss_image"], cv2.IMREAD_GRAYSCALE) # 加载 Boss 模板
                    center_x = max_loc[0] + template.shape[1] // 2 # Boss 中心 x 坐标
                    center_y = max_loc[1] + template.shape[0] // 2 # Boss 中心 y 坐标
                    time.sleep(5) # 等待 5 秒
                    self.bot._click_and_hold(center_x, center_y, duration=0.3) # 点击 Boss
                    n += 1
                    print(f"boss kill:{n}", flush=True)
                time.sleep(1) # 循环间隔 1 秒
        except Exception as e:
            print(f"boss_loop 异常: {e}", flush=True)

    # --- 开箱 ---
    def open_loop(self):
        """
        开箱循环
        Open box loop.
        """
        print("请选择开箱模式：")
        print("1. 不退出背包版")
        print("2. 退出背包版")
        open_choice = input("请输入选项 (1-2): ")

        if open_choice == '1':
            exit_inventory = False
        elif open_choice == '2':
            exit_inventory = True
        else:
            print("无效选项，默认选择不退出背包版。", flush=True)
            exit_inventory = False

        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["candy_position"], duration=0.3) # 点击糖果位置
                if exit_inventory:
                    time.sleep(0.5) # 稍长间隔 for exiting inventory
                    self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置 to exit inventory
                    time.sleep(0.1)
                else:
                    time.sleep(0.1) # 短暂间隔 for not exiting inventory
        except Exception as e:
            print(f"open_loop 异常: {e}", flush=True)

    # --- 猫头鹰 ---
    def owl_loop(self):
        """
        猫头鹰循环
        Owl loop.
        """
        owl_images = [self.bot.config["owl"][f"owl_0{i}_image"] for i in range(1, 10)] # 动态生成猫头鹰图像路径列表
        try:
            while self.bot.running:
                for image_path in owl_images: # 遍历猫头鹰图像路径
                    self.bot.interact_with_image(image_path, duration=3) # 与每个猫头鹰图像交互
        except Exception as e:
            print(f"owl_loop 异常: {e}", flush=True)

    # --- 农场 ---
    def farming_loop(self):
        """
        农场循环
        Farming loop.
        """
        try:
            while self.bot.running:
                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(1)
                self.bot.interact_with_image(self.bot.config["farming"]["boost_image"], duration=0.3)# 使用农场加速（避免卡着）
                time.sleep(0.2)
                self.bot.interact_with_image(self.bot.config["farming"]["boost_image"], duration=0.3, click_times=30, delay=0.1) # 使用农场加速
                time.sleep(0.2)
                self.bot.press_key('ESC') # ESC 退出
                time.sleep(0.5) # 0.5 秒等待
                self.bot._click_and_hold(*self.bot.config["farming"]["farm_position"]) # 点击农场位置
                time.sleep(0.5) # 0.5 秒等待
                self.bot.interact_with_image(self.bot.config["farming"]["collectall_image"]) # 点击 Collect All
                time.sleep(0.2) # 短暂间隔
        except Exception as e:
            print(f"farming_loop 异常: {e}", flush=True)

    # --- 勇气 ---
    def paying_loop(self):
        """
        勇气循环
        Paying loop.
        """
        num_fights = 60 # 默认战斗次数
        try:
            num_fights = int(input(f"请输入战斗次数 (默认 {num_fights}): ") or num_fights) # 输入战斗次数
        except ValueError:
            print("请输入数字", flush=True)
            return

        fight_count = 0
        try:
            while self.bot.running:
                print(f"开始新一轮战斗循环，总战斗次数: {num_fights}", flush=True)
                fight_count = 0

                self.bot._click_and_hold(*self.bot.config["default"]["item_position"]) # 点击默认物品位置
                time.sleep(0.5)
                self.bot.interact_with_image(self.bot.config["paying"]["boost_image"], duration=0.3, click_times=15, delay=0.2) # 使用洞穴加速
                time.sleep(0.5)
                self.bot.press_key('ESC') # ESC 退出
                time.sleep(0.5)
                self.bot._click_and_hold(*self.bot.config["paying"]["head_position"], duration=0.1) # 点击头部
                time.sleep(0.5)
                self.bot._click_and_hold(*self.bot.config["paying"]["story_position"], duration=0.1) # 点击 Story
                time.sleep(1)

                while fight_count < num_fights: # 战斗循环
                    print(f"开始第 {fight_count + 1} 次战斗", flush=True)
                    self.bot._click_and_hold(*self.bot.config["paying"]["fight_position"])
                    time.sleep(4.5) # 等待画面完全加载

                    if self.bot.find_image_location(self.bot.config["paying"]["victory"]):
                        print("检测到战斗直接胜利，跳过剑法释放。", flush=True)
                        time.sleep(1) # 等待画面完全加载
                    else:
                        print("未检测到直接胜利，释放剑法...", flush=True)
                        sword_positions = [ # 剑法位置列表
                            self.bot.config["paying"]["sword_1_position"],
                            self.bot.config["paying"]["sword_2_position"],
                            self.bot.config["paying"]["sword_3_position"],
                            self.bot.config["paying"]["sword_4_position"],
                            self.bot.config["paying"]["sword_5_position"],
                            self.bot.config["paying"]["sword_6_position"],
                            self.bot.config["paying"]["sword_7_position"]
                        ]
                        for pos in sword_positions: # 依次点击所有剑法
                            self.bot._click_and_hold(*pos)
                            time.sleep(0.1)
                        time.sleep(6) # 等待战斗结束

                    self.bot._click_and_hold(*self.bot.config["paying"]["treasure_1_position"])
                    time.sleep(3)
                    fight_count += 1

                print("本轮战斗循环结束，准备跑路", flush=True)
                self.bot._click_and_hold(*self.bot.config["paying"]["run_position"])
                time.sleep(5)
        except Exception as e:
            print(f"paying_loop 异常: {e}", flush=True)

    # --- 竞技场 ---
    def colo_loop(self):
        """
        竞技场循环, 支持 W3 和 W5, 强制用户选择世界
        Colosseum loop (W3/W5/W6).
        """
        world_choice = None
        while world_choice not in ['3', '5', '6']:
            print("请选择竞技场世界：")
            print("3. 世界 3 竞技场")
            print("5. 世界 5 竞技场")
            print("6. 世界 6 竞技场")
            world_choice_input = input("请输入选项 (3/5/6): ")
            if world_choice_input in ['3', '5', '6']:
                world_choice = world_choice_input
            else:
                print("无效选项，目前只支持世界 3 ， 世界 5 和 世界 6。", flush=True)

        if world_choice == '3':
            colo_enter_image = self.bot.config["colo"]["colo_w3"]
            colo_exit_action = "coordinate" # W3 exit is coordinate
            colo_exit_location = self.bot.config["colo"]["colo_exit_w3"]
            colo_w3_bone_image = self.bot.config["colo"]["colo_w3_bone"] # W3 特殊怪触发装置图像路径
        elif world_choice == '5':
            colo_enter_image = self.bot.config["colo"]["colo_w5"]
            colo_exit_action = "image" # W5 exit is image
            colo_exit_location = self.bot.config["colo"]["colo_exit_w5"]
        elif world_choice == '6':
            colo_enter_image = self.bot.config["colo"]["colo_w6"]
            colo_exit_action = "map" # W6 exit is map
            colo_exit_location = self.bot.config["colo"]["colo_exit_w6"]

        try:
            while self.bot.running:
                self.bot.interact_with_image(colo_enter_image) # 选择竞技场世界
                time.sleep(2)
                self.bot.interact_with_image(self.bot.config["colo"]["colo_enter"]) # 进入竞技场
                time.sleep(2)

                # 验证是否进入竞技场 (超时机制)
                start_time = time.time()
                colo_start_found = False
                while time.time() - start_time < 10: # 10 秒超时
                    if self.bot.find_image_location(self.bot.config["colo"]["colo_start"]): # 查找关卡开始标志
                        colo_start_found = True
                        print("成功进入竞技场，关卡开始!", flush=True)
                        break
                    time.sleep(0.5) # 0.5 秒检查间隔

                if not colo_start_found:
                    print("进入竞技场超时，未能检测到关卡开始标志。", flush=True)
                    continue

                # 检查自动战斗状态并打开
                if self.bot.find_image_location(self.bot.config["default"]["auto_off"]): # 查找自动战斗关闭标志
                    print("自动战斗为关闭状态，正在打开...", flush=True)
                    self.bot.interact_with_image(self.bot.config["default"]["auto_off"]) # 点击关闭按钮打开自动战斗
                    time.sleep(1) # 等待状态切换
                elif self.bot.find_image_location(self.bot.config["default"]["auto_on"]): # 查找自动战斗开启标志
                    print("自动战斗已打开。", flush=True)
                else:
                    print("无法检测自动战斗状态。", flush=True)
                    continue

                # 循环点击下一关卡坐标直到关卡结束
                while self.bot.running:
                    if world_choice == '3' and colo_w3_bone_image:
                        self.bot.interact_with_image(colo_w3_bone_image, duration=0.3, click_times=2) # 点击触发装置
                        
                    self.bot._click_and_hold(*self.bot.config["colo"]["colo_next_position"], click_times=10) # 循环点击下一关卡坐标
                    time.sleep(0.1) # 点击间隔

                    colo_end_location = self.bot.find_image_location(self.bot.config["colo"]["colo_end"]) # 查找关卡结束标志
                    if colo_end_location: # 如果找到关卡结束标志
                        end_number = self.bot.read_game_number(
                            self.bot.config["colo"]["colo_end"], 51, 28, 80, 25, save_screenshot=False) # OCR 读取剩余时间
                        if end_number == 0: # 如果剩余时间为 0，关卡结束
                            print("关卡结束。", flush=True)
                            # 关闭自动战斗
                            if self.bot.find_image_location(self.bot.config["default"]["auto_on"]): # 查找自动战斗开启标志
                                print("正在关闭自动战斗...", flush=True)
                                self.bot.interact_with_image(self.bot.config["default"]["auto_on"]) # 点击开启按钮关闭自动战斗
                                time.sleep(1)
                                print("自动战斗已关闭。", flush=True)
                            if colo_exit_action == "image":
                                self.bot.interact_with_image(colo_exit_location) # 点击退出竞技场 (Image for W5)
                            elif colo_exit_action == "coordinate":
                                self.bot._click_and_hold(*colo_exit_location) # 点击退出竞技场 (Coordinate for W3)
                            elif colo_exit_action == "map":
                                self.bot._click_and_hold(*self.bot.config["default"]["map_position"]) 
                                time.sleep(0.2)
                                self.bot._click_and_hold(*self.bot.config["default"]["map_world_6_position"]) 
                                time.sleep(0.2)
                                self.bot._click_and_hold(*colo_exit_location) # 点击退出竞技场
                                time.sleep(0.2)
                                self.bot._click_and_hold(*self.bot.config["default"]["map_teleport"]) 
                            time.sleep(2) # 等待退出完成
                            break # 跳出关卡循环
                        else:
                            print(f"关卡剩余时间: {end_number}, 关卡未结束，继续刷怪。", flush=True)
                    if not self.bot.running: # 如果脚本停止运行，跳出循环
                        break
                print("竞技场循环结束。", flush=True)
                if not self.bot.running: # 如果脚本停止运行，跳出主循环
                    break
                print("等待下一轮竞技场...", flush=True)
                time.sleep(2) # 等待下一轮竞技场开始
        except Exception as e:
            print(f"colo_loop 异常: {e}", flush=True)

    # --- 正义 ---
    def justice_loop(self):
        """
        正义循环
        Justice loop.
        """
        def get_current_mental_health():
            """
            **需要实现: 获取当前游戏角色的精神健康值。**
            """
            # 1. 截图精神健康值区域
            # 2. 使用 self.bot.read_game_number() 读取数值
            # 3. 返回读取到的精神健康值 (int)

            return 2  # 占位值，你需要替换成实际获取精神健康值的代码
        
        def get_mister_bribe_quest_status():
            """
            **需要实现: 获取 Mister Bribe quest 的状态。**
            """
            return self.mister_bribe_quest_status 
        
        def get_dismiss_item_available_status():
            """
            **需要实现: 获取 dismiss item available 状态。**
            """
            return self.dismiss_item_available_status
        
        def get_current_coins():
            """
            **需要实现: 获取当前游戏角色的 coins 数量。**
            """
            return self.current_coins #  返回当前 coins 数量
        
        def get_current_case_number():
            """
            **需要实现: 获取当前游戏进行到的案子编号。**
            """
            return self.current_case_number # 返回当前案子编号
        
        def calculate_approve_coin_change(current_case_num, approve_court_coins_base):
            """
            计算批准案件的金币变化量，考虑负 approve_court_coins_base 的最小消费为base值的设定。

            Args:
                current_case_num (int): 当前案子编号.
                approve_court_coins_base (int): approve_court_coins_base 配置值.

            Returns:
                int: 批准案件的金币变化量 (可能为负数).
            """
            coin_change = math.ceil(current_case_num * abs(approve_court_coins_base) / 5)
            
            if approve_court_coins_base < 0: # 如果 approve_court_coins_base 是负数
                coin_change = - coin_change # 赋予负数
                if abs(coin_change) < abs(approve_court_coins_base): # 如果向上取整后结果是小于base值
                    coin_change = approve_court_coins_base # 最小消费设定为base值

            return coin_change
        
        def calculate_coin_pop_ratio(case_config, current_case_num, current_coins):
            """
            计算 little_timmy_2 case 的金币/人气性价比。

            Args:
                case_config (dict): little_timmy_2 case 的配置信息.
                current_case_num (int): 当前案子编号.
                current_coins (int): 当前金币数量.

            Returns:
                float: 金币/人气性价比 (coin_pop_ratio). 返回 None 表示无法计算或出现错误.
            """
            approve_coins_base = case_config.get("approve_court_coins_base", 0)
            approve_popularity_base = case_config.get("approve_popularity_base", 0)
            zero_coins_approve = case_config.get("zero_coins_approve", False)

            if approve_coins_base is None or approve_popularity_base is None:
                print("Error: approve_court_coins_base 或 approve_popularity_base 未配置。", flush=True)
                return None

            # 计算理论金币消耗 (可能为负数，表示金币减少)
            theoretical_coin_change = calculate_approve_coin_change(current_case_num, approve_coins_base)
            # 计算实际人气增加 (正数)
            actual_popularity_gain = math.ceil(current_case_num * approve_popularity_base / 5)

            if actual_popularity_gain <= 0: # 避免除以零或负数
                print("Error: actual_popularity_gain 计算结果异常 (<= 0)。", flush=True)
                return None

            if theoretical_coin_change <= 0: # 金币是减少的
                theoretical_coin_cost = abs(theoretical_coin_change) # 理论金币成本取绝对值 (正数)

                if zero_coins_approve: # 允许 0 金币批准
                    if current_coins <= theoretical_coin_cost: # 金币不足以支付理论成本
                        actual_coin_cost = max(0, current_coins)  # 实际金币成本为花光所有金币 (至少为 0)
                    else:
                        actual_coin_cost = theoretical_coin_cost # 金币充足，实际成本为理论成本
                else: # 不允许 0 金币批准 (但实际上 little_timmy_2 case 是允许的，这里为了完整性保留)
                    actual_coin_cost = theoretical_coin_cost # 实际金币成本等于理论金币成本
            else: # theoretical_coin_change > 0, 金币是增加的 (不太可能在 approve_court_coins_base 为负数的情况下出现，但为了完整性保留)
                actual_coin_cost = 0 # 金币增加，成本为 0

            if actual_popularity_gain > 0:
                coin_pop_ratio = actual_coin_cost / actual_popularity_gain # 计算性价比
            else:
                coin_pop_ratio = float('inf') # 如果人气没有增加，性价比为无穷大 (表示极差)


            print(f"  little_timmy_2 性价比计算:", flush=True)
            print(f"    理论金币变化: {theoretical_coin_change}", flush=True)
            print(f"    实际人气增加: {actual_popularity_gain}", flush=True)
            print(f"    实际金币成本: {actual_coin_cost}", flush=True)
            print(f"    金币/人气 性价比 (coin_pop_ratio): {coin_pop_ratio:.2f}", flush=True) # 保留两位小数

            return coin_pop_ratio
        
        def _attempt_action_with_condition_and_global_check(action_priority, action_config, condition_config, case_config): 
            """
            尝试执行指定顺位的 Action，包括条件检查和全局检查。

            Args:
                action_priority (str): Action 的顺位，例如 "Primary", "Secondary", "Tertiary".
                action_config (dict):  Action 的配置字典 (从 case_config["actions"] 中获取).
                condition_config (dict): 条件配置字典 (例如 case_config["primary_condition"]).
                case_config (dict): 当前 case 的完整配置信息 (需要传递 case_config 才能获取 zero_coins_approve 和 approve_court_coins_base). 

            Returns:
                tuple: (action_to_execute, action_type) - 如果成功执行 action 或被全局条件阻止，则返回 action 信息，否则返回 (None, None).
            """
            action_to_execute = action_config # action_config 本身就是 action 值 (例如 "approve", "reject", "dismiss")
            action_type = action_config

            condition_met = True # 假设 condition 默认满足，如果没有定义 condition
            if condition_config: # 如果定义了 condition，则进行条件检查
                condition_type = condition_config.get("type")
                operator = condition_config.get("operator")
                condition_value = condition_config.get("value")

                if condition_type == "mental_health": 
                    current_mental_health = get_current_mental_health()
                    if operator == "greater_than": 
                        condition_met = current_mental_health > condition_value
                    elif operator == "less_than":
                        condition_met = current_mental_health < condition_value
                    elif operator == "equal_to":
                        condition_met = current_mental_health == condition_value
                        
                elif condition_type == "none": # 表示无条件
                    condition_met = True # 条件总是满足
                    
                elif condition_type == "Cost_Fizarre_Drink": 
                    fizarre_drink_config = self.bot.config.get("justice", {}).get("cases", {}).get("fizarre_drink_1") # 获取 fizarre_drink_1 配置
                    if fizarre_drink_config:
                        fizarre_drink_approve_coins_base = fizarre_drink_config.get("approve_court_coins_base", -1) 
                        current_case_num = get_current_case_number() # 获取当前案子编号
                        fizarre_drink_cost = math.ceil(current_case_num * abs(fizarre_drink_approve_coins_base) / 5) # 计算 Fizarre Drink 成本

                        approve_court_coins_base_for_current_case = case_config.get("approve_court_coins_base", 0) # 获取当前 case 的 base coins 成本
                        approve_cost_current_case = calculate_approve_coin_change(current_case_num, approve_court_coins_base_for_current_case) # 使用公式计算当前 case 的 coins 成本
                        current_coins = get_current_coins() # 获取当前 coins 数量
                        coins_after_approval = current_coins + approve_cost_current_case # 计算批准当前 case 后的剩余 coins 数量

                        if coins_after_approval >= fizarre_drink_cost: # 检查 批准后 coins 是否足够支付 Fizarre Drink 成本 
                            condition_met = True # coins 足够，条件满足
                            print(f"  Cost_Fizarre_Drink Condition 满足: Coins 批准后充足 (需要 {fizarre_drink_cost}, 批准后剩余 {coins_after_approval})。", flush=True) 
                        else:
                            condition_met = False # coins 不足，条件不满足
                            print(f"  Cost_Fizarre_Drink Condition 不满足: Coins 批准后不足 (需要 {fizarre_drink_cost}, 批准后剩余 {coins_after_approval})。", flush=True) 
                    else:
                        condition_met = False # 无法获取 fizarre_drink_1 配置，条件不满足
                        print("  Cost_Fizarre_Drink Condition 不满足: 无法获取 fizarre_drink_1 配置。", flush=True)
                
                elif condition_type == "coin_pop_ratio": 
                    current_case_num = get_current_case_number()
                    current_coins = get_current_coins()
                    ratio = calculate_coin_pop_ratio(case_config, current_case_num, current_coins) # 调用性价比计算函数

                    if ratio is not None: # 确保 ratio 计算成功
                        if operator == "less_than":
                            condition_met = ratio <= condition_value 
                        elif operator == "greater_than": 
                            condition_met = ratio > condition_value
                        else:
                            condition_met = False # 未知的操作符
                            print(f"  coin_pop_ratio Condition: 未知的操作符: {operator}", flush=True)
                        if condition_met:
                            print(f"  coin_pop_ratio Condition 满足: ratio {ratio:.2f} {operator} {condition_value}", flush=True)
                        else:
                            print(f"  coin_pop_ratio Condition 不满足: ratio {ratio:.2f} is not {operator} {condition_value}", flush=True)
                    else:
                        condition_met = False # ratio 计算失败
                        print("  coin_pop_ratio Condition 不满足: ratio 计算失败。", flush=True)
                
                elif condition_type == "case_number": 
                    current_case_number = get_current_case_number()
                    if operator == "greater_than": 
                        condition_met = current_case_number > condition_value
                    elif operator == "less_than":
                        condition_met = current_case_number < condition_value
                    elif operator == "equal_to":
                        condition_met = current_case_number == condition_value
                        
                else:
                    condition_met = False # 未知的操作符或条件类型，条件不满足

            if condition_met: # 如果 condition 满足 (或没有 condition)
                print(f"  {action_priority} Condition 满足， 尝试执行 {action_priority} Action: {action_to_execute}", flush=True)

                # --- Global Check for Action (Approve or Dismiss) ---
                if action_type == "approve":
                    mister_bribe_status = get_mister_bribe_quest_status()
                    if mister_bribe_status != 0:
                        print(f"  Mister Bribe Quest 激活， 阻止 {action_priority} 'approve' Action。", flush=True)
                        action_to_execute = None # 阻止 action
                        action_type = None # Reset action_type
                    else:
                        print(f"  Mister Bribe Quest 未激活， {action_priority} 'approve' Action Global Check (Quest) 通过。", flush=True)

                        # --- Coins Check for Approve Action ---
                        zero_coins_approve = case_config.get("zero_coins_approve", False) # 默认 False
                        approve_court_coins_base = case_config.get("approve_court_coins_base", 0) # 默认 0
                        approve_cost_current_case = calculate_approve_coin_change(get_current_case_number(), approve_court_coins_base)
                        
                        if not zero_coins_approve and approve_cost_current_case < 0: # 如果 zero_coins_approve 为 false 且 approve_cost_current_case 为负数时才检查金币(因为如果是正数或零，金币不会减少)
                            current_coins = get_current_coins()
                            if current_coins < abs(approve_cost_current_case):
                                print(f"  Coins 不足 (批准后减少 {-approve_cost_current_case}, 当前 {current_coins})， 阻止 {action_priority} 'approve' Action。", flush=True) 
                                action_to_execute = None # 阻止 action
                                action_type = None # Reset action_type
                            else:
                                print(f"  Coins 充足 (批准后减少 {-approve_cost_current_case}, 当前 {current_coins})， {action_priority} 'approve' Action Global Check (Coins) 通过。", flush=True) 
                        else:
                            print(f"  Zero Coins Approve 为 True 或 approve_court_coins_base >= 0， 跳过 {action_priority} 'approve' Action Global Check (Coins)。", flush=True)

                elif action_type == "dismiss":
                    dismiss_item_available = get_dismiss_item_available_status()
                    if not dismiss_item_available:
                        print(f"  Dismiss Items 不可用， 阻止 {action_priority} 'dismiss' Action。", flush=True)
                        action_to_execute = None # 阻止 action
                        action_type = None # Reset action_type
                    else:
                        print(f"  Dismiss Items 可用， {action_priority} 'dismiss' Action Global Check 通过。", flush=True)

                if action_to_execute: # 再次检查 action_to_execute 是否仍然有效 (未被全局条件阻止)
                    print(f"  最终确定执行 {action_priority} 顺位 Action: {action_to_execute}", flush=True)
                    # ... (执行 action 的代码应该在 process_justice_case_by_image 中处理) ...
                    return action_to_execute, action_type # 返回 action 信息

            return None, None # 条件不满足或全局检查失败，返回 None
        
        def process_justice_case_by_image(recognized_image_path):
            """
            根据识别到的图片路径，处理 Justice 类别下的 case。
            """
            justice_cases = self.bot.config.get("justice", {}).get("cases", {}) # 获取 justice.cases 配置

            for case_name, case_config in justice_cases.items():
                if case_config.get("image_path") == recognized_image_path:
                    print(f"识别到 Justice Case: {case_name} - {case_config.get('text')}", flush=True)

                    action_to_execute = None # 初始化 action_to_execute 为 None
                    action_type = None #  Initialize action_type

                    # --- 尝试执行 Primary Action ---
                    primary_action_config = case_config["actions"].get("primary") if case_config.get("actions") else None
                    if primary_action_config:
                        primary_condition_config = case_config.get("primary_condition")
                        action_to_execute, action_type = _attempt_action_with_condition_and_global_check("Primary", primary_action_config, primary_condition_config, case_config) 
                        if action_to_execute: # 如果 Primary Action 执行成功，则直接返回
                            if action_to_execute == "approve":
                                print("执行 Approve 操作", flush=True)
                            elif action_to_execute == "reject":
                                print("执行 Reject 操作", flush=True)
                            elif action_to_execute == "dismiss":
                                print("执行 Dismiss 操作", flush=True)
                            return

                    # --- 尝试执行 Secondary Action (只有当 Primary Action 没有成功执行时) ---
                    secondary_action_config = case_config["actions"].get("secondary") if case_config.get("actions") else None
                    if secondary_action_config:
                        secondary_condition_config = case_config.get("secondary_condition")
                        action_to_execute, action_type = _attempt_action_with_condition_and_global_check("Secondary", secondary_action_config, secondary_condition_config, case_config) 
                        if action_to_execute: # 如果 Secondary Action 执行成功，则直接返回
                            if action_to_execute == "approve":
                                print("执行 Approve 操作", flush=True)
                            elif action_to_execute == "reject":
                                print("执行 Reject 操作", flush=True)
                            elif action_to_execute == "dismiss":
                                print("执行 Dismiss 操作", flush=True)
                            return

                    # --- 执行 Tertiary Action (只有当 Primary 和 Secondary Action 都失败时) ---
                    tertiary_action = case_config["actions"].get("tertiary") if case_config.get("actions") else None
                    if tertiary_action:
                        action_to_execute = tertiary_action
                        action_type = tertiary_action
                        print(f"  执行 3号顺位 Action (Tertiary - 无条件，无全局检查): {action_to_execute}", flush=True)

                        if action_to_execute: # 再次检查 action_to_execute 是否仍然有效
                            print(f"  最终确定执行 3号顺位 Action: {action_to_execute}", flush=True)
                            # ... (执行 tertiary action 的代码) ...
                            if action_to_execute == "approve":
                                print("执行 Approve 操作", flush=True)
                            elif action_to_execute == "reject":
                                print("执行 Reject 操作", flush=True)
                            elif action_to_execute == "dismiss":
                                print("执行 Dismiss 操作", flush=True)
                            return # 执行 tertiary action 后直接返回

                    # --- 如果所有顺位 action 都没执行，则输出提示 ---
                    if action_to_execute is None:
                        print("  所有顺位 Action 的条件或全局检查均不满足， 没有 Action 执行。", flush=True)
                    return #  程序结束，如果没有找到匹配的 case 或任何 action 被执行

            print(f"未找到与图片路径 {recognized_image_path} 匹配的 Justice Case 配置。", flush=True)
        
        def process_justice_case_by_image_1(recognized_image_path):
            """
            根据识别到的图片路径，处理 Justice 类别下的 case。
            """
            justice_cases = self.bot.config.get("justice", {}).get("characters", {}) # 获取 justice.characters 配置
            
            # 遍历人物而不是直接遍历案件
            for character_name, character_data in justice_cases.items():
                for case_name, case_config in character_data.get("cases", {}).items():
                    if case_config.get("image_path") == recognized_image_path:
                        print(f"识别到 Justice Case: {case_name} - {case_config.get('text')}", flush=True)

                        action_to_execute = None
                        action_type = None

                        # --- 尝试执行 Primary Action ---
                        primary_action_config = case_config["actions"].get("primary") if case_config.get("actions") else None
                        if primary_action_config:
                            primary_condition_config = case_config.get("primary_condition")
                            action_to_execute, action_type = _attempt_action_with_condition_and_global_check("Primary", primary_action_config, primary_condition_config, case_config)
                            if action_to_execute:
                                if action_to_execute == "approve":
                                    print("执行 Approve 操作", flush=True)
                                elif action_to_execute == "reject":
                                    print("执行 Reject 操作", flush=True)
                                elif action_to_execute == "dismiss":
                                    print("执行 Dismiss 操作", flush=True)
                                return

                        # --- 尝试执行 Secondary Action ---
                        secondary_action_config = case_config["actions"].get("secondary") if case_config.get("actions") else None
                        if secondary_action_config:
                            secondary_condition_config = case_config.get("secondary_condition")
                            action_to_execute, action_type = _attempt_action_with_condition_and_global_check("Secondary", secondary_action_config, secondary_condition_config, case_config)
                            if action_to_execute:
                                if action_to_execute == "approve":
                                    print("执行 Approve 操作", flush=True)
                                elif action_to_execute == "reject":
                                    print("执行 Reject 操作", flush=True)
                                elif action_to_execute == "dismiss":
                                    print("执行 Dismiss 操作", flush=True)
                                return
                        # --- 执行 Tertiary Action ---
                        tertiary_action = case_config["actions"].get("tertiary") if case_config.get("actions") else None
                        if tertiary_action:
                            action_to_execute = tertiary_action
                            action_type = tertiary_action
                            print(f"  执行 3号顺位 Action (Tertiary - 无条件，无全局检查): {action_to_execute}", flush=True)

                            if action_to_execute:
                                print(f"  最终确定执行 3号顺位 Action: {action_to_execute}", flush=True)
                                if action_to_execute == "approve":
                                    print("执行 Approve 操作", flush=True)
                                elif action_to_execute == "reject":
                                    print("执行 Reject 操作", flush=True)
                                elif action_to_execute == "dismiss":
                                    print("执行 Dismiss 操作", flush=True)
                                return

                        if action_to_execute is None:
                            print("  所有顺位 Action 的条件或全局检查均不满足， 没有 Action 执行。", flush=True)
                        return

            print(f"未找到与图片路径 {recognized_image_path} 匹配的 Justice Case 配置。", flush=True)
        
        def detect_and_process_justice_case(): 
            """
            自动检测当前屏幕上的 Justice Case，并进行处理。
            """
            screenshot = self.bot.get_screenshot() # 获取游戏画面截图
            if screenshot is None:
                print("截图失败，无法检测 Justice Case。", flush=True)
                return

            justice_cases = self.bot.config.get("justice", {}).get("cases", {}) # 获取 justice.cases 配置

            detected_case_name = None # 初始化检测到的 case 名称
            detected_image_path = None # 初始化检测到的 image_path

            for case_name, case_config in justice_cases.items(): # 遍历所有 Justice Cases
                image_path_to_find = case_config["image_path"]
                image_location = self.bot.find_image_location(image_path_to_find) # 查找图片

                if image_location: # 如果找到图片
                    print(f"  在屏幕上找到图片: {image_path_to_find}，疑似 Case: {case_name}", flush=True)
                    detected_case_name = case_name # 记录检测到的 case 名称
                    detected_image_path = image_path_to_find # 记录检测到的 image_path
                    break # 找到一个匹配的 case 就可以停止遍历了，假设屏幕上一次只出现一个 Justice Case

            if detected_case_name: # 如果检测到 case
                print(f"检测到 Justice Case: {detected_case_name}", flush=True)
                process_justice_case_by_image(detected_image_path)
            else:
                print("  未在屏幕上检测到任何 Justice Case。", flush=True)
        
        def detect_and_process_justice_case_1():
            """
            自动检测当前屏幕上的 Justice Case，并进行处理。
            """
            screenshot = self.bot.get_screenshot()
            if screenshot is None:
                print("截图失败，无法检测 Justice Case。", flush=True)
                return

            justice_characters = self.bot.config.get("justice", {}).get("characters", {})

            detected_case_name = None
            detected_image_path = None
            
            # 先查找人物
            for character_name, character_data in justice_characters.items():
                character_image_path = character_data.get("image_path")
                if character_image_path:
                    character_location = self.bot.find_image_location(character_image_path)
                    if character_location:
                        print(f"  在屏幕上找到人物图片: {character_image_path}，疑似人物: {character_name}", flush=True)
                        # 找到人物后，只遍历该人物的 cases
                        for case_name, case_config in character_data.get("cases", {}).items():
                            image_path_to_find = case_config["image_path"]
                            image_location = self.bot.find_image_location(image_path_to_find)

                            if image_location:
                                print(f"    在屏幕上找到 Case 图片: {image_path_to_find}，Case: {case_name}", flush=True)
                                detected_case_name = case_name
                                detected_image_path = image_path_to_find
                                break  # 找到一个匹配的 case 就可以停止遍历了
                        if detected_case_name: #如果找到人物跳出人物循环
                            break
            if detected_case_name:
                print(f"检测到 Justice Case: {detected_case_name}", flush=True)
                process_justice_case_by_image_1(detected_image_path)
            else:
                print("  未在屏幕上检测到任何 Justice Case 人物。", flush=True)
        
        detect_and_process_justice_case_1()

    # --- 钓鱼 ---
    def _on_fishing_trigger(self, x_screen, y_screen, button, pressed): 
        """钓鱼模式下，右键点击触发钓鱼动作的回调。"""
        if not self.bot.running or not self.bot.fishing_initialized:
            return # 如果脚本停止或钓鱼未初始化，则不执行

        if button == mouse.Button.right and pressed:
            # 检查点击是否在游戏窗口内 (粗略检查)
            active_window_hwnd = win32gui.GetForegroundWindow()
            if active_window_hwnd != self.bot.hwnd:
                # print("右键点击不在游戏窗口内，忽略。", flush=True)
                return

            fish_position_x, _ = self.bot.get_client_mouse_position()
            if fish_position_x is None:
                print("钓鱼错误：无法获取鱼的位置。", flush=True)
                return

            # AHK 脚本中的边界检查，稍微放宽一点
            if not (self.bot.fishing_minimum_x - 50 < fish_position_x < self.bot.fishing_maximum_x + 50):
                print(f"钓鱼警告：点击位置 {fish_position_x:.2f} 超出合理范围 "
                      f"({self.bot.fishing_minimum_x - 50:.2f} - {self.bot.fishing_maximum_x + 50:.2f}). "
                      "可能需要重新设置。", flush=True)
                # 可以选择在这里重置 self.bot.fishing_initialized = False
                # self.bot.fishing_initialized = False
                # print("钓鱼设置已重置，请重新运行钓鱼功能进行设置。", flush=True)
                return # 或者允许尝试，但给出警告

            if self.bot.fishing_possible_distance <= 0:
                print("钓鱼错误：钓鱼区域宽度无效，请重新设置。", flush=True)
                self.bot.fishing_initialized = False
                return

            fish_distance_percent = (fish_position_x - self.bot.fishing_minimum_x) / self.bot.fishing_possible_distance
            # 限制百分比在0到1之间
            fish_distance_percent = max(0.0, min(1.0, fish_distance_percent))

            powerbar_percent_val = self.bot.calculate_fishing_power_bar_percent(fish_distance_percent)
            powerbar_bars_val = powerbar_percent_val * 7 # AHK脚本中的魔法数字
            hold_duration_ms = self.bot.calculate_fishing_power_bar_click_duration(powerbar_bars_val)

            print(f"鱼位置: {fish_position_x:.0f} -> 距离百分比: {fish_distance_percent*100:.1f}% -> "
                  f"力量条: {powerbar_bars_val:.2f}格 -> 按住: {hold_duration_ms:.0f}ms", flush=True)

            self.bot.hold_left_mouse_ms_at_current_pos(hold_duration_ms)

    def _setup_fishing_auto(self):
        """通过截屏和模板匹配自动设置钓鱼区域。"""
        print("\n--- 开始自动钓鱼设置 ---", flush=True)
        self.bot.fishing_initialized = False

        try:
            if self.bot.hwnd:
                win32gui.SetForegroundWindow(self.bot.hwnd)
                time.sleep(0.5) # 给窗口响应和画面刷新
        except Exception as e:
            print(f"激活游戏窗口失败: {e}", flush=True)
            return False

        start_template_path = self.fishing_config.get("start_template")
        fishing_area_width = self.fishing_config.get("width")
        # threshold = self.fishing_config.get("template_threshold", THRESHOLD) # 全局或特定阈值

        if not start_template_path:
            print("错误：未在配置文件中找到钓鱼起始模板路径。", flush=True)
            return False
        if fishing_area_width is None:
            print("错误：未在配置文件中找到钓鱼区域宽度 (width)。", flush=True)
            return False

        print("正在尝试自动识别钓鱼区域起始位置...", flush=True)
        self.bot.get_screenshot(refresh=True) # 确保使用最新截图

        print(f"查找钓鱼起始模板: {start_template_path}", flush=True)
        # 假设 find_image_location 使用全局 THRESHOLD 或 fishing_config 中的 template_threshold
        start_loc = self.bot.find_image_location(start_template_path)

        if not start_loc:
            print(f"自动设置失败：未能找到钓鱼起始模板图像 ({start_template_path})。", flush=True)
            self.bot._save_debug_screenshot("fishing_setup_fail_start_template")
            return False

        # start_loc 是模板的左上角 (x, y)
        # 加载模板图像以获取其宽度
        start_template_img = self.bot._load_image(start_template_path)
        if start_template_img is None:
            print(f"错误：无法加载模板图像 {start_template_path} 以获取尺寸。", flush=True)
            return False
        
        # 确定实际的 x_left_lake
        # 选项1: 如果 FISH_START.bmp 的左边缘就是钓鱼区域的左边缘
        # x_left_lake = start_loc[0]
        # 选项2: 如果 FISH_START.bmp 的右边缘是钓鱼区域的左边缘
        # x_left_lake = start_loc[0] + start_template_img.shape[1]
        # 选项3: 如果 FISH_START.bmp 的中心是钓鱼区域的左边缘 (不常见，但可能)
        # x_left_lake = start_loc[0] + start_template_img.shape[1] // 2

        # 最常见的情况是，模板图片本身就是那个左边界标记物，
        # 所以它的左上角x坐标 (start_loc[0]) 就是我们要找的湖的左边界。
        # 如果 FISH_START.bmp 覆盖了部分非钓鱼区，你可能需要做调整。
        # 假设 FISH_START.bmp 的左边缘就是实际的钓鱼区域左边缘
        self.bot.fishing_x_left_lake = start_loc[0]
        print(f"钓鱼起始模板找到于 X:{start_loc[0]}, Y:{start_loc[1]}. 钓鱼区域左边界设为: {self.bot.fishing_x_left_lake:.0f}", flush=True)

        # 最右侧位置是固定的偏移量
        self.bot.fishing_x_right_lake = self.bot.fishing_x_left_lake + fishing_area_width
        print(f"钓鱼区域右边界设为 (左边界 + {fishing_area_width}): {self.bot.fishing_x_right_lake:.0f}", flush=True)


        # --- 后续计算与之前相同 ---
        distance_pond = self.bot.fishing_x_right_lake - self.bot.fishing_x_left_lake
        # 使用和之前一样的偏移比例
        self.bot.fishing_minimum_x = self.bot.fishing_x_left_lake + distance_pond * 0.074
        self.bot.fishing_maximum_x = self.bot.fishing_x_right_lake - distance_pond * 0.062
        # 注意：这里的 0.074 和 0.062 是基于你之前手动设置时用的比例。
        # 如果钓鱼区域的宽度是完全固定的 (607px)，并且这个宽度本身就是 `possible_distance`
        # 那么 minimum_x 和 maximum_x 的计算可能需要调整。
        # 例如，如果 FISH_START.bmp 的左边就是 minimum_x，FISH_START.bmp左边+607 就是 maximum_x
        # 这种情况下：
        # self.bot.fishing_minimum_x = self.bot.fishing_x_left_lake
        # self.bot.fishing_maximum_x = self.bot.fishing_x_right_lake
        # self.bot.fishing_possible_distance = fishing_area_width
        # *你需要根据游戏内实际情况决定用哪种计算方式*
        # 我暂时保留之前的计算方式，因为它考虑了从湖边内缩一点作为有效区域：
        self.bot.fishing_possible_distance = self.bot.fishing_maximum_x - self.bot.fishing_minimum_x


        if self.bot.fishing_possible_distance <= 0:
            print(f"自动设置错误：计算出的有效钓鱼区域宽度无效 ({self.bot.fishing_possible_distance:.2f})。", flush=True)
            print(f"  x_left_lake: {self.bot.fishing_x_left_lake}, x_right_lake: {self.bot.fishing_x_right_lake}", flush=True)
            print(f"  minimum_x: {self.bot.fishing_minimum_x}, maximum_x: {self.bot.fishing_maximum_x}", flush=True)
            self.bot._save_debug_screenshot("fishing_setup_fail_width")
            return False

        print(f"钓鱼区域自动初始化成功！总宽度: {distance_pond:.0f}px, 有效宽度: {self.bot.fishing_possible_distance:.2f}px。", flush=True)
        print(f"  湖泊边界 X: {self.bot.fishing_x_left_lake:.2f} - {self.bot.fishing_x_right_lake:.2f}", flush=True)
        print(f"  有效抛竿X范围: {self.bot.fishing_minimum_x:.2f} - {self.bot.fishing_maximum_x:.2f}", flush=True)
        self.bot.fishing_initialized = True
        return True

    # fishing_loop 和 _on_fishing_trigger 保持不变
    def fishing_loop(self):
        """钓鱼主循环。"""
        if not self._setup_fishing_auto():
            print("自动钓鱼设置未完成，功能退出。", flush=True)
            return

        print("\n--- 钓鱼模式已激活 ---", flush=True)
        print("现在，在游戏内钓鱼区域【右键点击】鱼的位置即可自动抛竿。", flush=True)
        print("按 'delete' 键退出整个脚本。", flush=True)

        self._fishing_listener = mouse.Listener(on_click=self._on_fishing_trigger)
        self._fishing_listener.start()

        try:
            while self.bot.running:
                time.sleep(0.1)
        finally:
            if self._fishing_listener:
                print("停止钓鱼监听器...", flush=True)
                self._fishing_listener.stop()
                self._fishing_listener.join()
                self._fishing_listener = None
            print("钓鱼模式结束。", flush=True)
            self.bot.fishing_initialized = False
        
    # --- 测试功能 ---
    def test1_loop(self):
        """
        主程序入口 (测试功能)
        """
        while self.bot.running:
            self.bot._click_and_hold(*self.bot.config["summoning"]["spawn"])
            time.sleep(0.05) 
            
    # --- 测试功能 ---
    def test_loop(self):
        """
        主程序入口 (测试功能)
        """
        while self.bot.running:
            self.bot.interact_with_image(self.bot.config["summoning"]["upgrade_6"])
            time.sleep(0.05) 

# --- 主程序入口 ---
def main():
    """
    主程序入口
    """
    try:
        window_setup = {
            "idleon": "Legends Of Idleon",
            "edge": "Intermediate D3D Window" 
        }
        bot = GameBot()  # 初始化 GameBot 核心类
        functions = GameFunctions(bot)  # 初始化功能函数类
    except Exception as e:
        print(f"初始化失败: {e}", flush=True)
        return

    bot.start_exit_listener()  # 启动退出监听线程

    # 功能选项映射
    function_map = {
        "0": ("测试", functions.test_loop),
        "1": ("AFK 快进", functions.afk_fast_forward_loop),
        "2": ("AFK 基因萃取", functions.afk_gene_extract_loop),
        "3": ("航海", functions.sailing_loop),
        "4": ("邮差", functions.postman_loop),
        "5": ("地牢骰子", functions.dungeon_lootroll_loop),
        "6": ("养殖", functions.breeding_loop),
        "7": ("游戏", functions.gaming_loop),
        "8": ("Boss", functions.boss_loop),
        "9": ("开箱", functions.open_loop),
        "10": ("猫头鹰（未更新）", functions.owl_loop),
        "11": ("农场", functions.farming_loop),
        "12": ("勇气", functions.paying_loop),
        "13": ("竞技场", functions.colo_loop),
        "14": ("正义（未更新）", functions.justice_loop),
        "15": ("钓鱼", functions.fishing_loop),
    }

    # 动态生成菜单
    print("请选择功能：")
    for key, (description, _) in function_map.items():
        print(f"{key}. {description}")
    choice = input(f"请输入选项 (0-{len(function_map) - 1}): ")

    print("脚本运行中，按下 'delete' 键退出...")

    # 根据用户选择执行对应功能
    selected_function = function_map.get(choice)
    if selected_function:
        selected_function[1]()  # 执行对应的函数
    else:
        print("无效选项！", flush=True)

if __name__ == "__main__":
    main()