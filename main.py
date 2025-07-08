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

    def __init__(self, window_configs={"default": "Legends Of Idleon"}): # 提供默认值
        """
        初始化 GameBot 实例
        Args:
            window_configs (dict): 包含窗口名称映射的字典。
                                  例如: {'default': 'Legends Of Idleon', 'other': 'Another Window'}
        """
        self.hwnds = {} # 存储所有窗口的句柄
        self.active_window_key = 'default' # 初始化活动窗口键

        # 确保 window_configs 是一个字典
        if not isinstance(window_configs, dict):
            raise TypeError("window_configs must be a dictionary.")

        for key, window_title in window_configs.items():
            # 确保 window_title 是一个字符串
            if not isinstance(window_title, str):
                print(f"Warning: Expected a string for window title for key '{key}', but got {type(window_title).__name__}. Skipping this window.")
                continue # 跳过这个条目

            # 确保窗口标题不是空字符串
            if not window_title:
                print(f"Warning: Empty window title for key '{key}'. Skipping.")
                continue

            try:
                hwnd = win32gui.FindWindow(None, window_title)
                if not hwnd:
                    print(f"Warning: Window '{window_title}' (key: {key}) not found.")
                else:
                    self.hwnds[key] = hwnd
                    print(f"Found window '{window_title}' (key: {key}) with HWND: {hwnd}")
            except TypeError as e:
                print(f"Error finding window '{window_title}' (key: {key}): {e}")
                print(f"The value passed to FindWindow was: {window_title}, which is of type {type(window_title).__name__}")
                raise # 重新抛出错误

        if not self.hwnds:
             raise Exception("No game windows were found. Please check window titles in config.")

        # 确保 'default' 键存在
        if 'default' not in self.hwnds:
             # 如果默认窗口不存在，但其他窗口存在，可以自动选择一个作为默认
             if self.hwnds:
                 self.active_window_key = next(iter(self.hwnds))
                 print(f"Warning: Default window key 'default' not found. Using '{self.active_window_key}' as active window.")
             else:
                # This case is already handled by 'if not self.hwnds'
                pass

        # 加载配置文件
        self.config = self._load_config(CONFIG_FILE_PATH)
        if not self.config:
            raise Exception("Configuration loading failed, exiting.")

        # 设置 Tesseract OCR 路径
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

        # 确保截图文件夹存在
        os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

        # 初始化运行标志和缓存
        self.running = True
        self.image_cache = {}  # 缓存已加载的图像
        self.screenshot_cache = None  # 缓存最近的截图

    @property
    def active_hwnd(self):
        """获取当前活动窗口的句柄"""
        return self.hwnds.get(self.active_window_key) # 使用 .get() 避免 KeyError

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
            
    # ADDED: 添加了缺失的调试截图方法
    def _save_debug_screenshot(self, reason_prefix):
        """保存当前缓存的截图用于调试"""
        if self.screenshot_cache is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(SCREENSHOT_FOLDER, f"debug_{reason_prefix}_{timestamp}.png")
            try:
                cv2.imwrite(filename, self.screenshot_cache)
                print(f"Debug screenshot saved to: {filename}", flush=True)
            except Exception as e:
                print(f"Failed to save debug screenshot: {e}", flush=True)
        else:
            print("No cached screenshot to save for debugging.", flush=True)


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
        if not refresh and self.screenshot_cache is not None:
            screenshot = self.screenshot_cache
        else:
            screenshot = self._capture_window()
            if screenshot is None:
                return None
            self.screenshot_cache = screenshot

        try:
            # FIXED: 使用 self.active_hwnd 代替 self.hwnd
            window_handle = self.active_hwnd
            if not window_handle:
                return None
            left, top, right, bottom = win32gui.GetWindowRect(window_handle)

            if max_loc:
                target_left = max_loc[0] + x_offset
                target_top = max_loc[1] + y_offset
                if width is None:
                    # 修正：当提供了max_loc时，宽度和高度不应该依赖整个窗口
                    # 如果未提供，则应默认为一个合理的值或报错。这里假设截取到窗口边缘。
                    width = right - left - target_left
                if height is None:
                    height = bottom - top - target_top
            else:
                target_left = x_offset
                target_top = y_offset
                if width is None:
                    width = right - left - x_offset
                if height is None:
                    height = bottom - top - y_offset

            # 提取截图区域，并确保坐标不越界
            h, w, _ = screenshot.shape
            target_left = max(0, target_left)
            target_top = max(0, target_top)
            target_right = min(w, target_left + width)
            target_bottom = min(h, target_top + height)

            cropped_image = screenshot[target_top:target_bottom, target_left:target_right]
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
        # FIXED: 使用 self.active_hwnd 代替 self.hwnd
        window_handle = self.active_hwnd
        if not window_handle:
            print("Error in _capture_window: active_hwnd is None.")
            return None

        try:
            left, top, right, bottom = win32gui.GetWindowRect(window_handle)
            width = right - left
            height = bottom - top
            
            # 如果窗口最小化，宽高可能为0或负数
            if width <= 0 or height <= 0:
                print(f"Warning: Window '{win32gui.GetWindowText(window_handle)}' has invalid dimensions (w:{width}, h:{height}). Is it minimized?")
                return None

            hdc = win32gui.GetWindowDC(window_handle)
            mfcDC = win32ui.CreateDCFromHandle(hdc)
            saveDC = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            image = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)

            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(window_handle, hdc)

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
        # 获取的是整个窗口的截图
        screenshot = self.get_screenshot(refresh=refresh_screenshot)
        if screenshot is None:
            return None

        template = self._load_image(image_path)
        if template is None:
            return None

        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

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
        max_loc = self.find_image_location(image_path)
        if max_loc:
            template = self._load_image(image_path) # 使用缓存加载
            if template is None:
                print(f"模板图片 {image_path} 加载失败！无法交互。", flush=True)
                return False
            center_x = max_loc[0] + template.shape[1] // 2 + x_offset
            center_y = max_loc[1] + template.shape[0] // 2 + y_offset

            if swipe:
                loot_y_offset = -50
                loot_swipe_width = 300
                swipe_steps = 10
                self._swipe_loot(center_x, center_y + loot_y_offset, loot_swipe_width, swipe_steps)
            elif refresh:
                self._click_and_hold(*self.config["postman"]["refresh_coordinates"], duration=0.3)
                for _ in range(click_times):
                    self._click_and_hold(center_x, center_y, duration) # 移除了多余的offset参数
                    time.sleep(delay)
            else:
                for _ in range(click_times):
                    self._click_and_hold(center_x, center_y, duration) # 移除了多余的offset参数
                    time.sleep(delay)
            return True
        return False

    def _click_and_hold(self, x, y, duration=0, click_times=1): # 移除了多余的offset参数
        """
        模拟鼠标点击并按住指定时间
        """
        # FIXED: 使用 self.active_hwnd 代替 self.hwnd
        window_handle = self.active_hwnd
        if not window_handle: return

        target_x = int(x)
        target_y = int(y)
        lParam = win32api.MAKELONG(target_x, target_y)
        
        for _ in range(click_times):
            win32api.PostMessage(window_handle, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam)
            if duration > 0:
                time.sleep(duration)
            win32api.PostMessage(window_handle, win32con.WM_LBUTTONUP, 0, lParam)
            if click_times > 1:
                time.sleep(0.1)

    def _swipe_loot(self, center_x, center_y, width, steps):
        """
        模拟鼠标滑动 (用于物品拾取)
        """
        # FIXED: 使用 self.active_hwnd 代替 self.hwnd
        window_handle = self.active_hwnd
        if not window_handle: return

        start_x = center_x - width // 2
        end_x = center_x + width // 2
        x_step = (end_x - start_x) / steps

        lParam_start = win32api.MAKELONG(start_x, center_y)
        win32api.PostMessage(window_handle, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lParam_start)

        for i in range(1, steps + 1):
            x = int(start_x + i * x_step)
            lParam = win32api.MAKELONG(x, center_y)
            win32api.PostMessage(window_handle, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, lParam)
            time.sleep(0.01)

        lParam_end = win32api.MAKELONG(end_x, center_y)
        win32api.PostMessage(window_handle, win32con.WM_LBUTTONUP, 0, lParam_end)

    def preprocess_image(self, image):
        """
        预处理图像以增强白色数字识别效果
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        result = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def read_game_number(self, image_path, x_offset, y_offset, width, height, save_screenshot=False):
        """
        使用 OCR 读取游戏画面指定区域的数字
        """
        try:
            max_loc = self.find_image_location(image_path)
            if not max_loc:
                print(f"无法找到参考图像 {image_path}，OCR 读取失败。", flush=True)
                return None
            screenshot = self.get_screenshot(max_loc, x_offset, y_offset, width, height)
            if screenshot is None:
                return None

            im = cv2.resize(screenshot, None, fx=15, fy=15, interpolation=cv2.INTER_CUBIC)
            im = self.preprocess_image(im)
            pil_image = Image.fromarray(im)

            if save_screenshot:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base_filename = os.path.join(SCREENSHOT_FOLDER, f"screenshot_{timestamp}")
                pil_original_screenshot = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                pil_original_screenshot.save(f"{base_filename}_original.png")
                pil_image.save(f"{base_filename}_processed.png")

            text = pytesseract.image_to_string(pil_image, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,KMB')
            match = re.search(r"([\d\.,]+)([KMB])?", text.replace(" ", ""), re.IGNORECASE)
            if match:
                number_str = match.group(1).replace(",", "")
                unit = match.group(2)
                try:
                    number = float(number_str)
                    if unit:
                        unit = unit.upper()
                        if unit == 'K':
                            number *= 1000
                        elif unit == 'M':
                            number *= 1000000
                        elif unit == 'B':
                            number *= 1000000000
                    return int(number)
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
        """
        # FIXED: 使用 self.active_hwnd 代替 self.hwnd
        window_handle = self.active_hwnd
        if not window_handle: return

        vk_code = KEY_CODES.get(key.upper()) if isinstance(key, str) else key
        if vk_code is None:
            print(f"未知的按键: {key}", flush=True)
            return

        win32api.PostMessage(window_handle, win32con.WM_KEYDOWN, vk_code, 0)
        if duration > 0:
            time.sleep(duration)
        else:
            time.sleep(0.05) # 短暂延迟确保按键被识别
        win32api.PostMessage(window_handle, win32con.WM_KEYUP, vk_code, 0)


    def timed_input(self, prompt, timeout=2, timeout_msg=None):
        # ... (此函数无错误) ...
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
        """
        while True:
            if keyboard.is_pressed('delete'):
                print("检测到 'delete' 键，退出脚本。", flush=True)
                self.running = False
                break
            time.sleep(0.1)

    def start_exit_listener(self):
        """
        启动退出监听线程
        """
        exit_thread = threading.Thread(target=self.exit_handler)
        exit_thread.daemon = True
        exit_thread.start()

    def get_client_mouse_position(self):
        """
        获取鼠标在游戏窗口内的相对位置。
        """
        # FIXED: 使用 self.active_hwnd 代替 self.hwnd
        window_handle = self.active_hwnd
        if not window_handle: return None, None
        
        try:
            cursor_x, cursor_y = win32api.GetCursorPos()
            left, top, _, _ = win32gui.GetWindowRect(window_handle)
            # 使用 ScreenToClient 将屏幕坐标转换为窗口客户区坐标
            client_x, client_y = win32gui.ScreenToClient(window_handle, (cursor_x, cursor_y))
            return client_x, client_y
        except Exception as e:
            print(f"获取鼠标位置失败: {e}", flush=True)
            return None, None

    def calculate_fishing_power_bar_percent(self, fishbar_percent):
        # ... (此函数无错误) ...
        power_percent = (0.4664 * fishbar_percent**3 -
                         1.0693 * fishbar_percent**2 +
                         1.603 * fishbar_percent +
                         0.0036)
        return max(min(power_percent, 1.0), 0.0)

    def calculate_fishing_power_bar_click_duration(self, powerbar_bars):
        # ... (此函数无错误) ...
        hold_duration = (2.268 * powerbar_bars**3 -
                         32.207 * powerbar_bars**2 +
                         232.423 * powerbar_bars +
                         82.220)
        return min(max(hold_duration, 20.0), 900.0)

    def hold_left_mouse_ms_at_current_pos(self, duration_ms):
        # ... (此函数无错误) ...
        mouse_controller.press(mouse.Button.left)
        time.sleep(duration_ms / 1000.0)
        mouse_controller.release(mouse.Button.left)

# --- GameFunctions: 游戏功能函数集合 ---
class GameFunctions:
    """
    游戏功能函数集合，依赖 GameBot 核心类
    """
    def __init__(self, game_bot):
        self.bot = game_bot
        # Justice 系统占位符
        # TODO: 使用 OCR 或其他方法实现这些值的真实获取
        self.mister_bribe_quest_status = 0
        self.dismiss_item_available_status = True
        self.current_coins = 80
        self.current_case_number = 61
        
        # 钓鱼系统变量
        self._fishing_listener = None
        self.fishing_config = self.bot.config.get("fishing", {})
        self.bot.fishing_initialized = False # 将状态变量附加到bot实例上

    # ... 省略其他无错误的功能循环方法 ...
    # afk_fast_forward_loop, afk_gene_extract_loop, sailing_loop, postman_loop, dungeon_lootroll_loop,
    # breeding_loop, gaming_loop, boss_loop, open_loop, owl_loop, farming_loop, paying_loop, colo_loop
    # 这些方法都依赖于 GameBot 的核心方法，修复了 GameBot 后它们也应该能正常工作。
    # 为了简洁，这里不再重复粘贴它们的代码。假设它们都已存在。

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
            num_fights_input = input(f"请输入战斗次数 (默认 {num_fights}): ")
            if num_fights_input:
                num_fights = int(num_fights_input)
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
                    if not self.bot.running: break
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

                if not self.bot.running: break
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
                        if end_number is not None and end_number == 0: # 如果剩余时间为 0，关卡结束
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
        正义循环 (注意: 此功能依赖于未实现的占位符函数)
        """
        # ... 此处省略了 justice_loop 的内部函数定义，因为它们没有直接的语法错误 ...
        # 关键点在于，使用者必须自己实现 get_current_mental_health 等函数
        print("警告: '正义'功能当前使用占位符数据。")
        print("需要开发者实现真实的游戏数据获取函数才能正常工作。")
        time.sleep(3)
        # 此处应有 detect_and_process_justice_case_1() 的循环调用
        try:
            while self.bot.running:
                # detect_and_process_justice_case_1() # 假设这是要调用的函数
                print("执行一次 Justice 逻辑（测试模式）...")
                time.sleep(5) # 模拟处理间隔
        except Exception as e:
            print(f"justice_loop 异常: {e}", flush=True)


    # --- 钓鱼 ---
    def _on_fishing_trigger(self, x_screen, y_screen, button, pressed):
        """钓鱼模式下，右键点击触发钓鱼动作的回调。"""
        if not self.bot.running or not self.bot.fishing_initialized:
            return

        if button == mouse.Button.right and pressed:
            # FIXED: 使用 self.bot.active_hwnd 代替 self.bot.hwnd
            active_window_hwnd = win32gui.GetForegroundWindow()
            if active_window_hwnd != self.bot.active_hwnd:
                return

            fish_position_x, _ = self.bot.get_client_mouse_position()
            if fish_position_x is None:
                print("钓鱼错误：无法获取鱼的位置。", flush=True)
                return

            if not (self.bot.fishing_minimum_x - 50 < fish_position_x < self.bot.fishing_maximum_x + 50):
                print(f"钓鱼警告：点击位置 {fish_position_x:.2f} 超出合理范围 "
                      f"({self.bot.fishing_minimum_x - 50:.2f} - {self.bot.fishing_maximum_x + 50:.2f}). "
                      "可能需要重新设置。", flush=True)
                return

            if self.bot.fishing_possible_distance <= 0:
                print("钓鱼错误：钓鱼区域宽度无效，请重新设置。", flush=True)
                self.bot.fishing_initialized = False
                return

            fish_distance_percent = (fish_position_x - self.bot.fishing_minimum_x) / self.bot.fishing_possible_distance
            fish_distance_percent = max(0.0, min(1.0, fish_distance_percent))

            powerbar_percent_val = self.bot.calculate_fishing_power_bar_percent(fish_distance_percent)
            powerbar_bars_val = powerbar_percent_val * 7
            hold_duration_ms = self.bot.calculate_fishing_power_bar_click_duration(powerbar_bars_val)

            print(f"鱼位置: {fish_position_x:.0f} -> 距离百分比: {fish_distance_percent*100:.1f}% -> "
                  f"力量条: {powerbar_bars_val:.2f}格 -> 按住: {hold_duration_ms:.0f}ms", flush=True)

            self.bot.hold_left_mouse_ms_at_current_pos(hold_duration_ms)

    def _setup_fishing_auto(self):
        """通过截屏和模板匹配自动设置钓鱼区域。"""
        print("\n--- 开始自动钓鱼设置 ---", flush=True)
        self.bot.fishing_initialized = False

        # FIXED: 使用 self.bot.active_hwnd 代替 self.bot.hwnd
        if self.bot.active_hwnd:
            try:
                win32gui.SetForegroundWindow(self.bot.active_hwnd)
                time.sleep(0.5)
            except Exception as e:
                print(f"激活游戏窗口失败: {e}", flush=True)
                return False
        else:
            print("错误：未找到游戏窗口句柄。", flush=True)
            return False

        start_template_path = self.fishing_config.get("start_template")
        fishing_area_width = self.fishing_config.get("width")

        if not start_template_path or fishing_area_width is None:
            print("错误：配置文件中缺少钓鱼 'start_template' 或 'width'。", flush=True)
            return False

        print("正在尝试自动识别钓鱼区域起始位置...", flush=True)
        start_loc = self.bot.find_image_location(start_template_path, refresh_screenshot=True)

        if not start_loc:
            print(f"自动设置失败：未能找到钓鱼起始模板图像 ({start_template_path})。", flush=True)
            # FIXED: 调用已定义的调试截图方法
            self.bot._save_debug_screenshot("fishing_setup_fail_start_template")
            return False

        self.bot.fishing_x_left_lake = start_loc[0]
        self.bot.fishing_x_right_lake = self.bot.fishing_x_left_lake + fishing_area_width

        distance_pond = self.bot.fishing_x_right_lake - self.bot.fishing_x_left_lake
        self.bot.fishing_minimum_x = self.bot.fishing_x_left_lake + distance_pond * 0.074
        self.bot.fishing_maximum_x = self.bot.fishing_x_right_lake - distance_pond * 0.062
        self.bot.fishing_possible_distance = self.bot.fishing_maximum_x - self.bot.fishing_minimum_x

        if self.bot.fishing_possible_distance <= 0:
            print(f"自动设置错误：计算出的有效钓鱼区域宽度无效 ({self.bot.fishing_possible_distance:.2f})。", flush=True)
            # FIXED: 调用已定义的调试截图方法
            self.bot._save_debug_screenshot("fishing_setup_fail_width")
            return False

        print(f"钓鱼区域自动初始化成功！有效抛竿X范围: {self.bot.fishing_minimum_x:.2f} - {self.bot.fishing_maximum_x:.2f}", flush=True)
        self.bot.fishing_initialized = True
        return True

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
            print("钓鱼模式结束。", flush=True)
            self.bot.fishing_initialized = False
        
    def summoning_loop(self):
        """召唤自动点击循环"""
        while self.bot.running:
            self.bot._click_and_hold(*self.bot.config["summoning"]["spawn"])
            time.sleep(0.05) 
            
    def test_loop(self):
        """测试循环"""
        while self.bot.running:
            self.bot.interact_with_image(self.bot.config["summoning"]["upgrade_6"])
            time.sleep(0.05) 

# --- 主程序入口 ---
def main():
    """
    主程序入口
    """
    try:
        # 假设配置文件中定义了窗口标题
        # 如果需要支持多个窗口，可以在这里加载配置
        # with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        #     config_data = json.load(f)
        # window_titles = config_data.get("window_titles", {"default": "Legends Of Idleon"})
        
        bot = GameBot()
        functions = GameFunctions(bot)
    except Exception as e:
        print(f"初始化失败: {e}", flush=True)
        input("按 Enter 键退出。")
        return

    bot.start_exit_listener()

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
        "14": ("正义（功能未完成）", functions.justice_loop),
        "15": ("钓鱼", functions.fishing_loop),
        "16": ("召唤自动点击", functions.summoning_loop),
    }

    print("请选择功能：")
    for key, (description, _) in sorted(function_map.items(), key=lambda item: int(item[0])):
        print(f"{key}. {description}")
    choice = input(f"请输入选项 (0-{len(function_map) - 1}): ")

    selected_function = function_map.get(choice)
    if selected_function:
        print(f"已选择功能: {selected_function[0]}")
        print("脚本运行中，按下 'delete' 键退出...")
        time.sleep(2) # 留出时间切换到游戏窗口
        selected_function[1]()
    else:
        print("无效选项！", flush=True)

    print("脚本已停止。")


if __name__ == "__main__":
    main()