// src/App.jsx

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

// --- 单个功能控制组件 ---
function FunctionControl({ functionKey, name, isRunning, onAction }) {
    return (
        <div className={`function-control ${isRunning ? 'running' : ''}`}>
            <div className="function-control-status">
                <div className={`status-indicator ${isRunning ? 'running' : 'stopped'}`}></div>
                <span className="function-control-name">{name}</span> {/* 直接使用传入的中文名 */}
            </div>
            <div>
                {!isRunning && <button className="btn-start" onClick={() => onAction(functionKey, 'start')}>开始</button>}
                {isRunning && <button className="btn-stop" onClick={() => onAction(functionKey, 'stop')}>停止</button>}
            </div>
        </div>
    );
}

// --- 游戏截图显示组件 ---
function ScreenshotViewer() {
    const [imageUrl, setImageUrl] = useState(null);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const fetchScreenshot = useCallback(async () => {
        if (!isLoading) { // 只有在不处于加载状态时才将 isLoading 设置为 true
            setIsLoading(true);
        }
        try {
            const response = await axios.get(`${API_BASE_URL}/screenshot/default?region=full`, {
                responseType: 'arraybuffer'
            });
            const blob = new Blob([response.data], { type: 'image/png' });
            const newUrl = URL.createObjectURL(blob);

            // --- 解决抖动的核心逻辑 ---
            const img = new Image();
            img.src = newUrl;
            img.onload = () => {
                // 新图片加载完成后，才更新显示的 URL
                setImageUrl(currentUrl => {
                    // 销毁上一个旧的 URL
                    if (currentUrl) {
                        URL.revokeObjectURL(currentUrl);
                    }
                    return newUrl; // 使用新的 URL
                });
                setError(null);
                setIsLoading(false);
            };
            img.onerror = () => {
                // 如果新图片加载失败
                setError('加载截图失败。请确保游戏窗口可见。');
                setIsLoading(false);
                URL.revokeObjectURL(newUrl); // 别忘了清理失败的 URL
            }
            // -------------------------

        } catch (err) {
            console.error(`获取截图时出错:`, err);
            setError('加载截图失败。请确保游戏窗口可见。');
            setImageUrl(currentUrl => {
                if (currentUrl) URL.revokeObjectURL(currentUrl);
                return null;
            });
            setIsLoading(false);
        }
    }, [isLoading]); // 依赖 isLoading 来避免重复设置

    useEffect(() => {
        const intervalId = setInterval(fetchScreenshot, 2000); // 可以适当缩短刷新间隔，比如2秒
        // 组件加载时立即执行一次
        fetchScreenshot();
        
        return () => {
            clearInterval(intervalId);
            if (imageUrl) URL.revokeObjectURL(imageUrl);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []); // 初始加载只执行一次

    return (
        <div className="card">
            <h2>游戏截图</h2>
            <div className="screenshot-controls">
                <label htmlFor="screenshot-region">区域:</label>
                <select id="screenshot-region">
                    <option value="full">整个窗口</option>
                </select>
                <button className="btn-refresh" onClick={fetchScreenshot}>立即刷新</button>
            </div>
            <div className="screenshot-container">
                {/* 只有在第一次加载时显示 "正在加载" */}
                {isLoading && !imageUrl && <p>正在加载截图...</p>}
                {error && <p style={{ color: 'var(--error-color)' }}>{error}</p>}
                {/* 即使在加载新图时，依然显示旧图，避免空白 */}
                {imageUrl && <img src={imageUrl} alt="游戏截图" />}
            </div>
        </div>
    );
}

// --- 主应用组件 ---
function App() {
    // status 现在会是 { "afk_loop": { name: "AFK", running: false }, ... }
    const [status, setStatus] = useState({});
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const fetchStatus = useCallback(async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/status`);
            setStatus(response.data);
            setError(null);
        } catch (err) {
            console.error("获取状态时出错:", err);
            setError('无法获取状态。后端服务是否已启动？');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        const intervalId = setInterval(fetchStatus, 3000);
        return () => clearInterval(intervalId);
    }, [fetchStatus]);

    // onAction 现在接收 functionKey (英文函数名)
    const handleFunctionAction = async (functionKey, action) => {
        try {
            await axios.post(`${API_BASE_URL}/${action}/${functionKey}`);
            setTimeout(fetchStatus, 300);
        } catch (err) {
            alert(`操作失败，请查看控制台。`);
        }
    };

    const handleStopAll = async () => {
        try {
            await axios.post(`${API_BASE_URL}/stop/all`);
            setTimeout(fetchStatus, 300);
        } catch (err) {
            alert('停止所有功能失败，请查看控制台。');
        }
    };
    
    // 我们现在遍历 status 对象的键 (keys)，这些键是英文函数名
    const functionKeys = Object.keys(status);

    return (
        <div className="App">
            <header className="App-header">
                <h1>Idleon 自动化控制面板</h1>
                {error && <p className="error-message">{error}</p>}
                <button className="btn-stop-all" onClick={handleStopAll}>停止所有任务</button>
            </header>

            <div className="card">
                <h2>自动化任务</h2>
                <div className="functions-list">
                    {isLoading ? (
                        <p>正在加载任务列表...</p>
                    ) : functionKeys.length > 0 ? (
                        // 遍历 functionKeys
                        functionKeys.map(funcKey => (
                            <FunctionControl
                                key={funcKey}
                                functionKey={funcKey} // 把英文函数名作为 key 和参数
                                name={status[funcKey].name} // 把中文名传给 name 属性
                                isRunning={status[funcKey].running}
                                onAction={handleFunctionAction}
                            />
                        ))
                    ) : (
                        <p>无可用任务。</p>
                    )}
                </div>
            </div>

            <ScreenshotViewer />
        </div>
    );
}

export default App;