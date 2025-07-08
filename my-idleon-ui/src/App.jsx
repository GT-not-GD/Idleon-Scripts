// src/App.jsx (纯 JavaScript 版本)

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css'; // 你的样式文件

// --- 后端 API 地址 ---
const API_BASE_URL = 'http://127.0.0.1:8000';

// --- 单个功能控制组件 ---
// (这个组件没有使用任何 TypeScript 特性，所以不需要修改)
function FunctionControl({ functionName, isRunning, onAction }) {
    // 将函数名（如 "afk_fast_forward_loop"）转换为更易读的格式（如 "Afk Fast Forward Loop"）
    const readableName = functionName
        .replace(/_loop$/, '') // 移除末尾的 _loop
        .replace(/_/g, ' ')   // 将下划线替换为空格
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // 首字母大写
        .join(' ');

    return (
        <div style={{
            border: '1px solid #eee',
            padding: '10px 15px',
            margin: '5px 0',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: isRunning ? '#e8f5e9' : '#fff', // 运行时显示淡绿色背景
            transition: 'background-color 0.3s ease'
        }}>
            <span style={{ flex: 1 }}>
                <strong>{readableName}</strong>
                <span style={{ marginLeft: '15px', color: isRunning ? '#2e7d32' : '#d32f2f', fontWeight: 'bold' }}>
                    {isRunning ? '● 运行中' : '● 已停止'}
                </span>
            </span>
            <div>
                {!isRunning && (
                    <button onClick={() => onAction(functionName, 'start')} style={{ marginRight: '5px' }}>
                        开始
                    </button>
                )}
                {isRunning && (
                    <button onClick={() => onAction(functionName, 'stop')} style={{ marginRight: '5px' }}>
                        停止
                    </button>
                )}
            </div>
        </div>
    );
}

// --- 游戏截图显示组件 ---
function ScreenshotViewer() {
    // 移除 <string | null> 类型
    const [imageUrl, setImageUrl] = useState(null);
    const [error, setError] = useState(null);
    const [screenshotRegion, setScreenshotRegion] = useState('full');

    const fetchScreenshot = async () => {
        try {
            const regionParam = screenshotRegion === 'full' ? 'full' : screenshotRegion;
            const response = await axios.get(`${API_BASE_URL}/screenshot/default?region=${regionParam}`, {
                responseType: 'arraybuffer'
            });
            const blob = new Blob([response.data], { type: 'image/png' });
            const url = URL.createObjectURL(blob);
            if (imageUrl) {
                URL.revokeObjectURL(imageUrl);
            }
            setImageUrl(url);
            setError(null);
        } catch (err) {
            console.error(`获取截图时出错:`, err);
            setError('加载截图失败。请确保游戏正在运行并且窗口可见。');
            setImageUrl(null);
        }
    };

    useEffect(() => {
        fetchScreenshot();
        const intervalId = setInterval(fetchScreenshot, 5000);
        return () => {
            clearInterval(intervalId);
            if (imageUrl) {
                URL.revokeObjectURL(imageUrl);
            }
        };
    }, [screenshotRegion]);

    // 移除事件参数的类型
    const handleRegionChange = (event) => {
        setScreenshotRegion(event.target.value);
    };

    return (
        <div style={{ marginTop: '30px', border: '1px solid #ddd', padding: '15px', borderRadius: '8px', width: '100%', maxWidth: '800px' }}>
            <h2>游戏截图</h2>
            <div style={{ marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label htmlFor="screenshot-region">区域:</label>
                <select id="screenshot-region" value={screenshotRegion} onChange={handleRegionChange}>
                    <option value="full">整个窗口</option>
                </select>
                <button onClick={fetchScreenshot}>立即刷新</button>
            </div>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            {imageUrl ? (
                <img src={imageUrl} alt="游戏截图" style={{ width: '100%', border: '1px solid #eee' }} />
            ) : (
                <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f9f9f9', border: '1px dashed #ccc' }}>
                    <p>正在加载截图...</p>
                </div>
            )}
        </div>
    );
}

// --- 主应用组件 ---
function App() {
    const [status, setStatus] = useState({});

    const fetchStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/status`);
            setStatus(response.data);
        } catch (error) {
            console.error("获取状态时出错:", error);
            setStatus({ error: '无法获取状态。后端服务是否已启动？' });
        }
    };

    useEffect(() => {
        fetchStatus();
        const intervalId = setInterval(fetchStatus, 3000);
        return () => clearInterval(intervalId);
    }, []);

    const handleFunctionAction = async (functionName, action) => {
        const url = `${API_BASE_URL}/${action}/${functionName}`;
        try {
            await axios.post(url);
            setTimeout(fetchStatus, 500);
        } catch (error) {
            console.error(`执行 ${action} '${functionName}' 时出错:`, error);
            alert(`操作 ${functionName} 失败，请查看控制台获取详情。`);
        }
    };

    const handleStopAll = async () => {
        try {
            await axios.post(`${API_BASE_URL}/stop/all`);
            setTimeout(fetchStatus, 500);
        } catch (error) {
            console.error("停止所有功能时出错:", error);
            alert('停止所有功能失败，请查看控制台。');
        }
    };

    const functionNames = Object.keys(status).filter(key => key !== 'error');

    return (
        <div className="App">
            <header className="App-header">
                <h1>Idleon 自动化控制面板</h1>
                {status.error && <p style={{ color: 'red' }}>{status.error}</p>}
                <div style={{ margin: '10px 0' }}>
                    <button onClick={handleStopAll} style={{ padding: '10px 20px', fontSize: '1.1em', cursor: 'pointer', backgroundColor: '#c62828' }}>
                        停止所有任务
                    </button>
                </div>
            </header>

            <main>
                <div className="functions-list">
                    <h2>自动化任务</h2>
                    {functionNames.length > 0 ? (
                        functionNames.map(funcName => (
                            <FunctionControl
                                key={funcName}
                                functionName={funcName}
                                isRunning={status[funcName]?.running || false}
                                onAction={handleFunctionAction}
                            />
                        ))
                    ) : (
                        <p>正在加载任务列表或无可用任务...</p>
                    )}
                </div>

                <ScreenshotViewer />
            </main>
        </div>
    );
}

export default App;