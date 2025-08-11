#!/bin/bash
"""
Docker 容器启动脚本
同时启动 Streamlit 界面和 API 服务器
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path

def start_streamlit():
    """启动 Streamlit 服务器"""
    print("🎬 启动 Streamlit 界面...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "st.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false"
        ])
    except Exception as e:
        print(f"❌ Streamlit 启动失败: {e}")

def start_api_server():
    """启动 API 服务器"""
    print("🚀 启动 API 服务器...")
    try:
        # 等待一下让系统准备好
        time.sleep(3)
        
        # 启动 API 服务器
        import uvicorn
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=1,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ API 服务器启动失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🎬 VideoLingo Docker 容器启动")
    print("📝 Streamlit 界面: http://localhost:8501")
    print("🔗 API 文档: http://localhost:8000/docs")
    print("=" * 60)
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    # 创建 API 工作目录
    work_dir = Path("api_workspace")
    work_dir.mkdir(exist_ok=True)
    
    try:
        # 在后台线程中启动 API 服务器
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # 在主线程中启动 Streamlit（这样容器会保持运行）
        start_streamlit()
        
    except KeyboardInterrupt:
        print("\n👋 容器停止中...")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
