#!/usr/bin/env python3
"""
VideoLingo API 服务器启动脚本
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def install_api_dependencies():
    """安装 API 所需的额外依赖"""
    print("🔧 检查并安装 API 依赖...")
    
    try:
        import fastapi
        import uvicorn
        print("✅ API 依赖已安装")
    except ImportError:
        print("📦 安装 API 依赖...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "api_requirements.txt"
        ])
        print("✅ API 依赖安装完成")

def check_core_dependencies():
    """检查核心依赖是否已安装"""
    print("🔍 检查核心依赖...")
    
    try:
        import streamlit
        import whisperx
        import torch
        print("✅ 核心依赖已安装")
        return True
    except ImportError:
        print("❌ 核心依赖未安装，请先运行: python install.py")
        return False

def start_api_server():
    """启动 API 服务器"""
    print("🚀 启动 VideoLingo API 服务器...")
    
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(Path.cwd())
    
    try:
        # 导入并启动服务器
        import uvicorn
        
        print("📝 API 文档将在以下地址可用:")
        print("   - Swagger UI: http://localhost:8000/docs")
        print("   - ReDoc: http://localhost:8000/redoc")
        print("   - 测试界面: http://localhost:8000/static/../web_interface.html")
        print("🔄 健康检查: http://localhost:8000/health")
        print("\n💡 按 Ctrl+C 停止服务器")
        
        # 复制测试界面到静态文件目录
        web_interface_path = Path("web_interface.html")
        if web_interface_path.exists():
            static_dir = Path("api_workspace")
            static_dir.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(web_interface_path, static_dir / "web_interface.html")
        
        # 自动打开浏览器
        try:
            webbrowser.open("http://localhost:8000/docs")
        except:
            pass
        
        # 启动服务器
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=1,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("🎬 VideoLingo API 服务器启动器")
    print("=" * 60)
    
    # 检查核心依赖
    if not check_core_dependencies():
        print("\n请先安装核心依赖:")
        print("  python install.py")
        return False
    
    # 安装 API 依赖
    install_api_dependencies()
    
    print("\n" + "=" * 60)
    
    # 启动服务器
    return start_api_server()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
