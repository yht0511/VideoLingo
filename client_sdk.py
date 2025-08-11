"""
VideoLingo API 客户端 SDK
提供简单易用的 Python 接口来调用 VideoLingo API 服务
"""

import requests
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

class VideoLingoClient:
    """VideoLingo API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: API 服务器基础 URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        response = self.session.get(f"{self.base_url}/api/v1/config")
        response.raise_for_status()
        return response.json()
    
    def process_youtube_url(
        self, 
        url: str,
        target_language: str = "简体中文",
        source_language: Optional[str] = None,
        enable_dubbing: bool = False,
        burn_subtitles: bool = True,
        resolution: str = "1080"
    ) -> str:
        """
        处理 YouTube 视频
        
        Args:
            url: YouTube 视频链接
            target_language: 目标语言
            source_language: 源语言（可选）
            enable_dubbing: 是否启用配音
            burn_subtitles: 是否烧录字幕
            resolution: 下载分辨率
            
        Returns:
            任务 ID
        """
        data = {
            "url": url,
            "target_language": target_language,
            "source_language": source_language,
            "enable_dubbing": enable_dubbing,
            "burn_subtitles": burn_subtitles,
            "resolution": resolution
        }
        
        response = self.session.post(f"{self.base_url}/api/v1/process-url", json=data)
        response.raise_for_status()
        result = response.json()
        return result["task_id"]
    
    def upload_video(
        self,
        file_path: str,
        target_language: str = "简体中文",
        source_language: Optional[str] = None,
        enable_dubbing: bool = False,
        burn_subtitles: bool = True
    ) -> str:
        """
        上传并处理视频文件
        
        Args:
            file_path: 视频文件路径
            target_language: 目标语言
            source_language: 源语言（可选）
            enable_dubbing: 是否启用配音
            burn_subtitles: 是否烧录字幕
            
        Returns:
            任务 ID
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f, 'video/*')}
            data = {
                'target_language': target_language,
                'source_language': source_language or '',
                'enable_dubbing': enable_dubbing,
                'burn_subtitles': burn_subtitles
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/upload", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            return result["task_id"]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        response = self.session.get(f"{self.base_url}/api/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_all_tasks(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有任务"""
        params = {'limit': limit}
        if status:
            params['status'] = status
            
        response = self.session.get(f"{self.base_url}/api/v1/tasks", params=params)
        response.raise_for_status()
        return response.json()
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        response = self.session.delete(f"{self.base_url}/api/v1/tasks/{task_id}")
        response.raise_for_status()
        return True
    
    def download_file(self, task_id: str, file_type: str, save_path: str) -> bool:
        """
        下载结果文件
        
        Args:
            task_id: 任务 ID
            file_type: 文件类型 (video_sub, video_dub, src_srt, trans_srt, dub_audio)
            save_path: 保存路径
            
        Returns:
            是否下载成功
        """
        response = self.session.get(f"{self.base_url}/api/v1/download/{task_id}/{file_type}")
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    
    def wait_for_completion(
        self, 
        task_id: str, 
        timeout: int = 3600, 
        poll_interval: int = 5,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        等待任务完成
        
        Args:
            task_id: 任务 ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
            callback: 状态更新回调函数
            
        Returns:
            最终任务状态
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            
            if callback:
                callback(status)
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"任务 {task_id} 在 {timeout} 秒内未完成")
    
    def process_and_wait(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        target_language: str = "简体中文",
        source_language: Optional[str] = None,
        enable_dubbing: bool = False,
        burn_subtitles: bool = True,
        resolution: str = "1080",
        timeout: int = 3600,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        处理视频并等待完成
        
        Args:
            url: YouTube 视频链接（与 file_path 二选一）
            file_path: 视频文件路径（与 url 二选一）
            其他参数同上
            
        Returns:
            最终任务状态
        """
        if url:
            task_id = self.process_youtube_url(
                url, target_language, source_language, 
                enable_dubbing, burn_subtitles, resolution
            )
        elif file_path:
            task_id = self.upload_video(
                file_path, target_language, source_language,
                enable_dubbing, burn_subtitles
            )
        else:
            raise ValueError("必须提供 url 或 file_path 参数")
        
        return self.wait_for_completion(task_id, timeout, callback=callback)


# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = VideoLingoClient()
    
    # 健康检查
    try:
        health = client.health_check()
        print("✅ API 服务器运行正常")
        print(f"状态: {health}")
    except Exception as e:
        print(f"❌ API 服务器连接失败: {e}")
        exit(1)
    
    # 示例：处理 YouTube 视频
    def progress_callback(status):
        print(f"📊 进度: {status['progress']}% - {status['current_step']} - {status['message']}")
    
    try:
        # 你可以替换为实际的 YouTube URL
        # result = client.process_and_wait(
        #     url="https://www.youtube.com/watch?v=example",
        #     target_language="简体中文",
        #     enable_dubbing=True,
        #     callback=progress_callback
        # )
        
        # 或者上传本地文件
        # result = client.process_and_wait(
        #     file_path="path/to/your/video.mp4",
        #     target_language="简体中文",
        #     enable_dubbing=True,
        #     callback=progress_callback
        # )
        
        print("🎉 处理完成！")
        print(f"结果: {result}")
        
        # 下载结果文件
        if result['status'] == 'completed' and result.get('result'):
            task_id = result['task_id']
            
            # 下载带字幕的视频
            # client.download_file(task_id, 'video_sub', 'output_with_subtitles.mp4')
            # print("📥 带字幕视频已下载")
            
            # 如果启用了配音，下载带配音的视频
            # if 'video_with_dubbing' in result['result']:
            #     client.download_file(task_id, 'video_dub', 'output_with_dubbing.mp4')
            #     print("📥 带配音视频已下载")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
    
    # 获取所有任务
    tasks = client.get_all_tasks(limit=10)
    print(f"\n📋 最近 {len(tasks)} 个任务:")
    for task in tasks:
        print(f"  - {task['task_id']}: {task['status']} ({task['current_step']})")
