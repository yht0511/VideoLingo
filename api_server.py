"""
VideoLingo Web API Server
提供 RESTful API 接口用于视频翻译和配音服务
"""

import os
import sys
import asyncio
import uuid
import json
import shutil
from datetime import datetime
import traceback
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
os.environ['PYTHONPATH'] = current_dir

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, HttpUrl, validator
    import uvicorn
    
    # 导入核心模块
    from core.utils.config_utils import load_key, update_key
    from core import (
        _1_ytdlp, _2_asr, _3_1_split_nlp, _3_2_split_meaning,
        _4_1_summarize, _4_2_translate, _5_split_sub, _6_gen_sub,
        _7_sub_into_vid, _8_1_audio_task, _8_2_dub_chunks,
        _9_refer_audio, _10_gen_audio, _11_merge_audio, _12_dub_to_vid
    )
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ API 服务器依赖缺失: {e}")
    DEPENDENCIES_AVAILABLE = False
    
    # 创建一个简单的占位符应用
    class MockApp:
        def __init__(self):
            pass
        def add_middleware(self, *args, **kwargs):
            pass
        def mount(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def post(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def delete(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

if not DEPENDENCIES_AVAILABLE:
    app = MockApp()
else:
    # API 模型定义
    class VideoProcessRequest(BaseModel):
        """视频处理请求模型"""
        url: Optional[HttpUrl] = None
        target_language: str = "简体中文"
        source_language: Optional[str] = None
        enable_dubbing: bool = False
        burn_subtitles: bool = True
        resolution: str = "1080"
        
        @validator('resolution')
        def validate_resolution(cls, v):
            if v not in ['360', '720', '1080', 'best']:
                raise ValueError('Resolution must be one of: 360, 720, 1080, best')
            return v

    class TaskStatus(BaseModel):
        """任务状态模型"""
        task_id: str
        status: str  # pending, processing, completed, failed
        progress: int  # 0-100
        current_step: str
        message: str
        created_at: datetime
        updated_at: datetime
        result: Optional[Dict[str, Any]] = None

    class VideoUploadResponse(BaseModel):
        """视频上传响应模型"""
        task_id: str
        message: str

    class ProcessResponse(BaseModel):
        """处理响应模型"""
        task_id: str
        status: str
        message: str

# 全局变量
app = FastAPI(
    title="VideoLingo API",
    description="视频翻译和配音 API 服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 任务存储 (生产环境建议使用 Redis 或数据库)
tasks: Dict[str, TaskStatus] = {}

# 工作目录配置
WORK_DIR = Path("api_workspace")
WORK_DIR.mkdir(exist_ok=True)

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(WORK_DIR)), name="static")

class VideoProcessor:
    """视频处理核心类"""
    TASK_FILE_NAME = "task.json"
    
    @staticmethod
    def create_task(task_id: str, message: str = "任务已创建") -> TaskStatus:
        """创建新任务"""
        task = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0,
            current_step="初始化",
            message=message,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        tasks[task_id] = task
        # 持久化到磁盘
        try:
            VideoProcessor._save_task_to_disk(task)
        except Exception as e:
            print(f"持久化任务失败: {e}")
        return task
    
    @staticmethod
    def update_task(task_id: str, status: Optional[str] = None, progress: Optional[int] = None, 
                   current_step: Optional[str] = None, message: Optional[str] = None, result: Optional[Dict[str, Any]] = None):
        """更新任务状态"""
        if task_id not in tasks:
            return
        
        task = tasks[task_id]
        if status:
            task.status = status
        if progress is not None:
            task.progress = progress
        if current_step:
            task.current_step = current_step
        if message:
            task.message = message
        if result:
            task.result = result
        
        task.updated_at = datetime.now()
        # 同步保存到磁盘
        try:
            VideoProcessor._save_task_to_disk(task)
        except Exception as e:
            print(f"更新任务持久化失败: {e}")

    @staticmethod
    def _save_task_to_disk(task: 'TaskStatus') -> None:
        """将任务信息保存到 api_workspace/{task_id}/task.json"""
        task_dir = WORK_DIR / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        data = task.model_dump() if hasattr(task, 'model_dump') else task.dict()
        with open(task_dir / VideoProcessor.TASK_FILE_NAME, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _load_tasks_from_disk() -> None:
        """从 api_workspace 读取已有任务，重建内存任务表。正在进行中的任务标记为失败。"""
        recovered = 0
        for item in WORK_DIR.iterdir():
            if not item.is_dir():
                continue
            task_file = item / VideoProcessor.TASK_FILE_NAME
            if not task_file.exists():
                continue
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                task = TaskStatus(**data)
                # 如果上次未完成，标记失败（服务已重启）
                if task.status in {"pending", "processing"}:
                    task.status = "failed"
                    task.message = "服务重启导致任务中断"
                    task.updated_at = datetime.now()
                tasks[task.task_id] = task
                recovered += 1
            except Exception as e:
                print(f"恢复任务失败 {item.name}: {e}")
        if recovered:
            print(f"已从磁盘恢复 {recovered} 个任务")

    @staticmethod
    def _startup_cleanup_leftovers() -> None:
        """清理异常重启遗留的 output_backup_* / 临时 output_* 目录"""
        try:
            for name in os.listdir('.'):
                # 清理 output_backup_*
                if name.startswith('output_backup_') and os.path.isdir(name):
                    try:
                        shutil.rmtree(name)
                        print(f"清理遗留目录: {name}")
                    except Exception as e:
                        print(f"清理遗留目录失败 {name}: {e}")
                # 清理未使用的 output_{task}
                if name.startswith('output_') and name != 'output' and os.path.isdir(name):
                    try:
                        shutil.rmtree(name)
                        print(f"清理遗留临时目录: {name}")
                    except Exception as e:
                        print(f"清理遗留临时目录失败 {name}: {e}")
        except Exception:
            print("启动清理发生异常:\n" + traceback.format_exc())
    
    @staticmethod
    async def process_video_pipeline(task_id: str, request: VideoProcessRequest, input_file: Optional[str] = None):
        """视频处理主流程"""
        # 保存原始目录
        original_dir = os.getcwd()
        
        # 在项目根目录创建临时输出目录
        temp_output_dir = f"output_{task_id}"
        
        # 保存原始的 output 目录名
        backup_output = None
        if os.path.exists("output"):
            backup_output = f"output_backup_{task_id}"
            os.rename("output", backup_output)
        
        try:
            # 创建临时输出目录
            os.makedirs(temp_output_dir, exist_ok=True)
            os.rename(temp_output_dir, "output")  # 重命名为 output
            
            # 步骤1: 处理输入文件
            VideoProcessor.update_task(task_id, "processing", 5, "处理输入文件", "正在下载或处理视频文件...")
            
            if request.url:
                # 从 URL 下载视频
                await asyncio.get_event_loop().run_in_executor(
                    None, _1_ytdlp.download_video_ytdlp, str(request.url), "output", request.resolution
                )
                video_file = _1_ytdlp.find_video_files()
            elif input_file:
                # 复制上传的文件
                video_filename = Path(input_file).name
                target_path = Path("output") / video_filename
                shutil.copy2(input_file, target_path)
                video_file = str(target_path)
            else:
                raise Exception("必须提供视频 URL 或上传文件")
            
            # 配置语言设置
            if request.source_language:
                update_key("whisper.language", request.source_language)
            update_key("target_language", request.target_language)
            update_key("burn_subtitles", request.burn_subtitles)
            
            # 步骤2: 语音识别
            VideoProcessor.update_task(task_id, "processing", 15, "语音识别", "正在使用 WhisperX 进行语音识别...")
            await asyncio.get_event_loop().run_in_executor(None, _2_asr.transcribe)
            
            # 步骤3: 句子分割
            VideoProcessor.update_task(task_id, "processing", 25, "句子分割", "正在使用 NLP 和 LLM 进行句子分割...")
            await asyncio.get_event_loop().run_in_executor(None, _3_1_split_nlp.split_by_spacy)
            await asyncio.get_event_loop().run_in_executor(None, _3_2_split_meaning.split_sentences_by_meaning)
            
            # 步骤4: 总结和翻译
            VideoProcessor.update_task(task_id, "processing", 45, "总结和翻译", "正在进行总结和多步翻译...")
            await asyncio.get_event_loop().run_in_executor(None, _4_1_summarize.get_summary)
            
            # 关键修改：将翻译任务放在执行器中运行，避免阻塞
            VideoProcessor.update_task(task_id, "processing", 50, "翻译处理", "正在进行文本翻译，这可能需要一些时间...")
            try:
                # 设置翻译超时时间（10分钟）
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, _4_2_translate.translate_all),
                    timeout=600  # 10分钟超时
                )
            except asyncio.TimeoutError:
                raise Exception("翻译过程超时，请检查网络连接和 API 配置")
            except Exception as e:
                raise Exception(f"翻译过程出错: {str(e)}")
            
            # 步骤5: 字幕处理和对齐
            VideoProcessor.update_task(task_id, "processing", 65, "字幕处理", "正在处理和对齐字幕...")
            await asyncio.get_event_loop().run_in_executor(None, _5_split_sub.split_for_sub_main)
            await asyncio.get_event_loop().run_in_executor(None, _6_gen_sub.align_timestamp_main)
            
            # 步骤6: 合并字幕到视频
            VideoProcessor.update_task(task_id, "processing", 75, "合并字幕", "正在将字幕合并到视频...")
            await asyncio.get_event_loop().run_in_executor(None, _7_sub_into_vid.merge_subtitles_to_video)
            
            result_files = {
                "video_with_subtitles": f"/static/{task_id}/output_sub.mp4",
                "source_subtitles": f"/static/{task_id}/src.srt",
                "translated_subtitles": f"/static/{task_id}/trans.srt"
            }
            
            # 如果启用配音
            if request.enable_dubbing:
                # 步骤7: 生成音频任务
                VideoProcessor.update_task(task_id, "processing", 80, "生成音频任务", "正在生成音频任务...")
                await asyncio.get_event_loop().run_in_executor(None, _8_1_audio_task.gen_audio_task_main)
                await asyncio.get_event_loop().run_in_executor(None, _8_2_dub_chunks.gen_dub_chunks)
                
                # 步骤8: 提取参考音频
                VideoProcessor.update_task(task_id, "processing", 85, "提取参考音频", "正在提取参考音频...")
                await asyncio.get_event_loop().run_in_executor(None, _9_refer_audio.extract_refer_audio_main)
                
                # 步骤9: 生成音频
                VideoProcessor.update_task(task_id, "processing", 90, "生成音频", "正在生成配音音频...")
                await asyncio.get_event_loop().run_in_executor(None, _10_gen_audio.gen_audio)
                
                # 步骤10: 合并音频
                VideoProcessor.update_task(task_id, "processing", 95, "合并音频", "正在合并完整音频...")
                await asyncio.get_event_loop().run_in_executor(None, _11_merge_audio.merge_full_audio)
                
                # 步骤11: 合并配音到视频
                VideoProcessor.update_task(task_id, "processing", 98, "合并配音", "正在将配音合并到视频...")
                await asyncio.get_event_loop().run_in_executor(None, _12_dub_to_vid.merge_video_audio)
                
                result_files["video_with_dubbing"] = f"/static/{task_id}/output_dub.mp4"
                result_files["dubbing_audio"] = f"/static/{task_id}/dub.mp3"
            
            # 将结果文件移动到 API 工作空间
            result_dir = WORK_DIR / task_id
            result_dir.mkdir(exist_ok=True)
            
            if os.path.exists("output"):
                for file in os.listdir("output"):
                    src_path = os.path.join("output", file)
                    if os.path.isfile(src_path):
                        dst_path = result_dir / file
                        shutil.copy2(src_path, dst_path)
            
            # 完成
            VideoProcessor.update_task(
                task_id, 
                "completed", 
                100, 
                "完成", 
                "视频处理完成！", 
                result_files
            )
            
        except Exception as e:
            error_msg = f"处理过程中出现错误: {str(e)}"
            VideoProcessor.update_task(task_id, "failed", -1, "错误", error_msg)
            print(f"Task {task_id} failed: {error_msg}")
            
        finally:
            # 恢复原始目录结构
            try:
                # 删除临时的 output 目录
                if os.path.exists("output"):
                    shutil.rmtree("output")
                
                # 恢复原始的 output 目录
                if backup_output and os.path.exists(backup_output):
                    os.rename(backup_output, "output")
            except Exception as cleanup_error:
                print(f"清理临时文件时出错: {cleanup_error}")
            
            # 确保在原始目录
            os.chdir(original_dir)

# 在应用启动时执行：清理遗留目录并从磁盘恢复任务
@app.on_event("startup")
async def _on_startup():
    try:
        VideoProcessor._startup_cleanup_leftovers()
        VideoProcessor._load_tasks_from_disk()
    except Exception:
        print("启动恢复发生异常:\n" + traceback.format_exc())

# API 路由定义
@app.get("/", summary="API 信息")
async def root():
    """获取 API 基本信息"""
    return {
        "message": "VideoLingo API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.post("/api/v1/process-url", response_model=ProcessResponse, summary="处理 YouTube URL")
async def process_video_url(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    通过 YouTube URL 处理视频
    
    - **url**: YouTube 视频链接
    - **target_language**: 目标语言（默认: 简体中文）
    - **source_language**: 源语言（可选，自动检测）
    - **enable_dubbing**: 是否启用配音（默认: False）
    - **burn_subtitles**: 是否烧录字幕（默认: True）
    - **resolution**: 下载分辨率（默认: 1080）
    """
    if not request.url:
        raise HTTPException(status_code=400, detail="必须提供视频 URL")
    
    task_id = str(uuid.uuid4())
    VideoProcessor.create_task(task_id, f"准备处理视频: {request.url}")
    
    # 添加后台任务
    background_tasks.add_task(VideoProcessor.process_video_pipeline, task_id, request)
    
    return ProcessResponse(
        task_id=task_id,
        status="pending",
        message="任务已创建，开始处理中..."
    )

@app.post("/api/v1/upload", response_model=VideoUploadResponse, summary="上传视频文件")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("简体中文"),
    source_language: Optional[str] = Form(None),
    enable_dubbing: bool = Form(False),
    burn_subtitles: bool = Form(True)
):
    """
    上传视频文件进行处理
    
    - **file**: 视频文件
    - **target_language**: 目标语言
    - **source_language**: 源语言（可选）
    - **enable_dubbing**: 是否启用配音
    - **burn_subtitles**: 是否烧录字幕
    """
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")
        
    # 验证文件类型
    allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    task_id = str(uuid.uuid4())
    
    # 保存上传的文件
    upload_dir = WORK_DIR / task_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # 创建处理请求
    request = VideoProcessRequest(
        target_language=target_language,
        source_language=source_language,
        enable_dubbing=enable_dubbing,
        burn_subtitles=burn_subtitles
    )
    
    VideoProcessor.create_task(task_id, f"文件已上传: {file.filename}")
    
    # 添加后台任务
    background_tasks.add_task(VideoProcessor.process_video_pipeline, task_id, request, str(file_path))
    
    return VideoUploadResponse(
        task_id=task_id,
        message=f"文件 {file.filename} 上传成功，开始处理中..."
    )

@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatus, summary="查询任务状态")
async def get_task_status(task_id: str):
    """获取指定任务的处理状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]

@app.get("/api/v1/tasks", response_model=List[TaskStatus], summary="获取所有任务")
async def get_all_tasks(limit: int = 50, status: Optional[str] = None):
    """
    获取任务列表
    
    - **limit**: 返回任务数量限制
    - **status**: 按状态过滤（pending, processing, completed, failed）
    """
    task_list = list(tasks.values())
    
    if status:
        task_list = [task for task in task_list if task.status == status]
    
    # 按创建时间倒序排列
    task_list.sort(key=lambda x: x.created_at, reverse=True)
    
    return task_list[:limit]

@app.delete("/api/v1/tasks/{task_id}", summary="删除任务")
async def delete_task(task_id: str):
    """删除指定任务及其相关文件"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 删除任务目录
    task_dir = WORK_DIR / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    
    # 删除任务记录
    del tasks[task_id]
    
    return {"message": f"任务 {task_id} 已删除"}

@app.get("/api/v1/download/{task_id}/{file_type}", summary="下载结果文件")
async def download_file(task_id: str, file_type: str):
    """
    下载处理结果文件
    
    - **task_id**: 任务ID
    - **file_type**: 文件类型 (video_sub, video_dub, src_srt, trans_srt, dub_audio)
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    file_mapping = {
        "video_sub": "output_sub.mp4",
        "video_dub": "output_dub.mp4", 
        "src_srt": "src.srt",
        "trans_srt": "trans.srt",
        "dub_audio": "dub.mp3"
    }
    
    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail="不支持的文件类型")
    
    file_path = WORK_DIR / task_id / file_mapping[file_type]
    
    # 添加调试信息
    print(f"Debug: Looking for file at: {file_path}")
    print(f"Debug: File exists: {file_path.exists()}")
    if not file_path.exists():
        # 列出目录中的所有文件进行调试
        task_dir = WORK_DIR / task_id
        if task_dir.exists():
            files_in_dir = list(task_dir.iterdir())
            print(f"Debug: Files in task directory: {files_in_dir}")
        else:
            print(f"Debug: Task directory does not exist: {task_dir}")
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_mapping[file_type]}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=str(file_path),
        filename=f"{task_id}_{file_type}{file_path.suffix}",
        media_type='application/octet-stream'
    )

@app.get("/api/v1/config", summary="获取配置信息")
async def get_config():
    """获取当前配置信息"""
    return {
        "supported_languages": {
            "input": ["en", "zh", "ja", "fr", "de", "es", "ru", "it"],
            "output": "支持所有语言的自然语言描述"
        },
        "supported_formats": ["mp4", "mov", "avi", "mkv", "flv", "wmv", "webm"],
        "tts_methods": ["azure_tts", "openai_tts", "edge_tts", "gpt_sovits", "fish_tts"],
        "max_resolution": "1080p",
        "features": {
            "subtitle_generation": True,
            "dubbing": True,
            "multi_language": True,
            "batch_processing": False
        }
    }

@app.post("/api/v1/config", summary="更新配置")
async def update_config(config: Dict[str, Any]):
    """更新系统配置"""
    try:
        for key, value in config.items():
            update_key(key, value)
        return {"message": "配置更新成功", "updated_keys": list(config.keys())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"配置更新失败: {str(e)}")

# 健康检查
@app.get("/health", summary="健康检查")
async def health_check():
    """API 健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tasks_count": len(tasks),
        "active_tasks": len([t for t in tasks.values() if t.status == "processing"])
    }

if __name__ == "__main__":
    print("🚀 启动 VideoLingo API 服务器...")
    print("📝 API 文档: http://localhost:8000/docs")
    print("🔄 健康检查: http://localhost:8000/health")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
