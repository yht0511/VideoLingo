import os
import io
import json
import time
import librosa
import soundfile as sf
from rich import print as rprint
from core.utils import *
from core.utils.models import *

try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    rprint("[yellow]⚠️ Replicate library not installed. Run: pip install replicate[/yellow]")

OUTPUT_LOG_DIR = "output/log"

from typing import Optional

def transcribe_audio_replicate(raw_audio_path: str, vocal_audio_path: str, start: Optional[float] = None, end: Optional[float] = None):
    """
    使用 Replicate 平台的 WhisperX 模型进行语音识别
    
    Args:
        raw_audio_path: 原始音频文件路径
        vocal_audio_path: 人声分离后的音频文件路径  
        start: 音频片段开始时间（秒）
        end: 音频片段结束时间（秒）
        
    Returns:
        dict: 包含识别结果的字典
    """
    if not REPLICATE_AVAILABLE:
        raise ImportError("Replicate library not available. Please install with: pip install replicate")
    
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    LOG_FILE = f"{OUTPUT_LOG_DIR}/replicate_{start}_{end}.json"
    
    # 检查是否已有缓存结果
    if os.path.exists(LOG_FILE):
        rprint(f"[green]📁 Loading cached result from {LOG_FILE}[/green]")
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 获取配置
    api_token = load_key("whisper.replicate_api_token")
    if not api_token or api_token == "your_replicate_api_token":
        raise ValueError("Please set your Replicate API token in config.yaml")
    
    # 设置环境变量
    os.environ["REPLICATE_API_TOKEN"] = api_token
    
    WHISPER_LANGUAGE = load_key("whisper.language")
    
    # 加载和处理音频
    rprint(f"[cyan]🎵 Loading audio file: {vocal_audio_path}[/cyan]")
    y, sr = librosa.load(vocal_audio_path, sr=16000)
    audio_duration = len(y) / sr
    
    # 处理音频片段
    if start is None or end is None:
        start = 0
        end = audio_duration
        rprint(f"[cyan]🎵 Processing full audio: {audio_duration:.2f}s[/cyan]")
    else:
        rprint(f"[cyan]🎵 Processing audio segment: {start:.2f}s - {end:.2f}s[/cyan]")
        
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_slice = y[start_sample:end_sample]
    
    # 创建临时音频文件
    temp_audio_file = f"temp_audio_{int(time.time())}_{start}_{end}.wav"
    
    try:
        # 保存音频片段为临时文件
        sf.write(temp_audio_file, y_slice, sr, format='WAV', subtype='PCM_16')
        
        # 准备 Replicate 输入
        input_data = {
            "audio_file": open(temp_audio_file, "rb"),
            "align_output": True
        }
        
        # 添加语言参数（如果支持）
        if WHISPER_LANGUAGE and WHISPER_LANGUAGE != "auto":
            # WhisperX 模型可能支持语言参数，根据模型文档调整
            pass  # 当前模型可能不支持直接指定语言
        
        start_time = time.time()
        rprint(f"[cyan]🎤 Transcribing audio with Replicate WhisperX (language: {WHISPER_LANGUAGE})...[/cyan]")
        
        # 调用 Replicate API
        output = replicate.run(
            "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb",
            input=input_data
        )
        
        # 转换 Replicate 输出格式为 VideoLingo 期望的格式
        # Replicate 返回的是 list 格式，需要转换为 dict 格式
        if isinstance(output, list):
            converted_output = {'segments': []}
            
            for segment in output:
                # 清理和标准化文本
                text = segment.get('text', '').strip()
                start_time = segment.get('start', 0)
                end_time = segment.get('end', start_time + 1)
                
                # 创建符合 VideoLingo 期望的 segment 格式
                converted_segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text,
                    'words': []
                }
                
                # 将整个 segment 作为一个大的 "word"，保持时间戳的准确性
                # 这样避免了人为分割单词造成的时间戳错位问题
                if text:
                    converted_segment['words'].append({
                        'word': text,
                        'start': start_time,
                        'end': end_time
                    })
                
                converted_output['segments'].append(converted_segment)
            
            output = converted_output
        else:
            # 如果已经是 dict 格式，确保有正确的结构
            if 'segments' not in output:
                output = {'segments': []}
        
        # 调整时间戳（如果是音频片段）
        if start is not None and start > 0:
            for segment in output.get('segments', []):
                if 'start' in segment:
                    segment['start'] += start
                if 'end' in segment:
                    segment['end'] += start
                    
                # 调整词级别时间戳
                for word in segment.get('words', []):
                    if 'start' in word:
                        word['start'] += start
                    if 'end' in word:
                        word['end'] += start
        
        # 保存结果到缓存
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        
        elapsed_time = time.time() - start_time
        rprint(f"[green]✓ Replicate transcription completed in {elapsed_time:.2f} seconds[/green]")
        
        return output
        
    except Exception as e:
        rprint(f"[red]❌ Replicate transcription failed: {str(e)}[/red]")
        raise
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)


def test_replicate_connection():
    """测试 Replicate 连接"""
    if not REPLICATE_AVAILABLE:
        return False
        
    try:
        api_token = load_key("whisper.replicate_api_token")
        if not api_token or api_token == "your_replicate_api_token":
            rprint("[red]❌ Replicate API token not configured[/red]")
            return False
            
        os.environ["REPLICATE_API_TOKEN"] = api_token
        
        # 简单的连接测试
        rprint("[cyan]🔗 Testing Replicate connection...[/cyan]")
        
        # 这里可以添加一个简单的 API 调用来测试连接
        # 由于没有免费的测试端点，我们只验证 token 格式
        if api_token.startswith("r8_"):
            rprint("[green]✓ Replicate API token format looks correct[/green]")
            return True
        else:
            rprint("[yellow]⚠️ Replicate API token format may be incorrect[/yellow]")
            return False
            
    except Exception as e:
        rprint(f"[red]❌ Replicate connection test failed: {str(e)}[/red]")
        return False


if __name__ == "__main__":
    # 测试连接
    if test_replicate_connection():
        rprint("[green]✓ Replicate backend ready[/green]")
    else:
        rprint("[red]❌ Replicate backend not available[/red]")
        
    # 如果有音频文件可以测试
    if os.path.exists(_RAW_AUDIO_FILE):
        try:
            result = transcribe_audio_replicate(_RAW_AUDIO_FILE, _RAW_AUDIO_FILE)
            rprint("[green]✓ Test transcription successful[/green]")
            rprint(f"Found {len(result.get('segments', []))} segments")
        except Exception as e:
            rprint(f"[red]❌ Test transcription failed: {str(e)}[/red]")
