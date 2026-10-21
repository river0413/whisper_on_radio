"""Step 1: validate a local audio file and convert it to 16kHz mono WAV."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, fail  # noqa: E402


def _duration_seconds(src):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description="本地音频 -> 16kHz 单声道 WAV")
    p.add_argument("input", help="本地音频/视频文件")
    p.add_argument("-o", "--output", help="输出 WAV 路径，默认 work/<名称>.wav")
    args = p.parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        fail(f"找不到输入文件: {src}")
    if not shutil.which("ffmpeg"):
        fail("未找到 ffmpeg，请先安装并加入 PATH")

    out = Path(args.output) if args.output else ROOT / "work" / f"{src.stem}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-progress", "pipe:1",
           "-nostats", str(out)]
    duration = _duration_seconds(src)
    errors = []
    with tqdm(total=duration, desc="音频转换", unit="s", file=sys.stderr) as progress:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        last_time = 0.0
        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time="):
                value = line.partition("=")[2]
                try:
                    h, m, s = value.split(":")
                    current = int(h) * 3600 + int(m) * 60 + float(s)
                except ValueError:
                    continue
                if current > last_time:
                    progress.update(current - last_time)
                    last_time = current
            elif line and "=" not in line:
                errors.append(line)
        return_code = process.wait()
        if return_code == 0 and duration and progress.n < duration:
            progress.update(duration - progress.n)
    if return_code != 0:
        fail(f"ffmpeg 转换失败: {' '.join(errors).strip() or f'退出码 {return_code}'}")
    print(out)


if __name__ == "__main__":
    main()
