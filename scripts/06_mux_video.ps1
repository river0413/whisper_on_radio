#Requires -Version 5.1
<#
.SYNOPSIS
    把音频、SRT 字幕和封面图合成为一个视频。

.DESCRIPTION
    默认输出 MP4：封面图缩放到固定尺寸并居中，SRT 烧进画面（硬字幕），
    字幕可直接在播放器里看到，不需要额外字幕文件。
    使用 -Soft 则输出 MKV，把字幕作为可开关的字幕轨道。

    -Image 省略时用纯色背景；-Subtitle 省略时自动在音频同目录（以及 work\）
    查找 <文件名>.zh-ja.srt / .zh.srt / .ja.srt。

.PARAMETER Audio
    音频文件（wav/mp3/aac/m4a 等）。

.PARAMETER Subtitle
    SRT 字幕。省略时自动查找；用 -NoSubtitle 可强制不合成字幕。

.PARAMETER Image
    封面图。省略时自动查找 cover.jpg / cover.png / 封面.jpg 等，找不到则用纯色背景。

.PARAMETER Out
    输出文件。省略时输出到音频所在目录，文件名为 <音频名>.mp4（-Soft 时为 .mkv）。

.PARAMETER OutlineColour
    字幕描边颜色，六位 RGB（如 6677cc），默认 000000 黑色。

.PARAMETER Fit
    裁剪填满画面而不是加黑边。默认等比缩放 + 补边，保证图片不变形。

.PARAMETER SrtShift
    字幕整体平移秒数，正数延后、负数提前，例如 -SrtShift -0.5。

.PARAMETER PreviewSeconds
    只编码开头若干秒，用于快速确认字体、字号和排版。

.EXAMPLE
    .\scripts\06_mux_video.ps1 -Audio .\work\radio.wav -Image .\cover.jpg

.EXAMPLE
    .\scripts\06_mux_video.ps1 -Audio .\work\radio.wav -Soft -Out .\work\radio.mkv

.EXAMPLE
    .\scripts\06_mux_video.ps1 -Audio .\work\radio.wav -Image .\cover.jpg -PreviewSeconds 30
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Audio,
    [string]$Subtitle,
    [string]$Image,
    [string]$Out,
    [string]$Size = '1920x1080',
    [switch]$Fit,
    [int]$Fps = 10,
    [int]$Crf = 20,
    [string]$Preset = 'veryfast',
    [string]$AudioBitrate = '192k',
    [string]$Background = '0x101010',
    [string]$FontName = 'Microsoft YaHei',
    [int]$FontSize = 18,
    [string]$OutlineColour = '000000',
    [int]$MarginV = 36,
    [double]$SrtShift = 0,
    [double]$PreviewSeconds = 0,
    [switch]$Soft,
    [switch]$NoSubtitle,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

trap {
    Write-Host ("错误: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

function Resolve-InputFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "找不到${Label}：$Path" }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}

function Find-FirstFile {
    param([string[]]$Dirs, [string[]]$Names)
    foreach ($dir in $Dirs) {
        if (-not $dir -or -not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
        foreach ($name in $Names) {
            $candidate = Join-Path $dir $name
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $candidate).ProviderPath
            }
        }
    }
    return $null
}

function ConvertTo-AssColour([string]$Rgb) {
    # RGB(6677cc) -> ASS 的 &HAABBGGRR（BGR 顺序， alpha=00 表示不透明）
    $hex = $Rgb.TrimStart('#')
    if ($hex -notmatch '^[0-9a-fA-F]{6}$') { throw "-OutlineColour 需要六位 RGB 颜色（如 6677cc），收到：$Rgb" }
    return ('&H00{0}{1}{2}' -f $hex.Substring(4, 2), $hex.Substring(2, 2), $hex.Substring(0, 2)).ToUpper()
}

function Format-SrtTime([TimeSpan]$Time) {
    if ($Time.Ticks -lt 0) { $Time = [TimeSpan]::Zero }
    return ('{0:00}:{1:00}:{2:00},{3:000}' -f [int][math]::Floor($Time.TotalHours), $Time.Minutes, $Time.Seconds, $Time.Milliseconds)
}

function ConvertTo-Ticks([string]$Hours, [string]$Minutes, [string]$Seconds, [string]$Millis) {
    $total = ([int64]$Hours * 3600 + [int64]$Minutes * 60 + [int64]$Seconds) * 1000 + [int64]$Millis
    return $total * 10000
}

function New-ShiftedSrt {
    param([string]$Path, [double]$Seconds, [string]$Destination)
    # 同时兼容 SRT 常见的逗号和小数点毫秒分隔符
    $pattern = '^(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})(.*)$'
    $shift = [TimeSpan]::FromSeconds($Seconds)
    $lines = New-Object 'System.Collections.Generic.List[string]'
    foreach ($line in (Get-Content -LiteralPath $Path -Encoding UTF8)) {
        $m = [regex]::Match($line, $pattern)
        if (-not $m.Success) { $lines.Add($line); continue }
        $g = $m.Groups
        $start = [TimeSpan]::FromTicks((ConvertTo-Ticks $g[1].Value $g[2].Value $g[3].Value $g[4].Value)).Add($shift)
        $end = [TimeSpan]::FromTicks((ConvertTo-Ticks $g[5].Value $g[6].Value $g[7].Value $g[8].Value)).Add($shift)
        $lines.Add(('{0} --> {1}{2}' -f (Format-SrtTime $start), (Format-SrtTime $end), $g[9].Value))
    }
    [IO.File]::WriteAllLines($Destination, $lines, (New-Object System.Text.UTF8Encoding($false)))
    return $Destination
}

foreach ($tool in @('ffmpeg', 'ffprobe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "PATH 中找不到 $tool，请先安装 FFmpeg 并确认命令可用。"
    }
}

if ($Size -notmatch '^(\d+)x(\d+)$') { throw "-Size 需要形如 1920x1080，收到：$Size" }
$width = [int]$Matches[1]
$height = [int]$Matches[2]
if ($width -le 0 -or $height -le 0) { throw "-Size 的宽高必须为正数：$Size" }
if ($Fps -le 0) { throw "-Fps 必须为正数。" }

$audioPath = Resolve-InputFile $Audio '音频文件'
$audioDir = Split-Path -Parent $audioPath
$baseName = [IO.Path]::GetFileNameWithoutExtension($audioPath)

$subPath = $null
if (-not $NoSubtitle) {
    if ($Subtitle) {
        $subPath = Resolve-InputFile $Subtitle '字幕文件'
    }
    else {
        $subPath = Find-FirstFile -Dirs @($audioDir, (Join-Path $root 'work')) -Names @(
            "$baseName.zh-ja.srt", "$baseName.zh.srt", "$baseName.ja.srt"
        )
        if (-not $subPath) {
            Write-Warning "未找到 $baseName 的字幕（.zh-ja.srt / .zh.srt / .ja.srt），将输出无字幕视频。"
        }
    }
}

$imagePath = $null
if ($Image) {
    $imagePath = Resolve-InputFile $Image '封面图'
}
else {
    $imagePath = Find-FirstFile -Dirs @($audioDir, $root) -Names @(
        'cover.jpg', 'cover.jpeg', 'cover.png', 'cover.webp',
        '封面.jpg', '封面.jpeg', '封面.png'
    )
    if (-not $imagePath) { Write-Warning '未找到封面图，使用纯色背景。可用 -Image 指定图片。' }
}

$extension = if ($Soft) { '.mkv' } else { '.mp4' }
if ($Out) {
    $outPath = $Out
    if (-not [IO.Path]::IsPathRooted($outPath)) { $outPath = [IO.Path]::Combine((Get-Location).Path, $outPath) }
    if (Test-Path -LiteralPath $outPath -PathType Container) {
        $outPath = Join-Path $outPath ($baseName + $extension)
    }
    $outPath = [IO.Path]::GetFullPath($outPath)
    if (-not [IO.Path]::GetExtension($outPath)) { $outPath += $extension }
}
else {
    $outPath = Join-Path $audioDir ($baseName + $extension)
}
$outDir = Split-Path -Parent $outPath
if (-not (Test-Path -LiteralPath $outDir -PathType Container)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}
if ((Test-Path -LiteralPath $outPath) -and -not $Force) {
    throw "输出文件已存在：$outPath（加 -Force 覆盖）"
}

$duration = 0.0
try { $probe = & ffprobe -v error -show_entries format=duration -of 'default=noprint_wrappers=1:nokey=1' $audioPath 2>$null }
catch { $probe = $null }
if ($LASTEXITCODE -eq 0 -and $probe) {
    $parsed = 0.0
    if ([double]::TryParse(([string]($probe | Select-Object -First 1)).Trim(), [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
        $duration = $parsed
    }
}

# 静帧循环配上 -shortest 仍可能多出十几帧，这里用音频时长精确截断
$targetDuration = 0.0
if ($duration -gt 0) {
    $targetDuration = if ($PreviewSeconds -gt 0) { [math]::Min($PreviewSeconds, $duration) } else { $duration }
}
elseif ($PreviewSeconds -gt 0) {
    $targetDuration = $PreviewSeconds
}

$outExt = [IO.Path]::GetExtension($outPath).ToLowerInvariant()
$tempSub = $null
$tempOut = Join-Path $outDir ('~' + [IO.Path]::GetFileNameWithoutExtension($outPath) + '.tmp' + $outExt)

Write-Step '输入'
Write-Host "  音频: $audioPath"
Write-Host "  字幕: $(if ($subPath) { $subPath } else { '（无）' })"
Write-Host "  图片: $(if ($imagePath) { $imagePath } else { "纯色 $Background" })"
Write-Host "  输出: $outPath"

$ffmpegArgs = @('-hide_banner', '-nostdin', '-y', '-loglevel', 'warning', '-nostats', '-progress', 'pipe:1')

if ($imagePath) {
    $ffmpegArgs += @('-loop', '1', '-framerate', "$Fps", '-i', $imagePath)
}
else {
    $ffmpegArgs += @('-f', 'lavfi', '-i', ('color=c={0}:s={1}x{2}:r={3}' -f $Background, $width, $height, $Fps))
}
$ffmpegArgs += @('-i', $audioPath)

$mapArgs = @('-map', '0:v:0', '-map', '1:a:0')

Push-Location $outDir
try {
    if ($subPath -and $SrtShift -ne 0) {
        Write-Step "平移字幕 $SrtShift 秒"
        $tempSub = Join-Path $outDir ('~' + $baseName + '.shift.srt')
        New-ShiftedSrt -Path $subPath -Seconds $SrtShift -Destination $tempSub | Out-Null
        $subPath = $tempSub
    }

    $videoFilter = if ($Fit) {
        'scale={0}:{1}:force_original_aspect_ratio=increase:flags=lanczos,crop={0}:{1}' -f $width, $height
    }
    else {
        'scale={0}:{1}:force_original_aspect_ratio=decrease:flags=lanczos,pad={0}:{1}:(ow-iw)/2:(oh-ih)/2:color={2}' -f $width, $height, $Background
    }

    if ($subPath) {
        # 用绝对路径，避免 ffmpeg 的工作目录与字幕目录不一致时找不到文件；
        # Windows 盘符冒号和反斜杠按 ffmpeg 滤镜语法转义。
        $leaf = Split-Path -Leaf $subPath
        if ($leaf.Contains("'")) { throw "字幕文件名包含单引号，FFmpeg 滤镜无法处理：$leaf" }
        $style = 'FontName={0},FontSize={1},PrimaryColour=&H00FFFFFF,OutlineColour={2},BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV={3}' -f $FontName, $FontSize, (ConvertTo-AssColour $OutlineColour), $MarginV
        $escapedSub = $subPath.Replace('\', '/').Replace(':', '\:')
        $subtitleFilter = "subtitles=filename='$escapedSub':force_style='$style'"

        if ($Soft) {
            $ffmpegArgs += @('-i', $subPath)
            $mapArgs += @('-map', '2:s:0')
        }
        else {
            $videoFilter += ',' + $subtitleFilter
        }
    }

    $ffmpegArgs += @('-vf', $videoFilter)
    $ffmpegArgs += $mapArgs
    $ffmpegArgs += @('-c:v', 'libx264', '-preset', $Preset, '-crf', "$Crf", '-pix_fmt', 'yuv420p', '-r', "$Fps")
    $ffmpegArgs += @('-c:a', 'aac', '-b:a', $AudioBitrate)

    if ($subPath -and $Soft) {
        $subtitleCodec = if ($outExt -eq '.mp4') { 'mov_text' } else { 'srt' }
        $ffmpegArgs += @('-c:s', $subtitleCodec, '-metadata:s:s:0', 'language=chi', '-metadata:s:s:0', 'title=中日双语', '-disposition:s:0', 'default')
    }

    if ($outExt -eq '.mp4') { $ffmpegArgs += @('-movflags', '+faststart') }
    $ffmpegArgs += @('-shortest')

    if ($targetDuration -gt 0) {
        $ffmpegArgs += @('-t', ('{0:0.###}' -f $targetDuration))
    }

    $ffmpegArgs += $tempOut

    Write-Step '开始编码'
    # ffmpeg 的日志走 stderr，默认会被 PowerShell 当错误记录染红并中断脚本；
    # 这里合并两路输出，只把日志留到失败时打印，同时用 -progress 显示百分比。
    $messages = New-Object 'System.Collections.Generic.List[string]'
    $lastPercent = -1
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & ffmpeg @ffmpegArgs 2>&1 | ForEach-Object {
            $text = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { [string]$_ }
            if ($targetDuration -gt 0 -and $text -match '^out_time_us=(\d+)$') {
                $percent = [int][math]::Floor(([double]$Matches[1] / 1000000) / $targetDuration * 100)
                if ($percent -ge 0 -and $percent -le 100 -and $percent -ne $lastPercent) {
                    $lastPercent = $percent
                    Write-Host ("`r  进度 {0,3}%" -f $percent) -NoNewline
                }
            }
            elseif ($text.Trim()) {
                $messages.Add($text.Trim())
            }
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
        $stopwatch.Stop()
        if ($lastPercent -ge 0) { Write-Host '' }
    }

    if ($exitCode -ne 0) {
        if ($messages.Count -gt 0) { $messages | Select-Object -Last 40 | ForEach-Object { Write-Host "  $_" } }
        throw "ffmpeg 退出码 $exitCode"
    }

    Move-Item -LiteralPath $tempOut -Destination $outPath -Force
    Write-Host ('  用时 {0:0.0} 秒' -f $stopwatch.Elapsed.TotalSeconds)
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $tempOut) { Remove-Item -LiteralPath $tempOut -Force -ErrorAction SilentlyContinue }
    if ($tempSub -and (Test-Path -LiteralPath $tempSub)) { Remove-Item -LiteralPath $tempSub -Force -ErrorAction SilentlyContinue }
}

$sizeMb = (Get-Item -LiteralPath $outPath).Length / 1MB
Write-Step ('完成：{0}（{1:0.0} MB）' -f $outPath, $sizeMb)
Write-Output $outPath
