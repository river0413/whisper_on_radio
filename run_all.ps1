param(
    [Parameter(Mandatory = $true)][string]$Audio,
    [string]$PromptFile,
    [string]$TermsFile,
    [string]$ReferenceFile,
    [string]$ProofreadThinking
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# 让 Python 的中文进度与错误信息在本控制台正确显示
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = 'utf-8' }

# Python 侧会自动读 .env；PowerShell 侧也读同样的值，避免默认值覆盖用户配置
function Get-DotEnvValue([string]$Name) {
    $envFile = Join-Path $root '.env'
    if (-not (Test-Path -LiteralPath $envFile)) { return $null }
    foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

# 推理默认开启（推理 token 计入 PROOFREAD_MAX_TOKENS，默认 384k）；
# 可用 -ProofreadThinking disabled 或环境变量 / .env 的 PROOFREAD_THINKING 关闭。
if (-not $ProofreadThinking) {
    $resolvedThinking = $env:PROOFREAD_THINKING
    if (-not $resolvedThinking) { $resolvedThinking = Get-DotEnvValue 'PROOFREAD_THINKING' }
    if (-not $resolvedThinking) { $resolvedThinking = 'auto' }
    $ProofreadThinking = $resolvedThinking
}
$proofreadLines = Get-DotEnvValue 'PROOFREAD_LINES'
if (-not $env:PROOFREAD_LINES -and $proofreadLines) { $env:PROOFREAD_LINES = $proofreadLines }
$proofreadMaxTokens = Get-DotEnvValue 'PROOFREAD_MAX_TOKENS'
if ($proofreadMaxTokens) { $env:PROOFREAD_MAX_TOKENS = $proofreadMaxTokens }
Write-Host ("校对设置: 推理={0}，分批={1} 条，输出上限={2} tokens" -f `
    $ProofreadThinking, $(if ($env:PROOFREAD_LINES) { $env:PROOFREAD_LINES } else { '默认' }), `
    $(if ($env:PROOFREAD_MAX_TOKENS) { $env:PROOFREAD_MAX_TOKENS } else { '默认' }))
if ($PromptFile -and -not (Test-Path $PromptFile)) {
    Write-Error "找不到辅助文档: $PromptFile"
    exit 1
}
if ($TermsFile -and -not (Test-Path $TermsFile)) {
    Write-Error "找不到术语表文件: $TermsFile"
    exit 1
}
if ($ReferenceFile -and -not (Test-Path $ReferenceFile)) {
    Write-Error "找不到 Hy-MT 参考文档: $ReferenceFile"
    exit 1
}
$wav = python "$root\scripts\01_get_audio.py" $Audio
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$ja = python "$root\scripts\02_transcribe_srt.py" $wav
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$translationArgs = @($ja)
if ($TermsFile) {
    $translationArgs += @("--terms-file", $TermsFile)
}
if ($ReferenceFile) {
    $translationArgs += @("--reference-file", $ReferenceFile)
}
$zh = python "$root\scripts\03_translate_srt.py" @translationArgs
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$proofreadArgs = @($ja, "--zh-srt", $zh, "--proofread-thinking", $ProofreadThinking)
if ($ReferenceFile) {
    $proofreadArgs += @("--reference-file", $ReferenceFile)
}
if ($PromptFile) {
    $proofreadArgs += @("--proofread-prompt-file", $PromptFile)
}
if ($TermsFile) {
    $proofreadArgs += @("--terms-file", $TermsFile)
}
python "$root\scripts\04_proofread_srt.py" @proofreadArgs
exit $LASTEXITCODE
