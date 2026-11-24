param(
    [string]$ModelDirectory = (Join-Path $PSScriptRoot 'models\minilm-1110a243')
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
$modelId = 'sentence-transformers/all-MiniLM-L6-v2'
$metadata = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$modelId/revision/${revision}?blobs=true" -TimeoutSec 30
if ($metadata.sha -ne $revision) { throw 'Unexpected model revision' }
$resolvedModelDirectory = [IO.Path]::GetFullPath($ModelDirectory)
New-Item -ItemType Directory -Force -Path $resolvedModelDirectory | Out-Null
$modelFiles = @(
    'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'vocab.txt', 'sentence_bert_config.json',
    '1_Pooling/config.json', 'modules.json', 'README.md'
)
foreach ($modelFile in $modelFiles) {
    $destination = [IO.Path]::GetFullPath((Join-Path $resolvedModelDirectory $modelFile))
    if (-not $destination.StartsWith($resolvedModelDirectory + [IO.Path]::DirectorySeparatorChar)) {
        throw 'Model destination escaped requested directory'
    }
    $remote = $metadata.siblings | Where-Object { $_.rfilename -eq $modelFile }
    if ($null -eq $remote) { throw "Pinned revision does not contain $modelFile" }
    $isComplete = Test-Path -LiteralPath $destination
    if ($isComplete -and $remote.lfs.sha256) {
        $isComplete = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $remote.lfs.sha256
    }
    if (-not $isComplete) {
        New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($destination)) | Out-Null
        $temporary = $destination + '.download'
        Invoke-WebRequest -Uri "https://huggingface.co/$modelId/resolve/$revision/${modelFile}?download=true" -OutFile $temporary -TimeoutSec 120
        if ($remote.lfs.sha256 -and (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $remote.lfs.sha256) {
            throw "SHA256 mismatch for $modelFile"
        }
        Move-Item -LiteralPath $temporary -Destination $destination -Force
    }
    Write-Output "$modelFile verified ($( (Get-Item -LiteralPath $destination).Length ) bytes)"
}
