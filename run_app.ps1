$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\Users\thang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$streamlit = "$root\venv\Lib\site-packages\streamlit\web\cli.py"
$env:PYTHONPATH = "$root\venv\Lib\site-packages;$root"

& $python $streamlit run "$root\app.py" @args
