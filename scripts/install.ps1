Write-Host 'Checking python version...'
$Ver = python -c "import sys; ver=sys.version_info[:3];
if ver[0]==3 and ver[1]==8 and ver[2] >= 5:
    print('OK')
else:
    print('Python version not sufficient, 3.8 is needed with patch 5 at the very least.')"

Write-Host $Ver
if ($Ver = "OK") {
    pip install --upgrade pip
    pip install virtualenv
    python -m virtualenv .\..\env
    .\..\env\Scripts\Activate.ps1
    pip install -r .\.\python_requirements.txt
    deactivate
}
