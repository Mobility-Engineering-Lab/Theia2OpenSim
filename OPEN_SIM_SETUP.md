# OpenSim setup for Windows

This project expects two things to be available on your machine:

1. A working Conda environment for the Python workflow
2. The OpenSim command-line executable `opensim-cmd.exe`

## 1) Create and activate the Conda environment

From a PowerShell terminal:

```powershell
conda activate base
conda env create -f environment.yml
conda activate Theia2OpenSim
```

If the environment already exists:

```powershell
conda activate Theia2OpenSim
```

## 2) Install OpenSim

Install OpenSim on your machine (for example under `D:\OpenSim 4.6`).

The project needs the command-line tool, not just the GUI executable:

```text
D:\OpenSim 4.6\bin\opensim-cmd.exe
```

Add the OpenSim `bin` folder to your user PATH:

```powershell
$bin = 'D:\OpenSim 4.6\bin'
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not ($userPath -split ';' | Where-Object { $_ -eq $bin })) {
    [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $bin), 'User')
}
```

Then close and reopen PowerShell or VS Code.

Check that it works:

```powershell
opensim-cmd --help
```

## 3) Run the workflow

From the project root:

```powershell
conda activate Theia2OpenSim
python src/Python/run_pipeline.py
```

If needed, you can specify the OpenSim CLI path explicitly:

```powershell
python src/Python/run_pipeline.py --opensim-cmd "D:\OpenSim 4.6\bin\opensim-cmd.exe"
```

## Important note

Do not point the workflow at `OpenSim64.exe`.
The script expects the command-line tool `opensim-cmd.exe`.
