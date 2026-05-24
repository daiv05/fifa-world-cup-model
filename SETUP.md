# Setup

## Crear y activar entorno virtual

Asegurarse de tener instalado Python 3.13 (NO soporta versiones superiores). Luego, crear un entorno virtual para aislar las dependencias del proyecto.

### Windows (PowerShell)
```powershell
python -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
python -3.13 -m venv .venv
source .venv/bin/activate
```

## Instalar dependencias
```bash
pip3.13 install -r repository/requirements.txt
```

## Problemas conocidos

Puede lanzar un error de pip por intentar actualizar pip dentro del entorno virtual. Si esto ocurre, ejecutar:
```bash
.venv\Scripts\python.exe -m pip install -r repository/requirements.txt
```