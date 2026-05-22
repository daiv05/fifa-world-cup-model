# Setup

## Crear y activar entorno virtual

Asegurarse de tener instalado Python 3.13 (NO soporta versiones superiores). Luego, crear un entorno virtual para aislar las dependencias del proyecto.

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

## Instalar dependencias
```bash
pip install -r repository/requirements.txt
```
