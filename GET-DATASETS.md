# GET-DATASETS.md — Guía de Obtención de Datasets

Instrucciones completas para obtener, renombrar y ubicar todos los datasets del proyecto.
Todos los archivos van en `repository/data/raw/`.

---

## Resumen rápido

| Archivo | Método | Acción requerida |
|---|---|---|
| `international_results.csv` | Automático | Ninguna |
| `statsbomb_matches.csv` | Automático | Ninguna |
| `statsbomb_xg_by_team.csv` | Automático | Ninguna |
| `fbref_stats.csv` | Automático (puede fallar) | Ver sección FBref |
| `squad_values.csv` | Automático (valores aprox.) | Opcional: mejorar valores |
| `wc_matches_1974_2022.csv` | Kaggle manual | Descargar + renombrar columnas |
| `fifa_ranking.csv` | Kaggle manual | Descargar |
| `wc2026_fixture.csv` | Creación manual | Copiar CSV de esta guía |

---

## Paso 1 — Crear archivo manual: `wc2026_fixture.csv`

Crea el archivo `repository/data/raw/wc2026_fixture.csv` con este contenido exacto:

```csv
group,team
A,Mexico
B,Canada
C,Brazil
D,USA
E,Germany
F,Netherlands
G,Belgium
H,Spain
I,France
J,Argentina
K,Portugal
L,England
A,South Africa
B,Bosnia & Herzegovina
C,Morocco
D,Paraguay
E,Curacao
F,Japan
G,Egypt
H,Cape Verde
I,Senegal
J,Algeria
K,DR Congo
L,Croatia
A,South Korea
B,Qatar
C,Haiti
D,Australia
E,Ivory Coast
F,Sweden
G,Iran
H,Saudi Arabia
I,Iraq
J,Austria
K,Uzbekistan
L,Ghana
A,Czech Republic
B,Switzerland
C,Scotland
D,Turkey
E,Ecuador
F,Tunisia
G,New Zealand
H,Uruguay
I,Norway
J,Jordan
K,Colombia
L,Panama
```

---

## Paso 2 — Descargas manuales de Kaggle

### 2.1 FIFA World Cup Matches 1974–2022

**URL:** https://www.kaggle.com/datasets/ibrahimshahrukh/fifa-world-cup-matches-19742022-dataset

1. Descarga el CSV del dataset
2. El archivo original tiene columnas con nombres distintos. **Renómbralas** así antes de guardarlo:

| Nombre original | Renombrar a |
|---|---|
| `Home Team Name` | `home_team` |
| `Away Team Name` | `away_team` |
| `Home Team Goals` | `home_score` |
| `Away Team Goals` | `away_score` |
| `Stage` | `tournament` (o agrega columna con valor `"FIFA World Cup"`) |

> Las demás columnas pueden mantenerse — no afectan el pipeline.

3. Guarda el resultado como: `repository/data/raw/wc_matches_1974_2022.csv`

---

### 2.2 FIFA World Ranking 1993–2023

**URL:** https://www.kaggle.com/datasets/cashncarry/fifaworldranking

1. Descarga el CSV del dataset
2. No requiere renombrado de columnas. Las columnas que usa el pipeline son:
   - `rank_date`, `country_full`, `rank`, `total_points`
3. Guarda como: `repository/data/raw/fifa_ranking.csv`

---

## Paso 3 — Ejecución automática del pipeline

Con los archivos de los pasos 1 y 2 en su lugar, ejecuta los siguientes scripts
**desde la raíz del proyecto** (`fifa-world-cup-model/`):

```bash
# Descarga international_results.csv desde GitHub (~49,000 partidos 1872-2024)
python -m repository.src.data.data_loader

# Descarga StatsBomb xG, genera squad_values.csv y obtiene stats de FBref
python -m repository.src.data.scraper
```

### Qué hace cada script automáticamente

**`data_loader.py`:**
- Descarga `international_results.csv` desde `github.com/martj42/international_results`
- Lo guarda en caché en `repository/data/raw/international_results.csv`
- En ejecuciones posteriores usa el caché (no re-descarga)

**`scraper.py`:**
- **StatsBomb:** Descarga partidos internacionales (Mundiales, Euros, Copa América) con xG
  desde `github.com/statsbomb/open-data` vía `statsbombpy`
  - genera `statsbomb_matches.csv` y `statsbomb_xg_by_team.csv`
- **Squad values:** Genera `squad_values.csv` con valores de mercado aproximados
  para los 48 equipos (ver sección de actualización manual abajo)
- **FBref:** Intenta descargar estadísticas de torneos `INT-World Cup` e
  `INT-European Championship` vía `soccerdata`
  - genera `fbref_stats.csv` si tiene éxito

---

## Comportamiento esperado al correr `scraper.py`

```
Descargando datos de StatsBomb...
  NoAuthWarning: credentials were not supplied. open data access only  ← NORMAL, ignorar
  Equipos con xG: N   (debe ser > 0 si hay conexión a internet)

Descargando valores de plantilla (Transfermarkt)...
  squad_values.csv generado con valores aproximados (47 equipos).
  Para valores exactos actualiza: ...repository/data/raw/squad_values.csv

Descargando estadísticas FBref...
  (puede mostrar advertencias de soccerdata sobre config — son informativas)
  Registros FBref: N
OK
```

Si `Equipos con xG: 0` pero no hay errores de red, borra el caché y reintenta:
```bash
del repository\data\raw\statsbomb_matches.csv
python -m repository.src.data.scraper
```

## Borrar data de FBref (si falla la descarga)

Si la descarga de FBref falla, es posible que queden archivos corruptos en el caché de `soccerdata` que causen errores persistentes. Para limpiar completamente el caché de FBref, borra la carpeta local de datos de `soccerdata`:

```bash
rmdir /s /q "C:\Users\Usuario\soccerdata\data\FBref"

python -m repository.src.data.scraper
```

---

## Actualización manual de Squad Values (opcional pero recomendado)

Los valores hardcodeados en el código son **aproximados** (mayo 2026).
Para valores exactos:

1. Ve a: https://www.transfermarkt.com/statistik/weltrangliste/statistik
2. Filtra por los 48 equipos clasificados al Mundial 2026
3. Edita `repository/data/raw/squad_values.csv` con el formato:

```csv
team,squad_value_eur
France,1150000000
England,1300000000
Brazil,1050000000
...
```

> El pipeline carga este CSV automáticamente en la siguiente ejecución.
> Una vez que existe el archivo CSV, los valores hardcodeados en el código son ignorados.

---

## Errores conocidos (ya corregidos en el código)

Los siguientes problemas fueron encontrados y corregidos durante el desarrollo.
Se documentan aquí como referencia:

### `parents[3]` - `parents[2]` (path bug)
- **Síntoma:** Los archivos se guardaban en `fifa-world-cup-model/data/raw/`
  en vez de `repository/data/raw/`
- **Causa:** Error de conteo en `Path(__file__).parents[N]`
- **Corrección aplicada en:** `data_loader.py`, `scraper.py`, `features.py`,
  `train.py`, `simulate.py`, `dashboard.py`
- **Si ves archivos en la raíz del proyecto:** Bórralos y vuelve a ejecutar:
  ```bash
  rm -rf fifa-world-cup-model/data/
  ```

### `soccerdata.Transfermarkt` no existe
- **Síntoma:** `module 'soccerdata' has no attribute 'Transfermarkt'`
- **Causa:** La librería `soccerdata` no implementa Transfermarkt
- **Corrección:** Reemplazado por valores hardcodeados + CSV manual sobreescribible

### FBref league `"Internationals"` inválido
- **Síntoma:** `Invalid league 'Internationals'. Valid leagues are: [...]`
- **Causa:** El string de liga no coincide con los identificadores de FBref
- **Corrección:** Cambiado a `["INT-World Cup", "INT-European Championship"]`

### FBref season `'2023'` inválido (KeyError)
- **Síntoma:** `soccerdata/FBref no disponible: '2023'`
- **Causa:** No existe el season `2023` para ligas internacionales. El Mundial fue
  en 2022 y la Euro en 2024. FBref solo tiene datos de años en que se jugó el torneo.
- **Corrección:** La función `get_fbref_stats()` ahora itera sobre pares válidos
  `(liga, season)`: `INT-World Cup 2022`, `INT-World Cup 2018`,
  `INT-European Championship 2024`, `INT-European Championship 2020`.
  Cada par se intenta por separado; si falla, se omite y continúa con el siguiente.

### FBref `stat_type='possession'` inválido para ligas internacionales
- **Síntoma:** `FBref skip: INT-World Cup 2022 — Invalid argument: stat_type should be in ['standard', 'keeper', 'shooting', 'playing_time', 'misc']`
- **Causa:** `'possession'` no existe como stat_type para torneos internacionales (`INT-*`),
  solo para ligas de clubes. Como ambas llamadas (`shooting` y `possession`) estaban en el
  mismo bloque `try`, el fallo de `possession` cancelaba también los datos de `shooting`.
- **Corrección:** La función ahora itera sobre `['standard', 'shooting', 'misc']` de forma
  independiente para cada par `(liga, season)`. Si un stat_type falla se omite y continúa
  con el siguiente.

### StatsBomb columnas `home_team_name` vs `home_team`
- **Síntoma:** `Equipos con xG: 0` sin errores visibles
- **Causa:** statsbombpy puede devolver `home_team` o `home_team_name`
  según la versión instalada
- **Corrección:** El código ahora detecta automáticamente cuál columna existe

---

## Estructura final esperada de `repository/data/raw/`

```
repository/data/raw/
├── international_results.csv     ← AUTO (~49k partidos)
├── statsbomb_matches.csv         ← AUTO (partidos con xG)
├── statsbomb_xg_by_team.csv      ← AUTO (xG agregado por equipo)
├── fbref_stats.csv               ← AUTO si soccerdata funciona
├── squad_values.csv              ← AUTO (valores aprox.) o MANUAL (valores exactos)
├── wc_matches_1974_2022.csv      ← MANUAL Kaggle (columnas renombradas)
├── fifa_ranking.csv              ← MANUAL Kaggle
└── wc2026_fixture.csv            ← MANUAL (copiar CSV de esta guía)
```
