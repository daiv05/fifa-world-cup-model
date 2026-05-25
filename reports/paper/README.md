# Reporte técnico IEEE

Documento `paper.tex` en clase `IEEEtran` (conferencia, dos columnas).

## Compilación

Requiere una distribución LaTeX con la clase `IEEEtran` (incluida en TeX Live full y MiKTeX). Desde este directorio:

```bash
pdflatex paper.tex
```

La bibliografía está **embebida** como `thebibliography` dentro de `paper.tex`, por lo que **no se requiere ejecutar `bibtex`/`biber`**. El archivo `references.bib` se conserva solo como referencia documental para futuras citas.

Salida: `paper.pdf`.

## Contenido

- `paper.tex` - fuente principal.
- `references.bib` - bibliografía (Elo, Dixon-Coles, XGBoost, LightGBM, SHAP, Brier, Clopper-Pearson, FIFA 2026, fuentes de datos).
- `figs/` - figuras EDA/SHAP/calibración copiadas de `reports/figures/`.

Los números de las tablas (métricas, top-10, sensibilidad, progresión) provienen literalmente de `data/processed/*.csv` y deben regenerarse junto con esos CSV.
