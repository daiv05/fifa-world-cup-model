# RUN.md — Guía de Ejecución del Proyecto

> Previamente se tiene que haber creado un entorno virtual e instalado las dependencias necesarias. Para esto, sigue las instrucciones en `SETUP.md`.

Instrucciones para ejecutar cada parte del pipeline, desde la obtención de datasets hasta la simulación del Mundial.

1. Asegúrate de haber obtenido todos los datasets siguiendo la guía en `GET-DATASETS.md`.

2. Ejecuta cada módulo en el siguiente orden:
   - Crea las características necesarias para el modelo:

   ```
   python -m repository.src.features.features
   ```

   - Entrena el modelo con los datos procesados:

   ```
   python -m repository.src.models.train
   ```

   - Simula el Mundial utilizando el modelo entrenado:

   ```
   python -m repository.src.simulation.simulate
   ```

   Puedes ajustar el número de iteraciones para obtener resultados más precisos:

   ```
    python -m repository.src.simulation.simulate --iterations 10000
    ```

    Asi como especificar el modelo a utilizar (tienes que especificar uno de los modelos de la carpeta `repository/data/processed/models/`):

    ```
    python -m repository.src.simulation.simulate --model xgboost
    ```

3. Para visualizar los resultados de la simulación, puedes ejecutar el siguiente comando:

```
streamlit run repository/src/visualization/dashboard.py
```

4. Para generar el reporte de la simulación, ejecuta:

```
pytest repository/tests/ -v
```

5. Para ejecutar el checkpoint de Great Expectations y validar los datos, utiliza el siguiente comando:

```
great_expectations checkpoint run wc_data_checkpoint
```