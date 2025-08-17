
Pasos rápidos (Cmder + VS Code):

1) Activar el entorno virtual y ubicarse en la carpeta del proyecto:
   cd /d D:\ICONTINENTAL\ML_IDL02
   venv_ML_IDL02\Scripts\activate

2) Colocar el CSV en: proyecto_idl02/data/historial_compras.csv
   (ya se guardó automáticamente si subiste 'Copia de historial_compras.csv')

3) Ejecutar el pipeline:
   python src/idl02_pipeline.py --data data/historial_compras.csv --kmin 2 --kmax 8 --eps 0.5 --min_samples 5

4) Revisar salidas:
   - figures/*.png
   - outputs/segmented_*.csv
   - outputs/summary.txt

5) Abrir VS Code:
   code .
