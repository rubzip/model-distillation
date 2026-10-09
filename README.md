# Acoustic Scene Clasification Model Distillation
This repo contains a framework for implementing my Masters thesis work (it doesnt contain data nor specific architectures).

## Comandos Útiles (`uv`)

A continuación se detallan los comandos principales para trabajar con este proyecto utilizando `uv`:

### 1. Generar Datos Sintéticos (Mock Data)
Genera archivos `.wav` de prueba y su respectiva metadata en `mock_data/raw/`:
```bash
uv run python scripts/generate_mock_data.py
```

### 2. Procesar el Dataset
Ejecuta el pipeline completo de procesamiento (extracción de segmentos, generación de features MEL y reducción de dimensionalidad con PCA):
```bash
uv run python scripts/process_data.py
```

### 3. Ejecutar los Tests
Para correr la batería de pruebas unitarias y de validación:
```bash
uv run pytest tests/
```
