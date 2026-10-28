from pathlib import Path

import pandas as pd

ARCHIVO_ENTRADA = Path("poker-hand-training.csv")
CARPETA_SALIDA = Path("datasets")
COLUMNA_OBJETIVO = "Poker Hand"

def main():
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró {ARCHIVO_ENTRADA}"
        )

    df = pd.read_csv(ARCHIVO_ENTRADA)

    if COLUMNA_OBJETIVO not in df.columns:
        raise ValueError(
            f"No se encontró la columna {COLUMNA_OBJETIVO}. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    for porcentaje in (25, 50, 100):
        fraccion = porcentaje / 100
        grupos = []

        # Mantener la representación de cada clase de póker.
        for _, grupo in df.groupby(COLUMNA_OBJETIVO, sort=False):
            cantidad = (
                len(grupo)
                if porcentaje == 100
                else max(1, round(len(grupo) * fraccion))
            )

            grupos.append(
                grupo.sample(n=cantidad, random_state=42)
            )

        reducido = (
            pd.concat(grupos)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        salida = CARPETA_SALIDA / (
            f"poker-hand-training-{porcentaje}.csv"
        )

        reducido.to_csv(salida, index=False)

        print(
            f"{porcentaje}% -> {len(reducido)} registros: {salida}"
        )

if __name__ == "__main__":
    main()
