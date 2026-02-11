"""
Pipeline de procesamiento: Airbnb NYC 2019.
Carga datos crudos, limpia, divide en train/test y guarda en data/processed.
"""
import pandas as pd
import requests
from pathlib import Path


def train_test_split(df, test_size=0.2, random_state=42):
    """División train/test (solo pandas)."""
    shuffled = df.sample(frac=1, random_state=random_state)
    n = len(shuffled)
    test_n = int(n * test_size)
    return shuffled.iloc[:-test_n], shuffled.iloc[-test_n:]


def get_paths():
    """Rutas del proyecto (raíz, raw, processed)."""
    root = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return root, raw_dir, processed_dir


def download_raw(url: str, path: Path) -> None:
    """Descarga el CSV si no existe en path."""
    if path.exists():
        print(f"El archivo ya existe en {path}")
        return
    response = requests.get(url)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    print(f"Descargado y guardado en {path}")


def load_raw(path: Path) -> pd.DataFrame:
    """Carga el dataset desde data/raw."""
    df = pd.read_csv(path)
    print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas no relevantes y rellena NA en reviews_per_month."""
    drop_cols = ["id", "name", "host_name", "last_review"]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if "reviews_per_month" in df_clean.columns:
        df_clean["reviews_per_month"] = df_clean["reviews_per_month"].fillna(0)
    return df_clean


def main():
    url_ds = "https://breathecode.herokuapp.com/asset/internal-link?id=927&path=AB_NYC_2019.csv"
    _, raw_dir, processed_dir = get_paths()
    raw_path = raw_dir / "AB_NYC_2019.csv"

    download_raw(url_ds, raw_path)
    df = load_raw(raw_path)
    df_clean = clean(df)

    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)} filas | Test: {len(test_df)} filas")

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Guardado: {train_path}")
    print(f"Guardado: {test_path}")


if __name__ == "__main__":
    main()
