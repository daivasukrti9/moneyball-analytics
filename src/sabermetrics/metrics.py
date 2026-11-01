import numpy as np
import pandas as pd

def calculate_singles(df: pd.DataFrame) -> pd.Series:
    """Calcula sencillos (1B) = Hits - 2B - 3B - HR."""
    h = df.get('H', 0)
    doubles = df.get('2B', 0)
    triples = df.get('3B', 0)
    hr = df.get('HR', 0)
    return h - doubles - triples - hr

def calculate_obp(df: pd.DataFrame) -> pd.Series:
    """
    On-Base Percentage (OBP) = (H + BB + HBP) / (AB + BB + HBP + SF)
    Maneja valores nulos en SF (Elevados de sacrificio no registrados pre-1954)
    y evita ZeroDivisionError.
    """
    h = df.get('H', 0).fillna(0)
    bb = df.get('BB', 0).fillna(0)
    hbp = df.get('HBP', 0).fillna(0) if 'HBP' in df.columns else 0
    ab = df.get('AB', 0).fillna(0)
    sf = df.get('SF', 0).fillna(0) if 'SF' in df.columns else 0

    numerator = h + bb + hbp
    denominator = ab + bb + hbp + sf
    return np.where(denominator > 0, numerator / denominator, 0.0)

def calculate_slg(df: pd.DataFrame) -> pd.Series:
    """
    Slugging Percentage (SLG) = Bases Totales / Turnos al Bate
    Bases Totales = 1B + 2*2B + 3*3B + 4*HR
    """
    singles = df.get('1B', calculate_singles(df)).fillna(0)
    doubles = df.get('2B', 0).fillna(0)
    triples = df.get('3B', 0).fillna(0)
    hr = df.get('HR', 0).fillna(0)
    ab = df.get('AB', 0).fillna(0)

    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)
    return np.where(ab > 0, total_bases / ab, 0.0)

def calculate_ops(df: pd.DataFrame) -> pd.Series:
    """OPS = OBP + SLG"""
    obp = df.get('OBP', calculate_obp(df))
    slg = df.get('SLG', calculate_slg(df))
    return obp + slg

def calculate_runs_created(df: pd.DataFrame) -> pd.Series:
    """
    Fórmula clásica de Carreras Creadas (RC) de Bill James:
    RC = ((H + BB) * Bases Totales) / (AB + BB)
    """
    h = df.get('H', 0).fillna(0)
    bb = df.get('BB', 0).fillna(0)
    singles = df.get('1B', calculate_singles(df)).fillna(0)
    doubles = df.get('2B', 0).fillna(0)
    triples = df.get('3B', 0).fillna(0)
    hr = df.get('HR', 0).fillna(0)
    ab = df.get('AB', 0).fillna(0)

    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * hr)
    on_base = h + bb
    denominator = ab + bb

    return np.where(denominator > 0, (on_base * total_bases) / denominator, 0.0)

def calculate_pythagorean_wins(runs_scored: float | pd.Series, 
                               runs_allowed: float | pd.Series, 
                               exponent: float = 2.0, 
                               total_games: int = 162) -> float | pd.Series:
    """
    Expectativa Pitagórica de Victorias de Bill James:
    Win% = R^exp / (R^exp + RA^exp)
    Victorias esperadas = Win% * Total de Juegos
    """
    r_exp = np.power(runs_scored, exponent)
    ra_exp = np.power(runs_allowed, exponent)
    denominator = r_exp + ra_exp
    win_pct = np.where(denominator > 0, r_exp / denominator, 0.5)
    return win_pct * total_games

def calculate_cost_per_metric(salaries: pd.Series, metric: pd.Series) -> pd.Series:
    """Costo en dólares por unidad de métrica producida (ej. $/RC o $/WAR)."""
    return np.where(metric > 0, salaries / metric, np.nan)
