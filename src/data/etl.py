import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import sqlite3
import urllib.request
import pandas as pd
import numpy as np
from src.sabermetrics.metrics import (
    calculate_singles,
    calculate_obp,
    calculate_slg,
    calculate_ops,
    calculate_runs_created,
    calculate_cost_per_metric,
    calculate_pythagorean_wins
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DB_PATH = os.path.join(DATA_DIR, "moneyball.db")

TABLE_URLS = {
    "Batting": "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/Batting.csv",
    "People": "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/People.csv",
    "Teams": "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/Teams.csv",
    "Salaries": "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/contrib/Salaries.csv",
    "Appearances": "https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/Appearances.csv",
}

def ensure_directories():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def fetch_table(table_name: str) -> pd.DataFrame:
    """
    Descarga una tabla de la base de datos Lahman directamente desde el repositorio oficial
    de BaseballDataBank para máxima velocidad y fiabilidad sin prompts interactivos.
    """
    ensure_directories()
    local_raw_path = os.path.join(DATA_DIR, f"{table_name}.csv")
    
    if os.path.exists(local_raw_path):
        return pd.read_csv(local_raw_path, low_memory=False)

    url = TABLE_URLS.get(table_name, f"https://raw.githubusercontent.com/cbwinslow/baseballdatabank/master/core/{table_name}.csv")
    try:
        print(f"Descargando {table_name} desde {url}...", flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=45) as resp, open(local_raw_path, 'wb') as out_f:
            out_f.write(resp.read())
        
        df = pd.read_csv(local_raw_path, low_memory=False)
        print(f"Tabla {table_name} descargada exitosamente ({len(df)} registros).", flush=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Error crítico al obtener la tabla {table_name}: {e}")

def process_batting_and_salaries(min_year: int = 1960, max_year: int = 2023) -> pd.DataFrame:
    """
    Procesa, limpia y enriquece las estadísticas de bateo y salarios.
    Resuelve los bugs identificados en Moneyball.md:
    1. Imputación segura de SF para años previos a 1954.
    2. Agregación de jugadores con múltiples equipos en un mismo año (traspasos).
    3. Manejo de ZeroDivisionError en métricas de ratio.
    """
    batting = fetch_table("Batting")
    salaries = fetch_table("Salaries")
    people = fetch_table("People")

    # Filtrar rango de años
    batting = batting[(batting['yearID'] >= min_year) & (batting['yearID'] <= max_year)].copy()
    salaries = salaries[(salaries['yearID'] >= min_year) & (salaries['yearID'] <= max_year)].copy()

    # Sanitización de columnas
    numeric_cols = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
    for col in numeric_cols:
        if col in batting.columns:
            batting[col] = pd.to_numeric(batting[col], errors='coerce').fillna(0)
        else:
            batting[col] = 0

    # Traspasos (Trades): Agrupar por jugador y año para tener totales de temporada
    agg_dict = {col: 'sum' for col in numeric_cols}
    agg_dict['teamID'] = lambda x: '/'.join(x.dropna().unique())
    agg_dict['lgID'] = 'last'

    batting_agg = batting.groupby(['playerID', 'yearID'], as_index=False).agg(agg_dict)

    # Calcular métricas sabermétricas
    batting_agg['1B'] = calculate_singles(batting_agg)
    batting_agg['OBP'] = calculate_obp(batting_agg)
    batting_agg['SLG'] = calculate_slg(batting_agg)
    batting_agg['OPS'] = calculate_ops(batting_agg)
    batting_agg['Runs_Created'] = calculate_runs_created(batting_agg)

    # Unir con información biográfica de People
    people_cols = ['playerID', 'nameFirst', 'nameLast', 'bats', 'throws', 'birthYear']
    people_clean = people[people_cols].copy()
    people_clean['fullName'] = people_clean['nameFirst'].fillna('') + ' ' + people_clean['nameLast'].fillna('')
    people_clean['fullName'] = people_clean['fullName'].str.strip()

    batting_merged = pd.merge(batting_agg, people_clean, on='playerID', how='left')

    # Salarios por jugador y año (sumar si hay registros múltiples)
    salaries['salary'] = pd.to_numeric(salaries['salary'], errors='coerce').fillna(0)
    salaries_agg = salaries.groupby(['playerID', 'yearID'], as_index=False)['salary'].sum()

    full_df = pd.merge(batting_merged, salaries_agg, on=['playerID', 'yearID'], how='left')
    full_df['salary'] = full_df['salary'].fillna(0)

    # Eficiencia de costo
    full_df['Cost_per_RC'] = calculate_cost_per_metric(full_df['salary'], full_df['Runs_Created'])

    # Apariciones al plato (PA) y producción normalizada por 600 PA
    full_df['PA'] = full_df['AB'] + full_df['BB'] + full_df['HBP'] + full_df['SF'] + full_df['SH']
    full_df['RC_600'] = np.where(full_df['PA'] > 0, full_df['Runs_Created'] / full_df['PA'] * 600, 0.0)

    # Player Efficiency Index (PEI): RC por PA relativo a la liga de ese año (100 = promedio MLB)
    league = full_df.groupby('yearID').agg(lg_rc=('Runs_Created', 'sum'), lg_pa=('PA', 'sum'))
    league['lg_rc_pa'] = league['lg_rc'] / league['lg_pa']
    full_df = full_df.merge(league[['lg_rc_pa']], left_on='yearID', right_index=True, how='left')
    full_df['PEI'] = np.where(full_df['PA'] > 0,
                              100 * (full_df['Runs_Created'] / full_df['PA'].where(full_df['PA'] > 0, 1)) / full_df['lg_rc_pa'],
                              0.0)
    full_df = full_df.drop(columns=['lg_rc_pa'])

    # Posición principal (la de más juegos en la temporada) desde Appearances
    full_df = full_df.merge(primary_positions(min_year, max_year), on=['playerID', 'yearID'], how='left')
    full_df['Pos'] = full_df['Pos'].fillna('UT')

    # Nombre corto estilo pizarra: "M. Tejada"
    full_df['shortName'] = (full_df['nameFirst'].fillna('').str[:1] + '. ' + full_df['nameLast'].fillna('')).str.strip('. ')

    return full_df

POSITION_COLS = {
    'G_c': 'C', 'G_1b': '1B', 'G_2b': '2B', 'G_3b': '3B', 'G_ss': 'SS',
    'G_lf': 'LF', 'G_cf': 'CF', 'G_rf': 'RF', 'G_dh': 'DH', 'G_p': 'P',
}

def primary_positions(min_year: int, max_year: int) -> pd.DataFrame:
    """Devuelve la posición con más juegos por jugador y temporada (sumando stints por traspasos)."""
    app = fetch_table("Appearances")
    app = app[(app['yearID'] >= min_year) & (app['yearID'] <= max_year)].copy()
    cols = [c for c in POSITION_COLS if c in app.columns]
    for c in cols:
        app[c] = pd.to_numeric(app[c], errors='coerce').fillna(0)
    agg = app.groupby(['playerID', 'yearID'], as_index=False)[cols].sum()
    agg['Pos'] = agg[cols].idxmax(axis=1).map(POSITION_COLS)
    agg.loc[agg[cols].sum(axis=1) == 0, 'Pos'] = 'UT'
    return agg[['playerID', 'yearID', 'Pos']]

def process_teams(min_year: int = 1960, max_year: int = 2023) -> pd.DataFrame:
    """Procesa estadísticas a nivel de equipo incluyendo expectativa pitagórica y métricas de nómina."""
    teams = fetch_table("Teams")
    teams = teams[(teams['yearID'] >= min_year) & (teams['yearID'] <= max_year)].copy()

    numeric_cols = ['W', 'L', 'G', 'R', 'RA', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'HBP', 'SF']
    for col in numeric_cols:
        if col in teams.columns:
            teams[col] = pd.to_numeric(teams[col], errors='coerce').fillna(0)
        else:
            teams[col] = 0

    teams['1B'] = calculate_singles(teams)
    teams['OBP'] = calculate_obp(teams)
    teams['SLG'] = calculate_slg(teams)
    teams['OPS'] = calculate_ops(teams)
    teams['Runs_Created'] = calculate_runs_created(teams)
    teams['Pythagorean_Wins'] = calculate_pythagorean_wins(teams['R'], teams['RA'], total_games=teams['G'])
    teams['Win_Diff'] = teams['W'] - teams['Pythagorean_Wins']

    # Salarios totales por equipo
    salaries = fetch_table("Salaries")
    salaries['salary'] = pd.to_numeric(salaries['salary'], errors='coerce').fillna(0)
    team_payroll = salaries.groupby(['yearID', 'teamID'], as_index=False)['salary'].sum()
    team_payroll = team_payroll.rename(columns={'salary': 'Payroll'})

    teams = pd.merge(teams, team_payroll, on=['yearID', 'teamID'], how='left')
    teams['Payroll'] = teams['Payroll'].fillna(0)
    teams['Cost_per_Win'] = calculate_cost_per_metric(teams['Payroll'], teams['W'])

    return teams

def build_database_and_cache():
    """Ejecuta el ETL completo y persiste las tablas procesadas en SQLite y Parquet."""
    ensure_directories()
    print("Iniciando procesamiento de jugadores y bateo...")
    players_df = process_batting_and_salaries(1980, 2023)
    players_parquet = os.path.join(PROCESSED_DIR, "players_processed.parquet")
    players_df.to_parquet(players_parquet, index=False)
    print(f"Jugadores procesados: {len(players_df)} registros guardados en {players_parquet}")

    print("Iniciando procesamiento de equipos...")
    teams_df = process_teams(1980, 2023)
    teams_parquet = os.path.join(PROCESSED_DIR, "teams_processed.parquet")
    teams_df.to_parquet(teams_parquet, index=False)
    print(f"Equipos procesados: {len(teams_df)} registros guardados en {teams_parquet}")

    print("Guardando en base de datos SQLite...")
    conn = sqlite3.connect(DB_PATH)
    players_df.to_sql("players_sabermetrics", conn, if_exists="replace", index=False)
    teams_df.to_sql("teams_sabermetrics", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Base de datos SQLite lista en: {DB_PATH}")

    return players_df, teams_df

if __name__ == "__main__":
    build_database_and_cache()
