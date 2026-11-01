import unittest
import pandas as pd
import numpy as np
from src.sabermetrics.metrics import (
    calculate_singles,
    calculate_obp,
    calculate_slg,
    calculate_ops,
    calculate_runs_created,
    calculate_pythagorean_wins
)

class TestSabermetrics(unittest.TestCase):

    def test_calculate_singles(self):
        df = pd.DataFrame({
            'H': [150],
            '2B': [30],
            '3B': [5],
            'HR': [25]
        })
        singles = calculate_singles(df)
        self.assertEqual(singles.iloc[0], 90)

    def test_calculate_obp_standard(self):
        # H=150, BB=80, HBP=10, AB=500, SF=5
        # OBP = (150 + 80 + 10) / (500 + 80 + 10 + 5) = 240 / 595 = ~0.40336
        df = pd.DataFrame({
            'H': [150],
            'BB': [80],
            'HBP': [10],
            'AB': [500],
            'SF': [5]
        })
        obp = calculate_obp(df)
        self.assertEqual(round(float(obp[0]), 3), 0.403)

    def test_calculate_obp_pre_1954_no_sf(self):
        # Simulación de era previa a 1954 donde SF es NaN o columna ausente
        df = pd.DataFrame({
            'H': [100],
            'BB': [50],
            'HBP': [5],
            'AB': [400],
            'SF': [np.nan]
        })
        obp = calculate_obp(df)
        expected = (100 + 50 + 5) / (400 + 50 + 5 + 0)
        self.assertEqual(round(float(obp[0]), 4), round(expected, 4))

    def test_calculate_slg(self):
        # 1B=70, 2B=20, 3B=2, HR=18, AB=400
        # TB = 70 + (2*20) + (3*2) + (4*18) = 70 + 40 + 6 + 72 = 188
        # SLG = 188 / 400 = 0.470
        df = pd.DataFrame({
            '1B': [70],
            '2B': [20],
            '3B': [2],
            'HR': [18],
            'AB': [400]
        })
        slg = calculate_slg(df)
        self.assertEqual(round(float(slg[0]), 3), 0.470)

    def test_zero_division_safety(self):
        # Jugador sin turnos al bate (lanzador de relevo o sustituto defensivo)
        df = pd.DataFrame({
            'H': [0],
            '2B': [0],
            '3B': [0],
            'HR': [0],
            'AB': [0],
            'BB': [0],
            'HBP': [0],
            'SF': [0]
        })
        obp = calculate_obp(df)
        slg = calculate_slg(df)
        rc = calculate_runs_created(df)
        self.assertEqual(float(obp[0]), 0.0)
        self.assertEqual(float(slg[0]), 0.0)
        self.assertEqual(float(rc[0]), 0.0)

    def test_pythagorean_wins_oakland_2002(self):
        # Oakland 2002: R=800, RA=654 en 162 juegos
        # Win% = 800^2 / (800^2 + 654^2) = 640,000 / (640,000 + 427,716) = ~0.5994
        # Victorias esperadas = 0.5994 * 162 = ~97.1 victorias (Oakland ganó 103, +5.9 sobre el modelo)
        exp_wins = calculate_pythagorean_wins(800, 654, total_games=162)
        self.assertEqual(round(float(exp_wins), 1), 97.1)

if __name__ == '__main__':
    unittest.main()
