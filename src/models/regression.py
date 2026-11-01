import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from src.sabermetrics.metrics import calculate_pythagorean_wins

class MoneyballRegressionModel:
    """
    Modelo de Regresión Lineal múltiple para predecir Carreras Anotadas (R)
    en base a Porcentaje de Embasado (OBP) y Slugging (SLG) colectivo del equipo.
    Fundamento matemático del sistema de Oakland Athletics de 2002.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
        self.r2_ = 0.0
        self.rmse_ = 0.0
        self.coef_obp_ = 0.0
        self.coef_slg_ = 0.0
        self.intercept_ = 0.0

    def fit(self, teams_df: pd.DataFrame):
        """
        Entrena el modelo con estadísticas a nivel de equipo.
        Requiere columnas: OBP, SLG, R (Carreras anotadas).
        """
        df = teams_df.dropna(subset=['OBP', 'SLG', 'R']).copy()
        df = df[(df['OBP'] > 0) & (df['SLG'] > 0) & (df['R'] > 0)]

        X = df[['OBP', 'SLG']]
        y = df['R']

        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        self.r2_ = r2_score(y, y_pred)
        self.rmse_ = np.sqrt(mean_squared_error(y, y_pred))
        self.coef_obp_ = self.model.coef_[0]
        self.coef_slg_ = self.model.coef_[1]
        self.intercept_ = self.model.intercept_
        self.is_fitted = True

        return self

    def predict_runs(self, obp: float | np.ndarray, slg: float | np.ndarray) -> float | np.ndarray:
        """Proyecta las carreras de un equipo según su OBP y SLG colectivo."""
        if not self.is_fitted:
            # Coeficientes estándar históricos calibrados (1960-2002) de referencia
            return -800.0 + (2800.0 * obp) + (1600.0 * slg)
        
        X = pd.DataFrame({'OBP': np.atleast_1d(obp), 'SLG': np.atleast_1d(slg)})
        preds = self.model.predict(X)
        return preds[0] if np.isscalar(obp) else preds

    def predict_wins(self, 
                     team_obp: float, 
                     team_slg: float, 
                     runs_allowed: float = 654.0, 
                     total_games: int = 162) -> dict:
        """
        Calcula las carreras proyectadas y las victorias esperadas utilizando
        la Expectativa Pitagórica de Victorias.
        (654 RA fue el registro defensivo real de Oakland en 2002).
        """
        proj_runs = float(self.predict_runs(team_obp, team_slg))
        exp_wins = float(calculate_pythagorean_wins(proj_runs, runs_allowed, exponent=2.0, total_games=total_games))

        return {
            "projected_runs": round(proj_runs, 1),
            "runs_allowed": round(runs_allowed, 1),
            "projected_wins": round(exp_wins, 1),
            "projected_losses": round(total_games - exp_wins, 1),
            "win_percentage": round(exp_wins / total_games, 3)
        }

    def get_insights(self) -> dict:
        """Retorna el desglose del modelo y la relación de valor OBP vs SLG."""
        ratio = self.coef_obp_ / self.coef_slg_ if self.coef_slg_ != 0 else 0
        return {
            "r2": round(self.r2_, 3),
            "rmse": round(self.rmse_, 1),
            "intercept": round(self.intercept_, 2),
            "coef_obp": round(self.coef_obp_, 2),
            "coef_slg": round(self.coef_slg_, 2),
            "obp_vs_slg_multiplier": round(ratio, 2),
            "explanation": f"Cada punto de OBP vale aproximadamente {ratio:.2f} veces más que un punto de SLG en la producción neta de carreras."
        }
