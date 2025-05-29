import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timezone
from ..services.technical_analysis import TechnicalAnalysis
from ..services.ai_engine import AIDecisionEngine
from ..services.risk_manager import RiskManager

class BacktestEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ta = TechnicalAnalysis()
        self.ai = AIDecisionEngine()
        self.risk = RiskManager()
        
    async def run_backtest(self,
                          data: pd.DataFrame,
                          initial_capital: float = 100000.0,
                          test_method: str = "walk_forward") -> Dict:
        try:
            if test_method == "walk_forward":
                return await self._walk_forward_analysis(data, initial_capital)
            elif test_method == "monte_carlo":
                return await self._monte_carlo_simulation(data, initial_capital)
            elif test_method == "bootstrap":
                return await self._bootstrap_analysis(data, initial_capital)
            elif test_method == "stress_test":
                return await self._stress_testing(data, initial_capital)
            else:
                raise ValueError(f"Unknown test method: {test_method}")
                
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {"status": "error", "reason": str(e)}

    async def _walk_forward_analysis(self,
                                   data: pd.DataFrame,
                                   initial_capital: float) -> Dict:
        try:
            results = []
            window_size = len(data) // 5  # 20% pour chaque fenêtre
            
            for i in range(4):  # 4 fenêtres de test
                # Découpage des données
                train_data = data[i*window_size:(i+3)*window_size]  # 60% pour l'entraînement
                test_data = data[(i+3)*window_size:(i+4)*window_size]  # 20% pour le test
                
                # Optimisation sur les données d'entraînement
                optimal_params = await self._optimize_parameters(train_data)
                
                # Test sur les données de test
                window_results = await self._run_single_test(
                    test_data,
                    initial_capital,
                    optimal_params
                )
                
                results.append(window_results)
            
            return self._aggregate_results(results)
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis error: {e}")
            return {"status": "error", "reason": str(e)}

    async def _monte_carlo_simulation(self,
                                    data: pd.DataFrame,
                                    initial_capital: float,
                                    n_simulations: int = 1000) -> Dict:
        try:
            simulation_results = []
            
            # Calcul des rendements et de la volatilité
            returns = np.diff(np.log(data['close']))
            volatility = np.std(returns)
            drift = np.mean(returns) - (volatility ** 2) / 2
            
            # Simulations
            for _ in range(n_simulations):
                # Génération de prix simulés
                random_walks = np.random.standard_normal(len(data))
                price_path = data['close'].iloc[0] * np.exp(
                    np.cumsum(drift + volatility * random_walks)
                )
                
                # Création d'un DataFrame simulé
                simulated_data = data.copy()
                simulated_data['close'] = price_path
                
                # Exécution du test
                sim_result = await self._run_single_test(
                    simulated_data,
                    initial_capital
                )
                
                simulation_results.append(sim_result)
            
            return self._aggregate_monte_carlo_results(simulation_results)
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation error: {e}")
            return {"status": "error", "reason": str(e)}

    async def _stress_testing(self,
                            data: pd.DataFrame,
                            initial_capital: float) -> Dict:
        try:
            stress_scenarios = [
                {"name": "Market Crash", "price_modifier": 0.7},
                {"name": "High Volatility", "volatility_multiplier": 3},
                {"name": "Low Liquidity", "volume_modifier": 0.2},
                {"name": "Flash Crash", "flash_crash_severity": 0.3}
            ]
            
            stress_results = []
            
            for scenario in stress_scenarios:
                # Création des données de stress
                stressed_data = self._create_stress_scenario(data, scenario)
                
                # Exécution du test
                scenario_result = await self._run_single_test(
                    stressed_data,
                    initial_capital
                )
                
                stress_results.append({
                    "scenario": scenario["name"],
                    "results": scenario_result
                })
            
            return self._aggregate_stress_results(stress_results)
            
        except Exception as e:
            self.logger.error(f"Stress testing error: {e}")
            return {"status": "error", "reason": str(e)}

    def _create_stress_scenario(self,
                              data: pd.DataFrame,
                              scenario: Dict) -> pd.DataFrame:
        stressed_data = data.copy()
        
        if "price_modifier" in scenario:
            stressed_data['close'] *= scenario["price_modifier"]
            stressed_data['high'] *= scenario["price_modifier"]
            stressed_data['low'] *= scenario["price_modifier"]
            
        if "volatility_multiplier" in scenario:
            returns = np.diff(np.log(stressed_data['close']))
            new_returns = returns * scenario["volatility_multiplier"]
            stressed_data['close'] = np.exp(np.cumsum(new_returns))
            
        if "volume_modifier" in scenario:
            stressed_data['volume'] *= scenario["volume_modifier"]
            
        if "flash_crash_severity" in scenario:
            crash_idx = len(stressed_data) // 2
            stressed_data.loc[crash_idx:crash_idx+5, 'close'] *= (
                1 - scenario["flash_crash_severity"]
            )
            
        return stressed_data

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        try:
            all_returns = [r["returns"] for r in results]
            all_trades = [r["trades"] for r in results]
            
            return {
                "status": "success",
                "metrics": {
                    "sharpe_ratio": np.mean([r["metrics"]["sharpe_ratio"] for r in results]),
                    "max_drawdown": np.mean([r["metrics"]["max_drawdown"] for r in results]),
                    "win_rate": np.mean([r["metrics"]["win_rate"] for r in results]),
                    "profit_factor": np.mean([r["metrics"]["profit_factor"] for r in results])
                },
                "returns": np.mean(all_returns, axis=0).tolist(),
                "trades": sum(all_trades, []),
                "timestamp": datetime.now(timezone.utc)
            }
        except Exception as e:
            self.logger.error(f"Results aggregation error: {e}")
            return {"status": "error", "reason": str(e)}
