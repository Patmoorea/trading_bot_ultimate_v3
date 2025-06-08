"""
Callbacks pour l'interface utilisateur
Créé: 2025-05-23
@author: Patmoorea
"""
import json
import traceback
import time
from datetime import datetime
import dash
from dash import Input, Output, State, html, callback
import dash_bootstrap_components as dbc
import logging

from modules.web_connector import WebConnector

# Configuration du logging
logger = logging.getLogger(__name__)

def register_callbacks(app, config):
    """Enregistre tous les callbacks de l'application"""
    
    # Callback pour mettre à jour l'horodatage
    @app.callback(
        Output("last-update-time", "children"),
        Input("interval-component", "n_intervals")
    )
    def update_timestamp(n):
        return f"Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}"
    
    # Callback pour la navigation
    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def render_page_content(pathname):
        """Gère la navigation entre les différentes pages"""
        try:
            if pathname == "/":
                return html.Div([
                    html.H2("Dashboard", className="page-title"),
                    html.P("Vue d'ensemble de votre portefeuille et des performances du trading bot.")
                    # Le contenu principal est déjà inclus dans le layout de base
                ])
            
            elif pathname == "/live":
                return html.Div([
                    html.H2("Trading Live", className="page-title"),
                    html.P("Suivi en temps réel des opérations de trading."),
                    dbc.Alert(
                        "Le trading automatique est actif. Monitorez vos stratégies en direct.",
                        color="success"
                    ),
                    # Contenu spécifique au trading live
                    html.Div(id="live-trading-content")
                ])
            
            elif pathname == "/backtest":
                return html.Div([
                    html.H2("Backtesting", className="page-title"),
                    html.P("Testez vos stratégies sur des données historiques."),
                    # Formulaire de backtesting
                    dbc.Card([
                        dbc.CardHeader("Configuration du Backtesting"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Stratégie"),
                                    dbc.Select(
                                        id="strategy-select",
                                        options=[
                                            {"label": "Multi-Exchange Arbitrage", "value": "multi_arb"},
                                            {"label": "USDC/USDT Arbitrage", "value": "stablecoin_arb"},
                                            {"label": "Grid Trading", "value": "grid"}
                                        ],
                                        value="stablecoin_arb"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Période"),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Input(
                                                id="date-start",
                                                type="date",
                                                value="2025-04-01"
                                            ),
                                            width=6
                                        ),
                                        dbc.Col(
                                            dbc.Input(
                                                id="date-end",
                                                type="date",
                                                value="2025-05-01"
                                            ),
                                            width=6
                                        )
                                    ])
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Capital initial"),
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="initial-capital",
                                            type="number",
                                            value=10000,
                                            min=100
                                        ),
                                        dbc.InputGroupText("USD")
                                    ])
                                ], width=4)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Lancer le Backtesting",
                                        id="run-backtest-button",
                                        color="primary",
                                        className="w-100"
                                    )
                                ], width={"size": 4, "offset": 4})
                            ])
                        ])
                    ]),
                    html.Div(id="backtest-results", className="mt-4")
                ])
            
            elif pathname == "/strategies":
                return html.Div([
                    html.H2("Stratégies", className="page-title"),
                    html.P("Gestion et configuration de vos stratégies de trading."),
                    # Contenu pour la gestion des stratégies
                    html.Div(id="strategies-content")
                ])
            
            elif pathname == "/notifications":
                return html.Div([
                    html.H2("Notifications", className="page-title"),
                    html.P("Historique et configuration des alertes et notifications."),
                    # Contenu pour les notifications
                    html.Div(id="notifications-content")
                ])
            
            elif pathname == "/config":
                return html.Div([
                    html.H2("Configuration", className="page-title"),
                    html.P("Paramètres et configuration globale du système de trading."),
                    # Contenu pour la configuration
                    html.Div(id="config-content")
                ])
            
            elif pathname == "/profile":
                return html.Div([
                    html.H2("Profil", className="page-title"),
                    html.P("Gestion de votre profil utilisateur."),
                    # Contenu pour le profil
                    html.Div(id="profile-content")
                ])
            
            # Page non trouvée
            return html.Div([
                html.H1("404: Page non trouvée", className="text-danger"),
                html.P(f"La page {pathname} n'existe pas."),
                dbc.Button("Retour à l'accueil", color="primary", href="/")
            ], className="text-center py-5")
            
        except Exception as e:
            logger.error(f"Erreur lors du rendu de la page {pathname}: {str(e)}")
            logger.error(traceback.format_exc())
            return dbc.Alert(
                f"Une erreur est survenue lors du chargement de la page: {str(e)}",
                color="danger"
            )
    
    # Callback pour le bouton de trading live
    @app.callback(
        [Output("live-toggle", "children"), Output("live-toggle", "color")],
        Input("live-toggle", "n_clicks"),
        State("live-toggle", "children")
    )
    def toggle_live_trading(n_clicks, current_state):
        if n_clicks is None:
            return "Trading Live", "success"
        
        if current_state == "Trading Live":
            return "Trading Désactivé", "danger"
        else:
            return "Trading Live", "success"
    
    # Callback pour exécuter un backtest
    @app.callback(
        Output("backtest-results", "children"),
        Input("run-backtest-button", "n_clicks"),
        [
            State("strategy-select", "value"),
            State("date-start", "value"),
            State("date-end", "value"),
            State("initial-capital", "value")
        ]
    )
    def run_backtest(n_clicks, strategy, start_date, end_date, initial_capital):
        if n_clicks is None:
            return html.Div()
        
        # Dans un système réel, appeler ici le module de backtesting
        # Pour cette démo, nous attendons simplement un peu et renvoyons un résultat factice
        try:
            # Simuler un traitement
            time.sleep(1)
            
            # Exemple de résultats de backtest
            return html.Div([
                dbc.Alert(f"Backtest terminé sur la période du {start_date} au {end_date}", color="success"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Résultats du Backtest"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Performance"),
                                        html.H3("+12.8%", className="text-success"),
                                        html.P("Rendement annualisé: +24.5%")
                                    ], width=4),
                                    dbc.Col([
                                        html.H5("Statistiques"),
                                        html.Div([
                                            html.Div(["Ratio de Sharpe: ", html.Span("1.75", className="fw-bold")], className="d-flex justify-content-between"),
                                            html.Div(["Drawdown Max: ", html.Span("-4.2%", className="fw-bold text-danger")], className="d-flex justify-content-between"),
                                            html.Div(["Nombre de trades: ", html.Span("78", className="fw-bold")], className="d-flex justify-content-between"),
                                            html.Div(["Taux de réussite: ", html.Span("65.4%", className="fw-bold text-success")], className="d-flex justify-content-between")
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        html.H5("Actions"),
                                        dbc.Button("Télécharger Rapport", color="primary", className="me-2"),
                                        dbc.Button("Déployer en Live", color="success")
                                    ], width=4)
                                ])
                            ])
                        ])
                    ])
                ]),
                # Graphiques factices pour les résultats
                html.Div(id="backtest-charts", className="mt-4")
            ])
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return dbc.Alert(
                f"Une erreur est survenue lors de l'exécution du backtest: {str(e)}",
                color="danger"
            )

    # Autres callbacks à ajouter selon les besoins
