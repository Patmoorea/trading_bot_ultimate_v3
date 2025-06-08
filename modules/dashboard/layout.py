"""
Définition des layouts de l'interface utilisateur
Créé: 2025-05-23
@author: Patmoorea
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from modules.dashboard.components import (
    create_header, create_sidebar, create_portfolio_summary,
    create_active_strategies, create_recent_trades,
    create_equity_chart, create_drawdown_chart, create_returns_chart
)

def create_layout(config):
    """Crée le layout principal de l'application"""
    features = config["features"]
    layouts = config["layouts"]["dashboard"]
    
    # Créer le contenu principal en fonction de la configuration
    main_content = []
    for component in layouts["main_components"]:
        if component == "portfolio_summary":
            main_content.append(create_portfolio_summary())
        elif component == "active_strategies":
            main_content.append(create_active_strategies())
        elif component == "recent_trades":
            main_content.append(create_recent_trades())
    
    # Créer les graphiques en fonction de la configuration
    charts = []
    for chart in layouts["charts"]:
        if chart == "equity_curve":
            charts.append(create_equity_chart())
        elif chart == "drawdown":
            charts.append(create_drawdown_chart())
        elif chart == "daily_returns":
            charts.append(create_returns_chart())
    
    # Layout complet
    return html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="session-store", storage_type="session"),
        dcc.Store(id="theme-store", storage_type="local"),
        dcc.Interval(id="interval-component", interval=30*1000, n_intervals=0),  # Rafraîchissement toutes les 30s
        
        # Header
        create_header(),
        
        # Corps principal avec sidebar et contenu
        dbc.Container(
            [
                dbc.Row(
                    [
                        # Sidebar
                        dbc.Col(create_sidebar(features), width=2, className="sidebar"),
                        
                        # Contenu principal
                        dbc.Col(
                            [
                                html.Div(id="page-content", children=[
                                    # Résumé principal
                                    dbc.Row(main_content, className="mb-4"),
                                    
                                    # Graphiques
                                    dbc.Row(charts, className="mb-4"),
                                    
                                    # Zone pour le contenu dynamique
                                    html.Div(id="dynamic-content")
                                ])
                            ],
                            width=10,
                            className="content-container"
                        )
                    ]
                )
            ],
            fluid=True,
            className="main-container"
        ),
        
        # Footer
        html.Footer(
            dbc.Container(
                [
                    html.Hr(),
                    html.P(
                        "Trading Bot Ultimate © 2025 - Développé par Patmoorea",
                        className="text-center text-muted"
                    )
                ]
            ),
            className="footer"
        )
    ])
