"""
Composants réutilisables pour l'interface utilisateur
Créé: 2025-05-23
@author: Patmoorea
"""
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_header():
    """Crée l'en-tête de l'application"""
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="/assets/logo.png",
                                height="40px",
                                className="navbar-logo"
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Trading Bot Ultimate",
                                className="navbar-title"
                            ),
                            width="auto"
                        )
                    ],
                    align="center",
                    className="g-0"
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
                            dbc.NavItem(dbc.NavLink("Backtesting", href="/backtest")),
                            dbc.NavItem(dbc.NavLink("Stratégies", href="/strategies")),
                            dbc.NavItem(dbc.NavLink("Configuration", href="/config")),
                            dbc.DropdownMenu(
                                [
                                    dbc.DropdownMenuItem("Profil", href="/profile"),
                                    dbc.DropdownMenuItem("Paramètres", href="/settings"),
                                    dbc.DropdownMenuItem(divider=True),
                                    dbc.DropdownMenuItem("Déconnexion", href="/logout"),
                                ],
                                label="Mon Compte",
                                nav=True,
                                in_navbar=True,
                            ),
                            dbc.NavItem(
                                dbc.Button(
                                    "Trading Live",
                                    id="live-toggle",
                                    color="success",
                                    className="ms-2"
                                )
                            )
                        ],
                        className="ms-auto",
                        navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            fluid=True
        ),
        color="dark",
        dark=True,
        className="mb-4"
    )

def create_sidebar(features):
    """Crée la barre latérale avec navigation"""
    nav_items = [
        dbc.NavItem(dbc.NavLink("Vue d'ensemble", href="/", active="exact")),
    ]
    
    if features.get("live_trading", False):
        nav_items.append(dbc.NavItem(dbc.NavLink("Trading Live", href="/live", active="exact")))
    
    if features.get("backtesting", False):
        nav_items.append(dbc.NavItem(dbc.NavLink("Backtesting", href="/backtest", active="exact")))
    
    if features.get("strategy_editor", False):
        nav_items.append(dbc.NavItem(dbc.NavLink("Stratégies", href="/strategies", active="exact")))
    
    if features.get("notifications", False):
        nav_items.append(dbc.NavItem(dbc.NavLink("Notifications", href="/notifications", active="exact")))
        
    nav_items.extend([
        dbc.NavItem(dbc.NavLink("Configuration", href="/config", active="exact")),
        dbc.NavItem(dbc.NavLink("Documentation", href="/docs", active="exact")),
    ])
    
    return html.Div(
        [
            html.H5("Navigation", className="sidebar-heading"),
            dbc.Nav(
                nav_items,
                vertical=True,
                pills=True,
                className="flex-column sidebar-nav"
            ),
            html.Hr(),
            html.Div(
                [
                    html.H6("Statut du système", className="status-heading"),
                    html.Div(
                        [
                            html.Span("✓ ", className="status-icon text-success"),
                            "API Exchanges: Connecté"
                        ],
                        className="status-item"
                    ),
                    html.Div(
                        [
                            html.Span("✓ ", className="status-icon text-success"),
                            "Database: Opérationnelle"
                        ],
                        className="status-item"
                    ),
                    html.Div(
                        [
                            dbc.Badge("Actif", color="success", className="me-1"),
                            html.Span(id="last-update-time")
                        ],
                        className="status-update"
                    )
                ],
                className="system-status"
            )
        ],
        className="sidebar-content"
    )

def create_portfolio_summary():
    """Crée le résumé du portefeuille"""
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Résumé du Portefeuille")),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6("Valeur totale"),
                                    html.H3(
                                        html.Span("$24,856.78", id="portfolio-value"),
                                        className="text-primary"
                                    ),
                                    html.P(
                                        [
                                            html.Span("+3.2% ", className="text-success"),
                                            html.Small("dernières 24h")
                                        ]
                                    )
                                ],
                                width=4
                            ),
                            dbc.Col(
                                [
                                    html.H6("Performance totale"),
                                    dbc.Progress(
                                        [
                                            dbc.Progress(value=15, color="success", bar=True),
                                            dbc.Progress(value=0, color="danger", bar=True)
                                        ],
                                        className="mb-2",
                                        style={"height": "20px"}
                                    ),
                                    html.P(
                                        [
                                            html.Span("+15.4% ", className="text-success"),
                                            html.Small("depuis le début")
                                        ]
                                    )
                                ],
                                width=8
                            )
                        ],
                        className="mb-4"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6("Répartition des actifs"),
                                    dcc.Graph(
                                        id="portfolio-distribution",
                                        figure={
                                            "data": [
                                                {
                                                    "labels": ["BTC", "ETH", "USDC", "USDT", "Autres"],
                                                    "values": [35, 25, 20, 15, 5],
                                                    "type": "pie",
                                                    "hole": 0.4,
                                                    "marker": {
                                                        "colors": ["#f9a825", "#64b5f6", "#2e7d32", "#0d47a1", "#9e9e9e"]
                                                    }
                                                }
                                            ],
                                            "layout": {
                                                "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
                                                "showlegend": True,
                                                "legend": {"orientation": "h", "y": -0.1},
                                                "height": 200,
                                                "paper_bgcolor": "rgba(0,0,0,0)",
                                                "plot_bgcolor": "rgba(0,0,0,0)",
                                            }
                                        },
                                        config={"displayModeBar": False}
                                    )
                                ],
                                width=6
                            ),
                            dbc.Col(
                                [
                                    html.H6("Métriques clés"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Span("Drawdown Max", className="metric-label"),
                                                    html.Span("-7.8%", className="metric-value text-danger")
                                                ],
                                                className="d-flex justify-content-between mb-2"
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("Ratio de Sharpe", className="metric-label"),
                                                    html.Span("1.85", className="metric-value text-success")
                                                ],
                                                className="d-flex justify-content-between mb-2"
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("Trades gagnants", className="metric-label"),
                                                    html.Span("68%", className="metric-value text-success")
                                                ],
                                                className="d-flex justify-content-between mb-2"
                                            ),
                                            html.Div(
                                                [
                                                    html.Span("Exposition actuelle", className="metric-label"),
                                                    html.Span("42%", className="metric-value")
                                                ],
                                                className="d-flex justify-content-between"
                                            )
                                        ],
                                        className="metrics-container"
                                    )
                                ],
                                width=6
                            )
                        ]
                    )
                ]
            )
        ],
        className="portfolio-summary-card"
    )

def create_active_strategies():
    """Crée la carte des stratégies actives"""
    strategies = [
        {"name": "Multi-Exchange Arbitrage", "status": "Actif", "profit": "+2.3%", "active_since": "2j 15h"},
        {"name": "USDC/USDT Arbitrage", "status": "Actif", "profit": "+0.8%", "active_since": "12h 45m"},
        {"name": "BTC Grid Trading", "status": "Inactif", "profit": "+4.2%", "active_since": "-"}
    ]
    
    strategy_rows = []
    for strategy in strategies:
        status_color = "success" if strategy["status"] == "Actif" else "secondary"
        profit_class = "text-success" if "+" in strategy["profit"] else "text-danger"
        
        strategy_rows.append(
            html.Tr([
                html.Td(strategy["name"]),
                html.Td(dbc.Badge(strategy["status"], color=status_color)),
                html.Td(html.Span(strategy["profit"], className=profit_class)),
                html.Td(strategy["active_since"]),
                html.Td(
                    dbc.ButtonGroup(
                        [
                            dbc.Button("▶", size="sm", color="light", className="p-1"),
                            dbc.Button("⏸", size="sm", color="light", className="p-1"),
                            dbc.Button("⚙", size="sm", color="light", className="p-1")
                        ],
                        size="sm"
                    )
                )
            ])
        )
    
    return dbc.Card(
        [
            dbc.CardHeader(
                dbc.Row(
                    [
                        dbc.Col(html.H5("Stratégies actives"), width="auto"),
                        dbc.Col(
                            dbc.Button(
                                "Nouvelle stratégie",
                                color="primary",
                                size="sm",
                                className="float-end"
                            ),
                            width="auto",
                            className="ms-auto"
                        )
                    ]
                )
            ),
            dbc.CardBody(
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Nom"),
                                html.Th("Statut"),
                                html.Th("Performance"),
                                html.Th("Actif depuis"),
                                html.Th("Actions")
                            ])
                        ),
                        html.Tbody(strategy_rows)
                    ],
                    striped=True,
                    bordered=False,
                    hover=True,
                    className="strategies-table"
                )
            )
        ],
        className="strategies-card mb-4"
    )

def create_recent_trades():
    """Crée la carte des transactions récentes"""
    trades = [
        {"pair": "BTC/USDT", "side": "BUY", "size": "0.05", "price": "53,245.78", "time": "16:32:45", "strategy": "Multi-Exchange"},
        {"pair": "ETH/USDT", "side": "SELL", "size": "0.35", "price": "2,845.12", "time": "15:18:23", "strategy": "Grid Trading"},
        {"pair": "USDC/USDT", "side": "BUY", "size": "1000", "price": "0.9985", "time": "14:42:18", "strategy": "Stablecoin Arb"}
    ]
    
    trade_rows = []
    for trade in trades:
        side_color = "success" if trade["side"] == "BUY" else "danger"
        
        trade_rows.append(
            html.Tr([
                html.Td(trade["time"]),
                html.Td(trade["pair"]),
                html.Td(dbc.Badge(trade["side"], color=side_color)),
                html.Td(trade["size"]),
                html.Td(trade["price"]),
                html.Td(html.Small(trade["strategy"]))
            ])
        )
    
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Transactions récentes")),
            dbc.CardBody(
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("Heure"),
                                html.Th("Paire"),
                                html.Th("Type"),
                                html.Th("Taille"),
                                html.Th("Prix"),
                                html.Th("Stratégie")
                            ])
                        ),
                        html.Tbody(trade_rows)
                    ],
                    striped=True,
                    bordered=False,
                    hover=True,
                    className="trades-table"
                )
            )
        ],
        className="trades-card"
    )

def _generate_dummy_equity_data():
    """Génère des données d'équité factices pour les graphiques"""
    dates = pd.date_range(end=datetime.now(), periods=60).tolist()
    
    # Equity curve with general uptrend and some fluctuations
    base = 10000
    random_walk = np.random.normal(0.001, 0.008, len(dates)).cumsum()
    trend = np.linspace(0, 0.15, len(dates))
    equity = base * (1 + random_walk + trend)
    
    # Calculate drawdown
    peak = equity.copy()
    for i in range(1, len(equity)):
        peak[i] = max(peak[i-1], equity[i])
    drawdown = (equity - peak) / peak
    
    # Calculate daily returns
    returns = pd.Series(equity).pct_change().fillna(0).values
    
    return dates, equity, drawdown, returns

def create_equity_chart():
    """Crée le graphique de courbe d'équité"""
    dates, equity, _, _ = _generate_dummy_equity_data()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#2196f3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        )
    )
    
    fig.update_layout(
        title="Courbe d'équité",
        xaxis_title="Date",
        yaxis_title="Valeur ($)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat='%d/%m/%y'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat='$,.0f'
        ),
        showlegend=False
    )
    
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Performance du portefeuille")),
            dbc.CardBody(
                dcc.Graph(
                    id="equity-curve-graph",
                    figure=fig,
                    config={"displayModeBar": False}
                )
            )
        ],
        className="mb-4"
    )

def create_drawdown_chart():
    """Crée le graphique de drawdown"""
    dates, _, drawdown, _ = _generate_dummy_equity_data()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color='#f44336', width=2),
            fill='tozeroy',
            fillcolor='rgba(244, 67, 54, 0.1)'
        )
    )
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat='%d/%m/%y'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat=',.1f%'
        ),
        showlegend=False
    )
    
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Drawdown")),
            dbc.CardBody(
                dcc.Graph(
                    id="drawdown-graph",
                    figure=fig,
                    config={"displayModeBar": False}
                )
            )
        ],
        className="mb-4"
    )

def create_returns_chart():
    """Crée le graphique des rendements quotidiens"""
    dates, _, _, returns = _generate_dummy_equity_data()
    
    # Convert returns to percentage
    returns_pct = returns * 100
    
    # Create colors based on positive/negative returns
    colors = ['#4caf50' if r >= 0 else '#f44336' for r in returns_pct]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=returns_pct,
            marker_color=colors,
            name='Rendements quotidiens'
        )
    )
    
    fig.update_layout(
        title="Rendements quotidiens",
        xaxis_title="Date",
        yaxis_title="Rendement (%)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat='%d/%m/%y'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat='+.2f%'
        ),
        showlegend=False
    )
    
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Rendements quotidiens")),
            dbc.CardBody(
                dcc.Graph(
                    id="returns-graph",
                    figure=fig,
                    config={"displayModeBar": False}
                )
            )
        ],
        className="mb-4"
    )
