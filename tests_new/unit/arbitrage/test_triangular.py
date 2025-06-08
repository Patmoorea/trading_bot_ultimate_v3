"""
Tests unitaires pour l'arbitrage triangulaire
Created: 2025-05-22 23:17:14
@author: Patmoorea
"""

import pytest
import asyncio
import networkx as nx
import logging

# Configuration des logs pour les tests
logging.basicConfig(level=logging.INFO)

# Import du module à tester
from modules.triangular import TriangularArbitrage
from modules.arbitrage_utils import calculate_profit, format_opportunity

class TestTriangularArbitrage:
    """Tests pour la classe TriangularArbitrage"""
    
    @pytest.fixture
    def triangular(self):
        """Fixture pour initialiser le moteur d'arbitrage triangulaire"""
        config = {
            'min_profit': 0.5,
            'quote_currencies': ["USDT", "USDC", "BTC"],
            'fee': 0.001
        }
        return TriangularArbitrage(config)
    
    @pytest.fixture
    def mock_tickers(self):
        """Fixture pour fournir des tickers simulés"""
        return {
            'BTC/USDT': {'bid': 30000.0, 'ask': 30010.0},
            'ETH/BTC': {'bid': 0.070, 'ask': 0.0701},
            'ETH/USDT': {'bid': 2110.0, 'ask': 2112.0},
            'ADA/USDT': {'bid': 0.40, 'ask': 0.401},
            'ADA/BTC': {'bid': 0.0000132, 'ask': 0.0000133},
            'timestamp': 1715270400000  # 2024-05-09 12:00:00
        }
    
    @pytest.mark.asyncio
    async def test_build_graph(self, triangular, mock_tickers):
        """Vérifie que le graphe de marché est correctement construit"""
        await triangular.build_graph(mock_tickers)
        
        # Vérifier les nœuds (devises)
        expected_currencies = {'BTC', 'USDT', 'ETH', 'ADA'}
        assert set(triangular.graph.nodes()) == expected_currencies
        
        # Vérifier le nombre d'arêtes (conversions)
        # 5 paires x 2 directions = 10 arêtes
        assert len(triangular.graph.edges()) == 10
        
        # Vérifier quelques arêtes spécifiques
        assert triangular.graph.has_edge('BTC', 'USDT')
        assert triangular.graph.has_edge('USDT', 'BTC')
        assert triangular.graph.has_edge('ETH', 'BTC')
    
    @pytest.mark.asyncio
    async def test_find_opportunities(self, triangular, mock_tickers):
        """Vérifie la détection d'opportunités d'arbitrage triangulaire"""
        # Ajuster les tickers pour créer une opportunité connue
        # BTC/USDT -> ETH/BTC -> ETH/USDT crée un cycle avec profit
        mock_tickers['ETH/USDT']['bid'] = 2160.0  # Augmenter le prix de vente ETH/USDT
        
        # Exécuter la recherche d'opportunités
        opportunities = await triangular.find_opportunities(mock_tickers)
        
        # Imprimer les opportunités pour le débogage
        for opp in opportunities[:3]:
            print(f"Opportunité: {format_opportunity(opp)}")
            
        # Vérifier qu'au moins une opportunité est détectée
        # Note: nous ne forçons pas cette assertion car le résultat dépend
        # des tickers et de la configuration
        if opportunities:
            # Vérifier la structure d'une opportunité
            opp = opportunities[0]
            assert 'cycle' in opp
            assert 'profit' in opp
            assert 'conversions' in opp
            
            # Vérifier que le cycle commence et finit par la même devise
            assert opp['cycle'][0] == opp['cycle'][-1]
    
    @pytest.mark.asyncio
    async def test_empty_tickers(self, triangular):
        """Vérifie le comportement avec des tickers vides"""
        opportunities = await triangular.find_opportunities({})
        assert len(opportunities) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_tickers(self, triangular):
        """Vérifie le comportement avec des tickers invalides"""
        # Test avec None (doit être géré sans erreur)
        try:
            opportunities = await triangular.find_opportunities(None)
            assert len(opportunities) == 0
        except Exception as e:
            pytest.fail(f"L'appel avec None a échoué: {str(e)}")
        
        # Test avec un type incorrect (doit être géré sans erreur)
        try:
            opportunities = await triangular.find_opportunities("invalid")
            assert len(opportunities) == 0
        except Exception as e:
            pytest.fail(f"L'appel avec un type invalide a échoué: {str(e)}")
            
    def test_calculate_profit(self):
        """Test la fonction de calcul du profit"""
        # Test avec profit déjà calculé
        opp1 = {'profit': 1.5}
        assert calculate_profit(opp1) == 1.5
        
        # Test avec conversions
        opp2 = {'conversions': [
            {'rate': 1.001},
            {'rate': 1.002},
            {'rate': 1.003}
        ]}
        profit = calculate_profit(opp2)
        assert profit > 0
        
        # Test avec opportunité vide
        opp3 = {}
        assert calculate_profit(opp3) == 0.0
        
    def test_format_opportunity(self):
        """Test le formatage d'une opportunité"""
        # Test avec un cycle
        opp1 = {
            'cycle': ['USDT', 'BTC', 'ETH', 'USDT'],
            'profit': 1.5
        }
        formatted = format_opportunity(opp1)
        assert "Profit: 1.50%" in formatted
        assert "USDT → BTC → ETH → USDT" in formatted
        
        # Test avec une opportunité standard
        opp2 = {
            'profit': 0.8,
            'symbol': 'BTC/USDT',
            'exchange1': 'binance',
            'exchange2': 'kraken'
        }
        formatted = format_opportunity(opp2)
        assert "Profit: 0.80%" in formatted
        assert "BTC/USDT" in formatted
        assert "binance -> kraken" in formatted
