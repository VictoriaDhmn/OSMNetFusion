import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from unittest.mock import patch, MagicMock, Mock, mock_open
import os
import tempfile
import json

import osmnetfusion.p2_enrichData as p2


class TestGetLanduseRatio:
    """Test cases for get_landuse_ratio function"""
    
    def test_get_landuse_ratio_success(self):
        """Test successful landuse ratio calculation"""
        # Create test data
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            'length': [1.414, 1.414],
            'osmid': [1, 2]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        # Mock the landuse file layers
        mock_layers = {
            'Point': gpd.GeoDataFrame({
                'geometry': [Point(0.5, 0.5), Point(1.5, 1.5)],
                'amenity': ['retail', 'retail']
            }, crs='EPSG:4326'),
            'Polygon': gpd.GeoDataFrame({
                'geometry': [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
                'amenity': ['retail']
            }, crs='EPSG:4326')
        }
        
        with patch('osmnetfusion.p2_enrichData.fiona.listlayers') as mock_listlayers:
            with patch('osmnetfusion.p2_enrichData.gpd.read_file') as mock_read_file:
                mock_listlayers.return_value = ['Point', 'Polygon']
                mock_read_file.side_effect = lambda file, layer: mock_layers[layer]
                
                result = p2.get_landuse_ratio(gdf_edges, kind='retail', input_file='test.gpkg')
                
                assert 'retail_points' in result.columns
                assert 'retail_ratio_point' in result.columns
                assert 'retail_ratio_poly' in result.columns
                assert 'retail_ratio' in result.columns
                assert len(result) == 2
    
    def test_get_landuse_ratio_empty_file(self):
        """Test handling of empty landuse file"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'length': [1.414],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        with patch('osmnetfusion.p2_enrichData.fiona.listlayers') as mock_listlayers:
            with patch('osmnetfusion.p2_enrichData.gpd.read_file') as mock_read_file:
                mock_listlayers.return_value = ['Point']
                mock_read_file.return_value = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
                
                result = p2.get_landuse_ratio(gdf_edges, kind='retail', input_file='test.gpkg')
                
                assert result['retail_points'].iloc[0] == 0
                assert result['retail_ratio'].iloc[0] == 0


class TestImproveBikeEdges:
    """Test cases for improve_bike_edges function"""
    
    def test_improve_bike_edges_basic(self):
        """Test basic bike edge improvement"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            'highway': ['residential', 'trunk'],
            'bicycle': ['yes', 'no'],
            'u': [1, 2],
            'v': [2, 3],
            'oneway': [False, True],
            'reversed': [False, False],
            'key': [0, 0]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result = p2.improve_bike_edges(gdf_edges)
        
        assert 'bike_access' in result.columns
        assert result.loc[0, 'bike_access'] == 'yes'
        assert result.loc[1, 'bike_access'] == 'no'
    
    def test_improve_bike_edges_oneway_opposite(self):
        """Test adding opposite edges for oneway streets"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'highway': ['residential'],
            'bicycle': ['yes'],
            'u': [1],
            'v': [2],
            'oneway': [True],
            'reversed': [False],
            'key': [0]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result = p2.improve_bike_edges(gdf_edges)
        
        # Should add one opposite edge
        assert len(result) == 2
        assert result.loc[1, 'bike_access'] == 'bike_only'
        assert result.loc[1, 'u'] == 2
        assert result.loc[1, 'v'] == 1


class TestMergeSimilarColumns:
    """Test cases for merge_similar_columns function"""
    
    def test_merge_similar_columns_basic(self):
        """Test merging two similar columns"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'surface': ['asphalt'],
            'surface_30': [''],
            'highway': ['residential']
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result = p2.merge_similar_columns(gdf_edges, 'surface', 'surface_30', 'surface_merged')
        
        assert 'surface_merged' in result.columns
        assert 'surface' not in result.columns
        assert 'surface_30' not in result.columns
        assert result.loc[0, 'surface_merged'] == 'asphalt'
    
    def test_merge_similar_columns_second_column_priority(self):
        """Test that second column takes priority when first is empty"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'surface': [''],
            'surface_30': ['concrete'],
            'highway': ['residential']
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result = p2.merge_similar_columns(gdf_edges, 'surface', 'surface_30', 'surface_merged')
        
        assert result.loc[0, 'surface_merged'] == 'concrete'


class TestAddElevation:
    """Test cases for add_elevation function"""
    
    def test_add_elevation_success(self):
        """Test successful elevation addition"""
        nodes_data = {
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [1, 2]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        elevation_data = [{'idx': 0, 'elevation': 100}, {'idx': 1, 'elevation': 150}]
        
        with patch('builtins.open', mock_open(read_data=str(elevation_data))):
            with patch('os.path.isfile', return_value=True):
                result, success = p2.add_elevation(gdf_nodes, 'test.json')
                
                assert success == True
                assert 'elevation' in result.columns
                assert result.loc[0, 'elevation'] == 100
                assert result.loc[1, 'elevation'] == 150
    
    def test_add_elevation_file_not_found(self):
        """Test handling of missing elevation file"""
        nodes_data = {
            'geometry': [Point(0, 0)],
            'osmid': [1]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=False):
            result, success = p2.add_elevation(gdf_nodes, 'nonexistent.json')
            
            assert success == False
            assert 'elevation' in result.columns
            assert result.loc[0, 'elevation'] is None


class TestAddGradient:
    """Test cases for add_gradient function"""
    
    def test_add_gradient_no_elevation_file(self):
        """Test gradient calculation when elevation file is missing"""
        nodes_data = {
            'geometry': [Point(0, 0)],
            'osmid': [1]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'u': [1],
            'v': [2],
            'length': [1.414]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=False):
            result_nodes, result_edges = p2.add_gradient(gdf_nodes, gdf_edges, 'nonexistent.json')
            
            assert result_edges.loc[0, 'height_difference'] is None
            assert result_edges.loc[0, 'gradient'] is None
            assert result_edges.loc[0, 'severity'] is None

class TestAddCyclePathWidth:
    """Test cases for add_cycle_path_width function"""
    
    def test_add_cycle_path_width_success(self):
        """Test successful cycle path width addition"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        width_data = pd.DataFrame({
            'osmid': [1],
            'width': [2.5]
        })
        
        with patch('os.path.isfile', return_value=True):
            with patch('pandas.read_csv', return_value=width_data):
                result = p2.add_cycle_path_width(gdf_edges, 'test.csv')
                
                assert 'width_cycle_path' in result.columns
                assert result.loc[0, 'width_cycle_path'] == 2.5
    
    def test_add_cycle_path_width_file_not_found(self):
        """Test handling of missing cycle path width file"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=False):
            result = p2.add_cycle_path_width(gdf_edges, 'nonexistent.csv')
            
            assert 'width_cycle_path' in result.columns
            assert result.loc[0, 'width_cycle_path'] is None


class TestAddBicycleParking:
    """Test cases for add_bicycle_parking function"""
    
    def test_add_bicycle_parking_success(self):
        """Test successful bicycle parking addition"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'bike_access': ['yes'],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        parking_data = {
            'geometry': [Point(0.5, 0.5)],
            'amenity': ['bicycle_parking']
        }
        mock_parking = gpd.GeoDataFrame(parking_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=True):
            with patch('osmnetfusion.p2_enrichData.gpd.read_file', return_value=mock_parking):
                with patch('osmnetfusion.p2_enrichData.gpd.sjoin_nearest') as mock_sjoin:
                    # Mock the spatial join result
                    mock_sjoin.return_value = pd.DataFrame({
                        'index_edge': [0],
                        'distances': [10.0]
                    })
                    
                    result = p2.add_bicycle_parking(gdf_edges, 'test.gpkg')
                    
                    assert 'amenity_on' in result.columns
                    assert 'amenity_nearby' in result.columns
    
    def test_add_bicycle_parking_file_not_found(self):
        """Test handling of missing bicycle parking file"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'bike_access': ['yes'],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=False):
            result = p2.add_bicycle_parking(gdf_edges, 'nonexistent.gpkg')
            
            assert 'amenity_on' in result.columns
            assert 'amenity_nearby' in result.columns
            assert result.loc[0, 'amenity_on'] == ''
            assert result.loc[0, 'amenity_nearby'] == ''


class TestAddPtStops:
    """Test cases for add_pt_stops function"""
    
    def test_add_pt_stops_success(self):
        """Test successful public transport stops addition"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'highway': ['residential'],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        pt_data = {
            'geometry': [Point(0.5, 0.5)],
            'member_ref': ['stop1'],
            'name': ['Test Stop']
        }
        mock_pt = gpd.GeoDataFrame(pt_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=True):
            with patch('osmnetfusion.p2_enrichData.gpd.read_file', return_value=mock_pt):
                with patch('osmnetfusion.p2_enrichData.gpd.sjoin_nearest') as mock_sjoin:
                    # Mock the spatial join result
                    mock_sjoin.return_value = pd.DataFrame({
                        'index_edge': [0],
                        'name': ['Test Stop']
                    })
                    
                    result = p2.add_pt_stops(gdf_edges, 'test.gpkg')
                    
                    assert 'pt_stop_on' in result.columns
                    assert 'pt_stop_count' in result.columns
                    assert 'pt_stop_routes' in result.columns
                    assert result.loc[0, 'pt_stop_on'] == 1
    
    def test_add_pt_stops_file_not_found(self):
        """Test handling of missing public transport file"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'highway': ['residential'],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=False):
            result = p2.add_pt_stops(gdf_edges, 'nonexistent.gpkg')
            
            assert 'pt_stop_on' in result.columns
            assert 'pt_stop_count' in result.columns
            assert 'pt_stop_routes' in result.columns
            assert result.loc[0, 'pt_stop_on'] == 0


class TestUpdateIdxs:
    """Test cases for update_idxs function"""
    
    def test_update_idxs_success(self):
        """Test successful index update"""
        nodes_data = {
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [100, 200]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'u': [100],
            'v': [200],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result_nodes, result_edges = p2.update_idxs(gdf_nodes, gdf_edges)
        
        assert 'old_osmid' in result_nodes.columns
        assert 'old_osmid' in result_edges.columns
        assert result_nodes.loc[0, 'osmid'] == 0
        assert result_nodes.loc[1, 'osmid'] == 1
        assert result_edges.loc[0, 'u'] == 0
        assert result_edges.loc[0, 'v'] == 1


class TestAddMissingColumns:
    """Test cases for add_missing_columns function"""
    
    def test_add_missing_columns_success(self):
        """Test successful addition of missing columns"""
        nodes_data = {
            'geometry': [Point(0, 0)],
            'osmid': [1]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result_nodes, result_edges = p2.add_missing_columns(gdf_nodes, gdf_edges)
        
        # Check that required columns are added
        required_node_cols = ['old_osmid', 'y', 'x', 'street_count', 'highway']
        for col in required_node_cols:
            assert col in result_nodes.columns
        
        required_edge_cols = ['sidewalk', 'cycleway', 'bicycle_road', 'oneway:bicycle']
        for col in required_edge_cols:
            assert col in result_edges.columns

class TestMain:
    """Test cases for main function"""
    
    def test_main_success(self):
        """Test successful main function execution"""
        # Create mock config
        mock_config = Mock()
        mock_config.p1_result_filepath = "test_network.gpkg"
        mock_config.signals_filepath = "test_signals.gpkg"
        mock_config.cycle_path_w_filepath = "test_widths.csv"
        mock_config.bike_amenities_filepath = "test_amenities.gpkg"
        mock_config.elev_filepath = "test_elev.json"
        mock_config.green_landuse_filepath = "test_green.gpkg"
        mock_config.retail_filepath = "test_retail.gpkg"
        mock_config.building_filepath = "test_building.gpkg"
        mock_config.pt_stops_filepath = "test_pt.gpkg"
        mock_config.p2_result_filepath = "test_output.gpkg"
        
        # Mock GeoDataFrames
        mock_nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [1, 2]
        }, crs='EPSG:4326')
        
        mock_edges = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1],
            'u': [1],
            'v': [2],
            'key': [0],
            'length': [1.414],
            'highway': ['residential'],
            'bicycle': ['yes']
        }, crs='EPSG:4326')
        
        with patch('osmnetfusion.p2_enrichData.gpd.read_file') as mock_read_file:
            with patch('osmnetfusion.p2_enrichData.get_landuse_ratio') as mock_landuse:
                with patch('osmnetfusion.p2_enrichData.improve_bike_edges') as mock_improve:
                    with patch('osmnetfusion.p2_enrichData.add_cycle_paths') as mock_cycle:
                        with patch('osmnetfusion.p2_enrichData.add_gradient') as mock_gradient:
                            with patch('osmnetfusion.p2_enrichData.merge_similar_columns') as mock_merge:
                                with patch('osmnetfusion.p2_enrichData.add_traffic_lights') as mock_traffic:
                                    with patch('osmnetfusion.p2_enrichData.add_cycle_path_width') as mock_width:
                                        with patch('osmnetfusion.p2_enrichData.add_bicycle_parking') as mock_parking:
                                            with patch('osmnetfusion.p2_enrichData.add_pt_stops') as mock_pt:
                                                with patch('osmnetfusion.p2_enrichData.update_idxs') as mock_update:
                                                    with patch('osmnetfusion.p2_enrichData.add_missing_columns') as mock_missing:
                                                        with patch('osmnetfusion.p2_enrichData.save_p2_result_2_file') as mock_save:
                                                            # Mock file reading
                                                            mock_read_file.side_effect = [mock_edges, mock_nodes]
                                                            
                                                            # Mock function returns
                                                            mock_landuse.return_value = mock_edges
                                                            mock_improve.return_value = mock_edges
                                                            mock_cycle.return_value = mock_edges
                                                            mock_gradient.return_value = (mock_nodes, mock_edges)
                                                            mock_merge.return_value = mock_edges
                                                            mock_traffic.return_value = mock_nodes
                                                            mock_width.return_value = mock_edges
                                                            mock_parking.return_value = mock_edges
                                                            mock_pt.return_value = mock_edges
                                                            mock_update.return_value = (mock_nodes, mock_edges)
                                                            mock_missing.return_value = (mock_nodes, mock_edges)
                                                            
                                                            # Call main function
                                                            p2.main(mock_config)
                                                            
                                                            # Verify all functions were called
                                                            mock_read_file.assert_called()
                                                            mock_landuse.assert_called()
                                                            mock_improve.assert_called()
                                                            mock_cycle.assert_called()
                                                            mock_gradient.assert_called()
                                                            mock_merge.assert_called()
                                                            mock_traffic.assert_called()
                                                            mock_width.assert_called()
                                                            mock_parking.assert_called()
                                                            mock_pt.assert_called()
                                                            mock_update.assert_called()
                                                            mock_missing.assert_called()
                                                            mock_save.assert_called()
    
    def test_main_with_disabled_features(self):
        """Test main function with some features disabled"""
        mock_config = Mock()
        mock_config.p1_result_filepath = "test_network.gpkg"
        mock_config.signals_filepath = "test_signals.gpkg"
        mock_config.cycle_path_w_filepath = "test_widths.csv"
        mock_config.bike_amenities_filepath = "test_amenities.gpkg"
        mock_config.elev_filepath = "test_elev.json"
        mock_config.green_landuse_filepath = "test_green.gpkg"
        mock_config.retail_filepath = "test_retail.gpkg"
        mock_config.building_filepath = "test_building.gpkg"
        mock_config.pt_stops_filepath = "test_pt.gpkg"
        mock_config.p2_result_filepath = "test_output.gpkg"
        
        mock_nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0)],
            'osmid': [1]
        }, crs='EPSG:4326')
        
        mock_edges = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1],
            'u': [1],
            'v': [2],
            'key': [0],
            'length': [1.414],
            'highway': ['residential'],
            'bicycle': ['yes']
        }, crs='EPSG:4326')
        
        with patch('osmnetfusion.p2_enrichData.gpd.read_file') as mock_read_file:
            with patch('osmnetfusion.p2_enrichData.get_landuse_ratio') as mock_landuse:
                with patch('osmnetfusion.p2_enrichData.improve_bike_edges') as mock_improve:
                    with patch('osmnetfusion.p2_enrichData.add_cycle_paths') as mock_cycle:
                        with patch('osmnetfusion.p2_enrichData.add_gradient') as mock_gradient:
                            with patch('osmnetfusion.p2_enrichData.merge_similar_columns') as mock_merge:
                                with patch('osmnetfusion.p2_enrichData.add_traffic_lights') as mock_traffic:
                                    with patch('osmnetfusion.p2_enrichData.add_cycle_path_width') as mock_width:
                                        with patch('osmnetfusion.p2_enrichData.add_bicycle_parking') as mock_parking:
                                            with patch('osmnetfusion.p2_enrichData.add_pt_stops') as mock_pt:
                                                with patch('osmnetfusion.p2_enrichData.update_idxs') as mock_update:
                                                    with patch('osmnetfusion.p2_enrichData.add_missing_columns') as mock_missing:
                                                        with patch('osmnetfusion.p2_enrichData.save_p2_result_2_file') as mock_save:
                                                            # Mock file reading
                                                            mock_read_file.side_effect = [mock_edges, mock_nodes]
                                                            
                                                            # Mock function returns
                                                            mock_landuse.return_value = mock_edges
                                                            mock_improve.return_value = mock_edges
                                                            mock_cycle.return_value = mock_edges
                                                            mock_gradient.return_value = (mock_nodes, mock_edges)
                                                            mock_merge.return_value = mock_edges
                                                            mock_traffic.return_value = mock_nodes
                                                            mock_width.return_value = mock_edges
                                                            mock_parking.return_value = mock_edges
                                                            mock_pt.return_value = mock_edges
                                                            mock_update.return_value = (mock_nodes, mock_edges)
                                                            mock_missing.return_value = (mock_nodes, mock_edges)
                                                            
                                                            # Call main function with disabled features
                                                            p2.main(mock_config, public_transport=False, 
                                                                   accidents=False, cycle_path_width=False, elevation=False)
                                                            
                                                            # Verify functions were called appropriately
                                                            mock_read_file.assert_called()
                                                            mock_landuse.assert_called()
                                                            mock_improve.assert_called()
                                                            mock_cycle.assert_called()
                                                            # gradient should not be called when elevation=False
                                                            mock_gradient.assert_not_called()
                                                            mock_merge.assert_called()
                                                            mock_traffic.assert_called()
                                                            # width should not be called when cycle_path_width=False
                                                            mock_width.assert_not_called()
                                                            mock_parking.assert_called()
                                                            # pt should not be called when public_transport=False
                                                            mock_pt.assert_not_called()
                                                            mock_update.assert_called()
                                                            mock_missing.assert_called()
                                                            mock_save.assert_called()


class TestIntegration:
    """Integration tests for the module"""
    
    def test_full_workflow_integration(self):
        """Test the complete workflow integration"""
        # Create test data
        nodes_data = {
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [1, 2]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1],
            'u': [1],
            'v': [2],
            'key': [0],
            'length': [1.414],
            'highway': ['residential'],
            'bicycle': ['yes'],
            'oneway': True,
            'reversed': False
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        # Test the complete workflow with mocked external dependencies
        with patch('osmnetfusion.p2_enrichData.fiona.listlayers') as mock_listlayers:
            with patch('osmnetfusion.p2_enrichData.gpd.read_file') as mock_read_file:
                with patch('builtins.open', mock_open(read_data='[]')):
                    with patch('os.path.isfile', return_value=True):
                        # Mock landuse data
                        mock_listlayers.return_value = ['Point']
                        mock_read_file.return_value = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
                        
                        # Test landuse ratio
                        result_edges = p2.get_landuse_ratio(gdf_edges, kind='retail', input_file='test.gpkg')
                        assert 'retail_ratio' in result_edges.columns
                        
                        # Test bike edge improvement
                        result_edges = p2.improve_bike_edges(gdf_edges)
                        assert 'bike_access' in result_edges.columns
                        
                        # Test cycle path categorization
                        result_edges = p2.add_cycle_paths(gdf_edges)
                        assert 'cycleway_category' in result_edges.columns
    
    def test_error_handling_integration(self):
        """Test error handling in integration scenarios"""
        # Test with missing files
        nodes_data = {
            'geometry': [Point(0, 0)],
            'osmid': [1]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1],
            'u': [1],
            'v': [2],
            'key': [0],
            'length': [1.414],
            'highway': ['residential'],
            'bicycle': ['yes']
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        with patch('os.path.isfile', return_value=False):
            # Test elevation with missing file
            result_nodes, result_edges = p2.add_gradient(gdf_nodes, gdf_edges, 'nonexistent.json')
            assert result_edges.loc[0, 'gradient'] is None
            
            # Test cycle path width with missing file
            result_edges = p2.add_cycle_path_width(gdf_edges, 'nonexistent.csv')
            assert result_edges.loc[0, 'width_cycle_path'] is None


if __name__ == "__main__":
    pytest.main([__file__])
