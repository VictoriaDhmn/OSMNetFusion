import pytest
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from unittest.mock import patch, Mock
import osmnx as ox

import osmnetfusion.p1_getOSMNetwork as p1


class TestGetOSMGraphPlace:
    """Test cases for getOSMGraph_place function"""
    
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_from_address')
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_to_gdfs')
    def test_getOSMGraph_place_success(self, mock_graph_to_gdfs, mock_graph_from_address):
        """Test successful OSM graph retrieval by place name"""
        # Mock the graph
        mock_graph = nx.MultiDiGraph()
        mock_graph_from_address.return_value = mock_graph
        
        # Mock the GeoDataFrames
        mock_nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [1, 2]
        }, crs='EPSG:4326')
        mock_edges = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1]
        }, crs='EPSG:4326')
        mock_graph_to_gdfs.return_value = (mock_nodes, mock_edges)
        
        # Test parameters
        location = "Munich, Germany"
        distance = 1500
        tags = ['cycleway', 'cycleway:left']
        
        # Call function
        graph, gdf_nodes, gdf_edges = p1.getOSMGraph_place(location, distance, tags)
        
        # Assertions
        assert graph == mock_graph
        assert gdf_nodes.equals(mock_nodes)
        assert gdf_edges.equals(mock_edges)
        
        # Check that ox.settings were set correctly
        assert ox.settings.useful_tags_way == tags
        assert ox.settings.useful_tags_node == tags
        
        # Check that graph_from_address was called with correct parameters
        mock_graph_from_address.assert_called_once_with(
            location, dist=distance, dist_type='bbox',
            network_type='all', simplify=True, retain_all=False, 
            truncate_by_edge=False, custom_filter=None
        )
    
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_from_address')
    def test_getOSMGraph_place_invalid_location(self, mock_graph_from_address):
        """Test handling of invalid location"""
        mock_graph_from_address.side_effect = ValueError("Location not found")
        
        with pytest.raises(ValueError, match="Location not found"):
            p1.getOSMGraph_place("Invalid Location", 1000, ['cycleway'])
    
    def test_getOSMGraph_place_invalid_distance(self):
        """Test handling of invalid distance parameter"""
        with pytest.raises(TypeError):
            p1.getOSMGraph_place("Munich, Germany", "invalid", ['cycleway'])
    
    def test_getOSMGraph_place_empty_tags(self):
        """Test handling of empty tags list"""
        with patch('osmnetfusion.p1_getOSMNetwork.ox.graph_from_address') as mock_graph_from_address:
            mock_graph = nx.MultiDiGraph()
            mock_graph_from_address.return_value = mock_graph
            
            with patch('osmnetfusion.p1_getOSMNetwork.ox.graph_to_gdfs') as mock_graph_to_gdfs:
                mock_nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs='EPSG:4326')
                mock_edges = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
                mock_graph_to_gdfs.return_value = (mock_nodes, mock_edges)
                
                graph, gdf_nodes, gdf_edges = p1.getOSMGraph_place("Munich, Germany", 1000, [])
                
                assert ox.settings.useful_tags_way == []
                assert ox.settings.useful_tags_node == []


class TestGetOSMGraphCoords:
    """Test cases for getOSMGraph_coords function"""
    
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_from_bbox')
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_to_gdfs')
    def test_getOSMGraph_coords_success(self, mock_graph_to_gdfs, mock_graph_from_bbox):
        """Test successful OSM graph retrieval by coordinates"""
        # Mock the graph
        mock_graph = nx.MultiDiGraph()
        mock_graph_from_bbox.return_value = mock_graph
        
        # Mock the GeoDataFrames
        mock_nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [1, 2]
        }, crs='EPSG:4326')
        mock_edges = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1]
        }, crs='EPSG:4326')
        mock_graph_to_gdfs.return_value = (mock_nodes, mock_edges)
        
        # Test parameters
        coords_upper_left = (48.137, 11.575)  # Munich coordinates
        coords_lower_right = (48.135, 11.577)
        tags = ['cycleway', 'cycleway:left']
        
        # Call function
        graph, gdf_nodes, gdf_edges = p1.getOSMGraph_coords(coords_upper_left, coords_lower_right, tags)
        
        # Assertions
        assert graph == mock_graph
        assert gdf_nodes.equals(mock_nodes)
        assert gdf_edges.equals(mock_edges)
        
        # Check that ox.settings were set correctly
        assert ox.settings.useful_tags_way == tags
        assert ox.settings.useful_tags_node == tags
        
        # Check that graph_from_bbox was called with correct parameters
        mock_graph_from_bbox.assert_called_once_with(
            west=coords_upper_left[1], south=coords_lower_right[0], 
            east=coords_lower_right[1], north=coords_upper_left[0],
            network_type='all', simplify=True, retain_all=False, 
            truncate_by_edge=False, custom_filter=None
        )
    
    def test_getOSMGraph_coords_swapped_coordinates(self):
        """Test handling of coordinates where upper_left is actually lower than lower_right"""
        with patch('osmnetfusion.p1_getOSMNetwork.ox.graph_from_bbox') as mock_graph_from_bbox:
            mock_graph = nx.MultiDiGraph()
            mock_graph_from_bbox.return_value = mock_graph
            
            with patch('osmnetfusion.p1_getOSMNetwork.ox.graph_to_gdfs') as mock_graph_to_gdfs:
                mock_nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs='EPSG:4326')
                mock_edges = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')
                mock_graph_to_gdfs.return_value = (mock_nodes, mock_edges)
                
                coords_upper_left = (48.135, 11.577)  # Actually lower right
                coords_lower_right = (48.137, 11.575)  # Actually upper left
                
                graph, gdf_nodes, gdf_edges = p1.getOSMGraph_coords(coords_upper_left, coords_lower_right, ['cycleway'])
                
                mock_graph_from_bbox.assert_called_once()

class TestMain:
    """Test cases for main function"""
    
    def test_main_place_mode(self):
        """Test main function with place mode"""
        # Create mock config
        mock_config = Mock()
        mock_config.version = "1.0"
        mock_config.boundary_mode = "place"
        mock_config.custom_OSM_date = None
        mock_config.location = "Munich, Germany"
        mock_config.dist_in_meters = 1500
        mock_config.used_tags = ['cycleway', 'cycleway:left']
        mock_config.p1_result_filepath = "test_output.gpkg"
        
        with patch('osmnetfusion.p1_getOSMNetwork.getOSMGraph_place') as mock_get_place:
            with patch('osmnetfusion.p1_getOSMNetwork.make_plot') as mock_plot:
                with patch('osmnetfusion.p1_getOSMNetwork.ox.save_graph_geopackage') as mock_save:
                    # Mock return values
                    mock_graph = nx.MultiDiGraph()
                    mock_nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs='EPSG:4326')
                    mock_edges = gpd.GeoDataFrame({'geometry': [LineString([(0, 0), (1, 1)])]}, crs='EPSG:4326')
                    mock_get_place.return_value = (mock_graph, mock_nodes, mock_edges)
                    
                    # Call main function
                    p1.main(mock_config)
                    
                    # Check that functions were called correctly
                    mock_get_place.assert_called_once_with(
                        location="Munich, Germany", 
                        distance=1500, 
                        tags=['cycleway', 'cycleway:left']
                    )
                    mock_plot.assert_called_once()
                    mock_save.assert_called_once_with(
                        mock_graph, directed=True, filepath="test_output.gpkg"
                    )
    
    def test_main_coords_mode(self):
        """Test main function with coordinates mode"""
        # Create mock config
        mock_config = Mock()
        mock_config.version = "1.0"
        mock_config.boundary_mode = "coords"
        mock_config.custom_OSM_date = None
        mock_config.coords_upper_left = (48.137, 11.575)
        mock_config.coords_lower_right = (48.135, 11.577)
        mock_config.used_tags = ['cycleway', 'cycleway:left']
        mock_config.p1_result_filepath = "test_output.gpkg"
        
        with patch('osmnetfusion.p1_getOSMNetwork.getOSMGraph_coords') as mock_get_coords:
            with patch('osmnetfusion.p1_getOSMNetwork.make_plot') as mock_plot:
                with patch('osmnetfusion.p1_getOSMNetwork.ox.save_graph_geopackage') as mock_save:
                    # Mock return values
                    mock_graph = nx.MultiDiGraph()
                    mock_nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs='EPSG:4326')
                    mock_edges = gpd.GeoDataFrame({'geometry': [LineString([(0, 0), (1, 1)])]}, crs='EPSG:4326')
                    mock_get_coords.return_value = (mock_graph, mock_nodes, mock_edges)
                    
                    # Call main function
                    p1.main(mock_config)
                    
                    # Check that functions were called correctly
                    mock_get_coords.assert_called_once_with(
                        (48.137, 11.575), (48.135, 11.577), ['cycleway', 'cycleway:left']
                    )
                    mock_plot.assert_called_once()
                    mock_save.assert_called_once()

class TestIntegration:
    """Integration tests for the module"""
    
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_from_address')
    @patch('osmnetfusion.p1_getOSMNetwork.ox.graph_to_gdfs')
    def test_full_workflow_place_mode(self, mock_graph_to_gdfs, mock_graph_from_address):
        """Test the complete workflow for place mode"""
        # Mock the graph and GeoDataFrames
        mock_graph = nx.MultiDiGraph()
        mock_graph_from_address.return_value = mock_graph
        
        mock_nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1)],
            'osmid': [1, 2]
        }, crs='EPSG:4326')
        mock_edges = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])],
            'osmid': [1]
        }, crs='EPSG:4326')
        mock_graph_to_gdfs.return_value = (mock_nodes, mock_edges)
        
        # Test the complete workflow
        location = "Munich, Germany"
        distance = 1500
        tags = ['cycleway', 'cycleway:left']
        
        graph, gdf_nodes, gdf_edges = p1.getOSMGraph_place(location, distance, tags)
        
        # Verify the results
        assert isinstance(graph, nx.MultiDiGraph)
        assert isinstance(gdf_nodes, gpd.GeoDataFrame)
        assert isinstance(gdf_edges, gpd.GeoDataFrame)
        assert len(gdf_nodes) == 2
        assert len(gdf_edges) == 1
    
    def test_settings_restoration(self):
        """Test that OSM settings are properly restored after main function"""
        original_settings = ox.settings.overpass_settings
        
        mock_config = Mock()
        mock_config.version = "1.0"
        mock_config.boundary_mode = "place"
        mock_config.custom_OSM_date = "2023-01-01"
        mock_config.location = "Munich, Germany"
        mock_config.dist_in_meters = 1500
        mock_config.used_tags = ['cycleway']
        mock_config.p1_result_filepath = "test_output.gpkg"
        
        with patch('osmnetfusion.p1_getOSMNetwork.getOSMGraph_place') as mock_get_place:
            with patch('osmnetfusion.p1_getOSMNetwork.make_plot'):
                with patch('osmnetfusion.p1_getOSMNetwork.ox.save_graph_geopackage'):
                    mock_graph = nx.MultiDiGraph()
                    mock_nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0)]}, crs='EPSG:4326')
                    mock_edges = gpd.GeoDataFrame({'geometry': [LineString([(0, 0), (1, 1)])]}, crs='EPSG:4326')
                    mock_get_place.return_value = (mock_graph, mock_nodes, mock_edges)
                    
                    p1.main(mock_config)
        
        assert ox.settings.overpass_settings == original_settings


if __name__ == "__main__":
    pytest.main([__file__]) 