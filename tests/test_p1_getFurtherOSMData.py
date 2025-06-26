import pytest
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, box
from unittest.mock import patch, Mock, mock_open
import osmnx as ox
import overpy
import os
import tempfile

import osmnetfusion.p1_getFurtherOSMData as p1


class TestGetCityBoundary:
    """Test cases for get_city_boundary function"""
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.geocode_to_gdf')
    def test_get_city_boundary_success(self, mock_geocode_to_gdf):
        """Test successful city boundary retrieval"""
        # Mock GeoDataFrame with boundary data
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [box(11.5, 48.1, 11.6, 48.2)],
            'name': ['Munich']
        }, crs='EPSG:4326')
        mock_geocode_to_gdf.return_value = mock_gdf
        
        place = "Munich, Germany"
        boundary = p1.get_city_boundary(place)
        
        # Check that geocode_to_gdf was called correctly
        mock_geocode_to_gdf.assert_called_once_with(place)
        
        # Check that boundary is a shapely geometry
        assert hasattr(boundary, 'bounds')
        assert boundary.bounds == (11.5, 48.1, 11.6, 48.2)
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.geocode_to_gdf')
    def test_get_city_boundary_multiple_boundaries(self, mock_geocode_to_gdf):
        """Test handling when multiple boundaries are found"""
        # Mock GeoDataFrame with multiple boundaries
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [box(11.5, 48.1, 11.6, 48.2), box(11.6, 48.2, 11.7, 48.3)],
            'name': ['Munich', 'Munich District']
        }, crs='EPSG:4326')
        mock_geocode_to_gdf.return_value = mock_gdf
        
        place = "Munich, Germany"
        boundary = p1.get_city_boundary(place)
        
        assert hasattr(boundary, 'bounds')
        assert boundary.bounds == (11.5, 48.1, 11.7, 48.3)


class TestGeneratePTStopsAndRouteFile:
    """Test cases for generate_pt_stops_and_route_file function"""
    
    @patch('osmnetfusion.p1_getFurtherOSMData.get_city_boundary')
    @patch('osmnetfusion.p1_getFurtherOSMData.overpy.Overpass')
    def test_generate_pt_stops_and_route_file_success(self, mock_overpass, mock_get_boundary):
        """Test successful PT stops and route file generation"""
        # Mock boundary
        mock_boundary = box(11.5, 48.1, 11.6, 48.2)
        mock_get_boundary.return_value = mock_boundary
        
        # Mock overpy result
        mock_api = Mock()
        mock_overpass.return_value = mock_api
        
        # Create mock relation and members
        mock_relation = Mock()
        mock_relation.id = 123
        mock_relation.tags = {
            'route': 'bus',
            'name': 'Bus Line 1',
            'operator': 'MVG'
        }
        
        mock_node = Mock()
        mock_node.lon = 11.55
        mock_node.lat = 48.15
        
        mock_member = Mock()
        mock_member.ref = 456
        mock_member.role = 'stop'
        mock_member.__class__ = overpy.RelationNode
        
        mock_relation.members = [mock_member]
        
        mock_result = Mock()
        mock_result.relations = [mock_relation]
        mock_result.get_node.return_value = mock_node
        mock_api.query.return_value = mock_result
        
        # Mock matplotlib to avoid actual plotting
        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_fig, mock_ax = Mock(), Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                
                # Mock contextily
                with patch('contextily.add_basemap'):
                    # Mock file operations
                    with patch('geopandas.GeoDataFrame.to_file'):
                        p1.generate_pt_stops_and_route_file(
                            place="Munich, Germany",
                            manual_query=None,
                            output_file_stops="test_stops.gpkg"
                        )
        
        # Check that overpass query was called
        mock_api.query.assert_called()
    
    @patch('osmnetfusion.p1_getFurtherOSMData.get_city_boundary')
    def test_generate_pt_stops_and_route_file_with_manual_query(self, mock_get_boundary):
        """Test PT stops generation with manual query file"""
        # Mock boundary
        mock_boundary = box(11.5, 48.1, 11.6, 48.2)
        mock_get_boundary.return_value = mock_boundary
        
        # Mock osmium handler
        with patch('osmium.SimpleHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            
            # Mock matplotlib to avoid actual plotting
            with patch('matplotlib.pyplot.show'):
                with patch('matplotlib.pyplot.subplots') as mock_subplots:
                    mock_fig, mock_ax = Mock(), Mock()
                    mock_subplots.return_value = (mock_fig, mock_ax)
                    
                    # Mock contextily
                    with patch('contextily.add_basemap'):
                        # Mock file operations
                        with patch('geopandas.GeoDataFrame.to_file'):
                            p1.generate_pt_stops_and_route_file(
                                place="Munich, Germany",
                                manual_query="test_query.pbf",
                                output_file_stops="test_stops.gpkg"
                            )
            
class TestGenerateLanduseFile:
    """Test cases for generate_landuse_file function"""
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.features_from_place')
    def test_generate_landuse_file_success(self, mock_features_from_place):
        """Test successful landuse file generation"""
        # Mock GeoDataFrame
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [box(11.5, 48.1, 11.6, 48.2)],
            'landuse': ['residential'],
            'name': ['Residential Area']
        }, crs='EPSG:4326')
        mock_features_from_place.return_value = mock_gdf
        
        place = "Munich, Germany"
        tags = {"landuse": "residential"}
        cols = ["name", "landuse", "geometry"]
        output_file = "test_landuse.gpkg"
        
        # Mock file operations
        with patch('geopandas.GeoDataFrame.to_file'):
            p1.generate_landuse_file(place, output_file, tags, cols)
        
        # Check that features_from_place was called correctly
        mock_features_from_place.assert_called_once_with(place, tags)
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.features_from_place')
    def test_generate_landuse_file_missing_columns(self, mock_features_from_place):
        """Test handling when requested columns don't exist"""
        # Mock GeoDataFrame with only some columns
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [box(11.5, 48.1, 11.6, 48.2)],
            'landuse': ['residential']
        }, crs='EPSG:4326')
        mock_features_from_place.return_value = mock_gdf
        
        place = "Munich, Germany"
        tags = {"landuse": "residential"}
        cols = ["name", "landuse", "geometry", "nonexistent_column"]
        output_file = "test_landuse.gpkg"
        
        # Mock file operations
        with patch('geopandas.GeoDataFrame.to_file'):
            p1.generate_landuse_file(place, output_file, tags, cols)
        
        # Should only include existing columns
        assert mock_gdf.columns.tolist() == ['geometry', 'landuse']


class TestGenerateObjectsFile:
    """Test cases for generate_objects_file function"""
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.features_from_place')
    def test_generate_objects_file_success(self, mock_features_from_place):
        """Test successful objects file generation"""
        # Mock GeoDataFrame
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [Point(11.55, 48.15)],
            'amenity': ['bicycle_parking'],
            'name': ['Bike Parking 1']
        }, crs='EPSG:4326')
        mock_features_from_place.return_value = mock_gdf
        
        place = "Munich, Germany"
        output_file = "test_amenities.gpkg"
        tags = {"amenity": "bicycle_parking"}
        cols = ["name", "amenity", "geometry"]
        objects = "bike_amenities"
        
        # Mock file operations
        with patch('geopandas.GeoDataFrame.to_file'):
            p1.generate_objects_file(place, output_file, tags, cols, objects)
        
        # Check that features_from_place was called correctly
        mock_features_from_place.assert_called_once_with(place, tags)
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.features_from_place')
    def test_generate_objects_file_no_cols_specified(self, mock_features_from_place):
        """Test objects file generation without specifying columns"""
        # Mock GeoDataFrame
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [Point(11.55, 48.15)],
            'amenity': ['bicycle_parking'],
            'name': ['Bike Parking 1'],
            'capacity': ['10']
        }, crs='EPSG:4326')
        mock_features_from_place.return_value = mock_gdf
        
        place = "Munich, Germany"
        output_file = "test_amenities.gpkg"
        tags = {"amenity": "bicycle_parking"}
        
        # Mock file operations
        with patch('geopandas.GeoDataFrame.to_file'):
            p1.generate_objects_file(place, output_file, tags)
        
        # Should keep all columns when cols=None
        assert len(mock_gdf.columns) == 4


class TestMain:
    """Test cases for main function"""
    
    def test_main_all_features_enabled(self):
        """Test main function with all features enabled"""
        # Create mock config
        mock_config = Mock()
        mock_config.city_info = {'city_OSM': 'Munich, Germany'}
        mock_config.place = 'Munich, Germany'
        mock_config.custom_OSM_date = None
        mock_config.pt_stops_tags = {'route': 'bus'}
        mock_config.pt_stops_cols = ['name', 'route']
        mock_config.pt_stops_filepath = "test_output/pt_stops.gpkg"
        mock_config.manual_OSM_PT_query = False
        mock_config.manual_OSM_query_fp = None
        mock_config.bike_amenities_tags = {'amenity': 'bicycle_parking'}
        mock_config.bike_amenities_cols = ['name', 'amenity']
        mock_config.bike_amenities_filepath = "test_output/amenities.gpkg"
        mock_config.building_tags = {'building': 'yes'}
        mock_config.building_cols = ['name', 'building']
        mock_config.building_filepath = "test_output/buildings.gpkg"
        mock_config.green_landuse_tags = {'landuse': 'grass'}
        mock_config.green_landuse_cols = ['name', 'landuse']
        mock_config.green_landuse_filepath = "test_output/green_landuse.gpkg"
        mock_config.retail_tags = {'shop': 'supermarket'}
        mock_config.retail_cols = ['name', 'shop']
        mock_config.retail_filepath = "test_output/retail.gpkg"
        mock_config.signals_tags = {'highway': 'traffic_signals'}
        mock_config.signals_cols = ['name', 'highway']
        mock_config.signals_filepath = "test_output/signals.gpkg"
        
        # Mock all the functions
        with patch('osmnetfusion.p1_getFurtherOSMData.generate_pt_stops_and_route_file') as mock_pt:
            with patch('osmnetfusion.p1_getFurtherOSMData.generate_objects_file') as mock_objects:
                with patch('osmnetfusion.p1_getFurtherOSMData.generate_landuse_file') as mock_landuse:
                    with patch('os.makedirs'):
                        # Call main function
                        p1.main(mock_config)
                        
                        # Check that all functions were called
                        mock_pt.assert_called_once()
                        assert mock_objects.call_count == 2  # amenities and signals
                        assert mock_landuse.call_count == 3  # buildings, green_landuse, retail
    
    def test_main_selective_features(self):
        """Test main function with selective features enabled"""
        # Create mock config
        mock_config = Mock()
        mock_config.city_info = {'city_OSM': 'Munich, Germany'}
        mock_config.place = 'Munich, Germany'
        mock_config.custom_OSM_date = None
        mock_config.pt_stops_filepath = "test_output/pt_stops.gpkg"
        mock_config.bike_amenities_filepath = "test_output/amenities.gpkg"
        mock_config.building_filepath = "test_output/buildings.gpkg"
        mock_config.green_landuse_filepath = "test_output/green_landuse.gpkg"
        mock_config.retail_filepath = "test_output/retail.gpkg"
        mock_config.signals_filepath = "test_output/signals.gpkg"
        
        # Mock all the functions
        with patch('osmnetfusion.p1_getFurtherOSMData.generate_pt_stops_and_route_file') as mock_pt:
            with patch('osmnetfusion.p1_getFurtherOSMData.generate_objects_file') as mock_objects:
                with patch('osmnetfusion.p1_getFurtherOSMData.generate_landuse_file') as mock_landuse:
                    with patch('os.makedirs'):
                        # Call main function with only PT stops and amenities
                        p1.main(mock_config, ptstops=True, amenities=True, 
                               buildings=False, landuse=False, retail=False, signals=False)
                        
                        # Check that only selected functions were called
                        mock_pt.assert_called_once()
                        mock_objects.assert_called_once()  # only amenities
                        mock_landuse.assert_not_called()
    
    def test_main_with_manual_osm_query(self):
        """Test main function with manual OSM query"""
        # Create mock config
        mock_config = Mock()
        mock_config.city_info = {'city_OSM': 'Munich, Germany'}
        mock_config.place = 'Munich, Germany'
        mock_config.custom_OSM_date = None
        mock_config.manual_OSM_PT_query = True
        mock_config.manual_OSM_query_fp = "manual_query.pbf"
        mock_config.pt_stops_filepath = "test_output/pt_stops.gpkg"
        mock_config.bike_amenities_filepath = "test_output/amenities.gpkg"
        mock_config.building_filepath = "test_output/buildings.gpkg"
        mock_config.green_landuse_filepath = "test_output/green_landuse.gpkg"
        mock_config.retail_filepath = "test_output/retail.gpkg"
        mock_config.signals_filepath = "test_output/signals.gpkg"
        
        # Mock all the functions
        with patch('osmnetfusion.p1_getFurtherOSMData.generate_pt_stops_and_route_file') as mock_pt:
            with patch('osmnetfusion.p1_getFurtherOSMData.generate_objects_file'):
                with patch('osmnetfusion.p1_getFurtherOSMData.generate_landuse_file'):
                    with patch('os.makedirs'):
                        # Call main function
                        p1.main(mock_config)
                        
                        # Check that PT function was called with manual query
                        mock_pt.assert_called_once_with(
                            place='Munich, Germany',
                            manual_query="manual_query.pbf",
                            output_file_stops="test_output/pt_stops.gpkg"
                        )

class TestIntegration:
    """Integration tests for the module"""
    
    @patch('osmnetfusion.p1_getFurtherOSMData.ox.geocode_to_gdf')
    def test_full_workflow_city_boundary(self, mock_geocode_to_gdf):
        """Test the complete workflow for city boundary retrieval"""
        # Mock GeoDataFrame with boundary data
        mock_gdf = gpd.GeoDataFrame({
            'geometry': [box(11.5, 48.1, 11.6, 48.2)],
            'name': ['Munich']
        }, crs='EPSG:4326')
        mock_geocode_to_gdf.return_value = mock_gdf
        
        place = "Munich, Germany"
        boundary = p1.get_city_boundary(place)
        
        # Verify the results
        assert isinstance(boundary, Polygon)
        assert boundary.bounds == (11.5, 48.1, 11.6, 48.2)
    
    def test_settings_restoration(self):
        """Test that OSM settings are properly restored after main function"""
        original_settings = ox.settings.overpass_settings
        
        mock_config = Mock()
        mock_config.city_info = {'city_OSM': 'Munich, Germany'}
        mock_config.place = 'Munich, Germany'
        mock_config.custom_OSM_date = "2023-01-01"
        mock_config.pt_stops_filepath = "test_output/pt_stops.gpkg"
        mock_config.bike_amenities_filepath = "test_output/amenities.gpkg"
        mock_config.building_filepath = "test_output/buildings.gpkg"
        mock_config.green_landuse_filepath = "test_output/green_landuse.gpkg"
        mock_config.retail_filepath = "test_output/retail.gpkg"
        mock_config.signals_filepath = "test_output/signals.gpkg"
        
        with patch('osmnetfusion.p1_getFurtherOSMData.generate_pt_stops_and_route_file'):
            with patch('osmnetfusion.p1_getFurtherOSMData.generate_objects_file'):
                with patch('osmnetfusion.p1_getFurtherOSMData.generate_landuse_file'):
                    with patch('os.makedirs'):
                        p1.main(mock_config)
        
        # Check that settings were restored
        assert ox.settings.overpass_settings == original_settings


if __name__ == "__main__":
    pytest.main([__file__])
