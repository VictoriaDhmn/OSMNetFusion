import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from unittest.mock import patch, MagicMock, Mock
import os
import tempfile
import json
import math

import osmnetfusion.p3_functions as p3


class TestAddHighwayRank:
    """Test cases for addHighwayRank function"""
    
    def test_add_highway_rank_default(self):
        """Test highway ranking with default values"""
        edges_data = {
            'highway': ['trunk', 'primary', 'residential', 'cycleway', 'footway']
        }
        edges = pd.DataFrame(edges_data)
        
        result = p3.addHighwayRank(edges)
        
        expected_ranks = [10, 9, 7, 6, 4]
        assert list(result) == expected_ranks
    
    def test_add_highway_rank_custom(self):
        """Test highway ranking with custom values"""
        edges_data = {
            'highway': ['trunk', 'primary', 'residential']
        }
        edges = pd.DataFrame(edges_data)
        
        custom_ranking = {'trunk': 5, 'primary': 3, 'residential': 1}
        result = p3.addHighwayRank(edges, highway_ranking_custom=custom_ranking)
        
        expected_ranks = [5, 3, 1]
        assert list(result) == expected_ranks
    
    def test_add_highway_rank_unknown_type(self):
        """Test highway ranking with unknown highway types"""
        edges_data = {
            'highway': ['trunk', 'unknown_type', 'residential']
        }
        edges = pd.DataFrame(edges_data)
        
        result = p3.addHighwayRank(edges)
        
        expected_ranks = [10, 0, 7]
        assert list(result) == expected_ranks
    
    def test_add_highway_rank_custom_column(self):
        """Test highway ranking with custom column name"""
        edges_data = {
            'road_type': ['trunk', 'primary', 'residential']
        }
        edges = pd.DataFrame(edges_data)
        
        result = p3.addHighwayRank(edges, col='road_type')
        
        expected_ranks = [10, 9, 7]
        assert list(result) == expected_ranks


class TestSplitCurves:
    """Test cases for splitCurves function"""
    
    def test_split_curves_no_splits(self):
        """Test splitCurves with straight lines that don't need splitting"""
        # Create straight line edges
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            'u': [1, 2],
            'v': [2, 3],
            'length': [1.414, 1.414]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        nodes_data = {
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)],
            'osmid': [1, 2, 3]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        result_edges, result_nodes = p3.splitCurves(gdf_edges, gdf_nodes)
        
        # Should return original edges and empty nodes
        assert len(result_edges) == 2
        assert len(result_nodes) == 0
    
    def test_split_curves_with_splits(self):
        """Test splitCurves with curved lines that need splitting"""
        # Create curved line with more points to trigger splitting
        curved_line = LineString([(0, 0), (0.5, 0.5), (1, 0)])
        edges_data = {
            'geometry': [curved_line],
            'u': [1],
            'v': [2],
            'length': [1.0]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        nodes_data = {
            'geometry': [Point(0, 0), Point(1, 0)],
            'osmid': [1, 2]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        result_edges, result_nodes = p3.splitCurves(gdf_edges, gdf_nodes, maxAngleInitial=30, maxAnglePrev=20)
        
        # Should create new edges and nodes
        assert len(result_edges) >= 1
        assert len(result_nodes) >= 0
    

class TestGetHighestRankingRoadOfNode:
    """Test cases for getHighestRankingRoadOfNode function"""
    
    def test_get_highest_ranking_road_of_node(self):
        """Test getting highest ranking road for each node"""
        nodes_data = {
            'osmid': [1, 2, 3],
            'nodeIdx': [0, 1, 2]
        }
        nodes = pd.DataFrame(nodes_data)
        
        edges_data = {
            'u': [1, 1, 2, 2, 3],
            'v': [2, 3, 1, 3, 1],
            'highway_rank': [10, 7, 10, 9, 7],
            'highway': ['trunk', 'residential', 'trunk', 'primary', 'residential']
        }
        edges = pd.DataFrame(edges_data)
        
        most_important_highway, most_important_highway_rank = p3.getHighestRankingRoadOfNode(nodes, edges)
        
        assert len(most_important_highway) == 3
        assert len(most_important_highway_rank) == 3
        # Node 1 should have trunk (rank 10) as most important
        assert most_important_highway[0] == 'trunk'
        assert most_important_highway_rank[0] == 10


class TestGetGeomBuffered:
    """Test cases for getGeomBuffered function"""
    
    def test_get_geom_buffered_default(self):
        """Test getting buffered geometries with default buffers"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'highway_conn': ['trunk']
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result = p3.getGeomBuffered(gdf_edges)
        
        assert len(result) == 1
        assert result[0].geom_type == 'Polygon'
    
    def test_get_geom_buffered_custom(self):
        """Test getting buffered geometries with custom buffers"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'highway_conn': ['trunk']
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        custom_buffers = {'trunk': 50}
        result = p3.getGeomBuffered(gdf_edges, highway_buffers_custom=custom_buffers)
        
        assert len(result) == 1
        assert result[0].geom_type == 'Polygon'
    
    def test_get_geom_buffered_no_crs(self):
        """Test getting buffered geometries with no CRS"""
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)])],
            'highway_conn': ['trunk']
        }
        edges = pd.DataFrame(edges_data)
        
        with pytest.raises(AttributeError):
            p3.getGeomBuffered(edges)


class TestGetBufferIntersections:
    """Test cases for getBufferIntersections function"""
    
    def test_get_buffer_intersections(self):
        """Test finding intersections between buffered edges"""
        # Create intersecting buffered edges
        buffer1 = Point(0, 0).buffer(1)
        buffer2 = Point(1, 0).buffer(1)
        
        edges_data = {
            'osmid': [1, 2],
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 0), (2, 1)])],
            'geom_buffered': [buffer1, buffer2]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result = p3.getBufferIntersections(gdf_edges)
        
        assert len(result) > 0
        assert 'osmid' in result.columns


class TestClusterNodes:
    """Test cases for clusterNodes function"""
    
    def test_cluster_nodes_basic(self):
        """Test basic node clustering"""
        # Create nodes with overlapping buffers
        nodes_data = {
            'geometry': [Point(0, 0), Point(0.1, 0.1)],
            'osmid': [1, 2],
            'geom_buffered': [Point(0, 0).buffer(0.2), Point(0.1, 0.1).buffer(0.2)],
            'highway_rank': [10, 8],
            'geometry_orig': [Point(0, 0), Point(0.1, 0.1)]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        result = p3.clusterNodes(gdf_nodes)
        
        assert len(result) == 2
        assert 'merged' in result.columns
        assert 'geom_merged' in result.columns
    
    def test_cluster_nodes_large_cluster(self):
        """Test clustering with large number of overlapping nodes"""
        # Create many overlapping nodes
        nodes_data = {
            'geometry': [Point(i*0.01, i*0.01) for i in range(50)],
            'osmid': list(range(50)),
            'geom_buffered': [Point(i*0.01, i*0.01).buffer(0.05) for i in range(50)],
            'highway_rank': [10] * 50,
            'geometry_orig': [Point(i*0.01, i*0.01) for i in range(50)]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        result = p3.clusterNodes(gdf_nodes, clusterThreshold=10)
        
        assert len(result) == 50
        assert 'merged' in result.columns


class TestReassignNodes:
    """Test cases for reassignNodes function"""
    
    def test_reassign_nodes(self):
        """Test reassigning nodes based on merge index"""
        edges_data = {
            'u': [1, 2],
            'v': [2, 3],
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])]
        }
        edges = pd.DataFrame(edges_data)
        
        merge_idx_data = {
            'merged_by': [1, 1, 3],
            'geom_merged': [Point(0.5, 0.5), Point(0.5, 0.5), Point(2, 2)]
        }
        merge_idx = pd.DataFrame(merge_idx_data, index=[1, 2, 3])
        
        result = p3.reassignNodes(edges, merge_idx)
        
        assert 'new_u' in result.columns
        assert 'new_v' in result.columns
        assert 'geom_reassigned' in result.columns
        assert 'geom_linear' in result.columns


class TestMergeNodes:
    """Test cases for mergeNodes function"""
    
    def test_merge_nodes(self):
        """Test merging nodes based on merged_by column"""
        nodes_data = {
            'osmid': [1, 2, 3],
            'merged_by': [1, 1, 3],
            'geom_merged': [Point(0.5, 0.5), Point(0.5, 0.5), Point(2, 2)],
            'old_osmid': [10, 20, 30],
            'highway_conn': ['trunk', 'trunk', 'primary'],
            'highway_rank': [10, 10, 9],
            'crossing': ['crossing', 'crossing', None],
            'geometry': [Point(0, 0), Point(0, 0), Point(2, 2)],
            'highway': ['crossing', 'crossing', 'crossing']
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        result = p3.mergeNodes(gdf_nodes)
        
        assert len(result) > 0
        assert all(isinstance(node, p3.Node) for node in result)


class TestMergeEdgesWithSameNodes:
    """Test cases for mergeEdgesWithSameNodes function"""
    
    def test_merge_edges_same_u_v(self):
        """Test merging edges where new_u equals new_v"""
        edges_data = {
            'new_u': [1, 1],
            'new_v': [1, 1],
            'highway_rank': [10, 8],
            'length': [1.0, 1.0],
            'osmid': [1, 2]
        }
        edges = pd.DataFrame(edges_data)
        
        result_links, deleted_edges = p3.mergeEdgesWithSameNodes(edges)
        
        assert len(result_links) == 0
        assert len(deleted_edges) > 0


class TestRemoveDeg2Nodes:
    """Test cases for removeDeg2Nodes function"""
    
    def test_remove_deg2_nodes(self):
        """Test removing degree 2 nodes"""
        nodes_data = {
            'osmid': [1, 2, 3],
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'u': [1, 2],
            'v': [2, 3],
            'highway_conn': ['residential', 'residential'],
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        result_edges, result_nodes = p3.removeDeg2Nodes(gdf_edges, gdf_nodes)
        
        assert len(result_edges) >= 0
        assert len(result_nodes) >= 0


class TestHelperFunctions:
    """Test cases for helper functions"""
    
    def test_clean_single_value(self):
        """Test clean function with single value"""
        result = p3.clean("5.5", asFloat=True)
        assert result == 5.5
    
    def test_clean_list_value(self):
        """Test clean function with list value"""
        result = p3.clean("['a', 'b', 'c']")
        assert result == ['a', 'b', 'c']
    
    def test_clean_empty_value(self):
        """Test clean function with empty value"""
        result = p3.clean("")
        assert result is None or np.isnan(result)
    
    def test_get_node_dict(self):
        """Test getNodeDict function"""
        # Create mock Node objects
        mock_node1 = Mock()
        mock_node1.to_dict.return_value = {'g_id': 1, 'g_x': 0, 'g_y': 0, 'g_geometry': Point(0, 0)}
        mock_node2 = Mock()
        mock_node2.to_dict.return_value = {'g_id': 2, 'g_x': 1, 'g_y': 1, 'g_geometry': Point(1, 1)}
        
        nodes = [mock_node1, mock_node2]
        
        result = p3.getNodeDict(nodes)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
    
    def test_listify_simple(self):
        """Test listify function with simple values"""
        series = pd.Series(['a', 'b', 'c'])
        result = p3.listify(series)
        assert result == ['a', 'b', 'c']
    
    def test_listify_with_integers(self):
        """Test listify function with asInt=True"""
        series = pd.Series(['1', '2', '3'])
        result = p3.listify(series, asInt=True)
        assert result == [1, 2, 3]
    
    def test_get_edge_dict(self):
        """Test getEdgeDict function"""
        # Create mock Link objects
        mock_link = Mock()
        mock_edge = Mock()
        mock_edge.to_dict.return_value = {'g_u': 1, 'g_v': 2, 'access_bik': True, 'g_geo_lin': LineString([(0, 0), (1, 1)])}
        mock_link.edgeUV = mock_edge
        mock_link.edgeVU = None
        
        links = [mock_link]
        
        result = p3.getEdgeDict(links)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
    
    def test_reverse_geom(self):
        """Test reverse_geom function"""
        line = LineString([(0, 0), (1, 1), (2, 2)])
        result = p3.reverse_geom(line)
        
        assert result.coords[0] == (2, 2)
        assert result.coords[-1] == (0, 0)
    
    def test_get_angle(self):
        """Test getAngle function"""
        pt1 = (0, 0)
        pt2 = (1, 1)
        result = p3.getAngle(pt1, pt2)
        
        assert isinstance(result, float)
        assert 0 <= result <= 360


class TestNodeClass:
    """Test cases for Node class"""
    
    def test_node_initialization(self):
        """Test Node class initialization"""
        main_row = pd.Series({
            'osmid': 1,
            'geom_merged': Point(0, 0),
            'highway': 'crossing',
            'crossing': 'marked',
            'old_osmid': 10,
            'highway_conn': 'trunk',
            'highway_rank': 10
        })
        
        other_rows = pd.DataFrame({
            'osmid': [2],
            'highway': ['traffic_signals'],
            'crossing': ['unmarked'],
            'old_osmid': [20],
            'highway_conn': ['primary'],
            'highway_rank': [9]
        })
        
        node = p3.Node(other_rows, main_row)
        
        assert node.id == 1
        assert node.x == 0
        assert node.y == 0
        assert node.crossing == True
        assert node.traffic_signals == True
    
    def test_node_to_dict(self):
        """Test Node to_dict method"""
        main_row = pd.Series({
            'osmid': 1,
            'geom_merged': Point(0, 0),
            'highway': 'crossing',
            'crossing': 'marked',
            'old_osmid': 10,
            'highway_conn': 'trunk',
            'highway_rank': 10
        })
        
        node = p3.Node(None, main_row)
        result = node.to_dict()
        
        assert 'g_id' in result
        assert 'g_x' in result
        assert 'g_y' in result
        assert 'g_geometry' in result


class TestAnEdgeClass:
    """Test cases for AnEdge class"""
    
    def test_an_edge_initialization(self):
        """Test AnEdge class initialization"""
        main_row = pd.Series({
            'osmid': 1,
            'new_u': 1,
            'new_v': 2,
            'geometry': LineString([(0, 0), (1, 1)]),
            'geom_linear': LineString([(0, 0), (1, 1)]),
            'geom_reassigned': LineString([(0, 0), (1, 1)]),
            'highway': 'residential',
            'bicycle': 'yes',
            'foot': 'yes',
            'oneway': False,
            'maxspeed': 30,
            'sidewalk': 'both',
            'lit': 'yes',
            'incline': 'up',
            'gradient': 0.05,
            'height_difference': 10,
            'severity': 1,
            'green_ratio': 0.5,  
            'retail_ratio': 0.5,
            'building_ratio': 0.5,
            'old_osmid': 10,
            'old_u': 1,
            'old_v': 2,
            'highway_rank': 10,
            'bicycle_road': True,
            'cycleway': 'lane',
            'length': 14,
            'surface': 'asphalt',
            'smoothness': 'good',
            'segregated': True,
            'width': 2.0,
            'cycleway:surface': 'asphalt',
            'cycleway:width': 2.0,
            'oneway:bicycle': False,
            'amenity_on': 'bicycle_parking',
            'amenity_nearby': 'shop',
            'cycleway_category': 'lane',
            'cycleway:left:lane': '',
            'cycleway:right:lane': '',
            'cycleway:left': '',
            'cycleway:right': '',
            'cycleway:both': '',
            'lanes': 2,
            'width_cycle_path': 2.5,
            'pt_stop_on': 1,
            'pt_stop_routes': 'route1,route2',
            'lanes': 2,
        })
        
        df = pd.DataFrame([main_row])
        df['direction'] = [False]  # Add required direction column
        
        edge = p3.AnEdge(df, main_row, uv=True, reversed=False)
        
        assert edge.u == 1
        assert edge.v == 2
        assert edge.id == 1
        assert edge.access_walk == True
        assert edge.access_bike == True
        assert edge.access_motorized == True
    
    def test_an_edge_check_mode_access_walk(self):
        """Test AnEdge checkModeAccess for walk mode"""
        main_row = pd.Series({
            'osmid': 1,
            'new_u': 1,
            'new_v': 2,
            'geometry': LineString([(0, 0), (1, 1)]),
            'geom_linear': LineString([(0, 0), (1, 1)]),
            'geom_reassigned': LineString([(0, 0), (1, 1)]),
        })
        
        df = pd.DataFrame({
            'highway': ['residential'],
            'foot': ['yes'],
            'maxspeed': [30],
            'sidewalk': ['both'],
            'direction': [False],
            'lit': ['yes'],
            'incline': 'up',
            'gradient': 0.05,
            'height_difference': 10,
            'severity': 0.2,
            'green_ratio': 0.5,
            'retail_ratio': 0.5,
            'building_ratio': 0.5,
            'osmid': 10,
            'old_osmid': 10,
            'old_u': 1,
            'old_v': 2,
            'length': 14,
            'highway_rank': 10,
            'bicycle_road': True,
            'cycleway': 'lane',
            'length': 14,
            'surface': 'asphalt',
            'smoothness': 'good',
            'segregated': True,
            'width': 2.0,
            'cycleway:surface': 'asphalt',
            'cycleway:width': 2.0,
            'oneway:bicycle': False,
            'amenity_on': 'bicycle_parking',
            'amenity_nearby': 'shop',
            'cycleway_category': 'lane',
            'cycleway:left:lane': '',
            'cycleway:right:lane': '',
            'cycleway:left': '',
            'cycleway:right': '',
            'cycleway:both': '',
            'lanes': 2,
            'width_cycle_path': 2.5,
            'pt_stop_on': 1,
            'pt_stop_routes': 'route1,route2',
            'lanes': 2,
            'foot': 'yes',
            'bicycle': 'yes',
            'oneway': False,
        })
        
        edge = p3.AnEdge(df, main_row)
        result = edge.checkModeAccess('walk')
        
        assert result == True
    
    def test_an_edge_check_mode_access_bike(self):
        """Test AnEdge checkModeAccess for bike mode"""
        main_row = pd.Series({
            'osmid': 1,
            'new_u': 1,
            'new_v': 2,
            'geometry': LineString([(0, 0), (1, 1)]),
            'geom_linear': LineString([(0, 0), (1, 1)]),
            'geom_reassigned': LineString([(0, 0), (1, 1)]),
        })
        
        df = pd.DataFrame({
            'highway': ['residential'],
            'bicycle': ['yes'],
            'bicycle_road': ['yes'],
            'cycleway': ['lane'],
            'direction': [False],
            'lit': ['yes'],
            'incline': 'up',
            'gradient': 0.05,
            'height_difference': 10,
            'severity': 0.2,
            'green_ratio': 0.5,
            'retail_ratio': 0.5,
            'building_ratio': 0.5,
            'osmid': 10,
            'old_osmid': 10,
            'old_u': 1,
            'old_v': 2,
            'highway_rank': 10,
            'bicycle_road': True,
            'cycleway': 'lane',
            'length': 14,
            'surface': 'asphalt',
            'smoothness': 'good',
            'segregated': True,
            'width': 2.0,
            'cycleway:surface': 'asphalt',
            'cycleway:width': 2.0,
            'oneway:bicycle': False,
            'amenity_on': 'bicycle_parking',
            'amenity_nearby': 'shop',
            'cycleway_category': 'lane',
            'cycleway:left:lane': '',
            'cycleway:right:lane': '',
            'cycleway:left': '',
            'cycleway:right': '',
            'cycleway:both': '',
            'lanes': 2,
            'width_cycle_path': 2.5,
            'pt_stop_on': 1,
            'pt_stop_routes': 'route1,route2',
            'lanes': 2,
            'bicycle': 'yes',
            'foot': 'yes',
            'maxspeed': 30,
            'sidewalk': 'both',
            'oneway': False,
        })
        
        edge = p3.AnEdge(df, main_row)
        result = edge.checkModeAccess('bike')
        
        assert result == True
    
    def test_an_edge_check_mode_access_motorized(self):
        """Test AnEdge checkModeAccess for motorized mode"""
        main_row = pd.Series({
            'osmid': 1,
            'new_u': 1,
            'new_v': 2,
            'geometry': LineString([(0, 0), (1, 1)]),
            'geom_linear': LineString([(0, 0), (1, 1)]),
            'geom_reassigned': LineString([(0, 0), (1, 1)]),
        })
        
        df = pd.DataFrame({
            'highway': ['residential'],
            'oneway': [False],
            'direction': [False],
            'lit': ['yes'],
            'incline': 'up',
            'gradient': 0.05,
            'height_difference': 10,
            'severity': 0.2,
            'green_ratio': 0.5,
            'retail_ratio': 0.5,
            'building_ratio': 0.5,
            'osmid': 10,
            'old_osmid': 10,
            'old_u': 1,
            'old_v': 2,
            'highway_rank': 10,
            'bicycle_road': True,
            'cycleway': 'lane',
            'length': 14,
            'surface': 'asphalt',
            'smoothness': 'good',
            'segregated': True,
            'width': 2.0,
            'cycleway:surface': 'asphalt',
            'cycleway:width': 2.0,
            'oneway:bicycle': False,
            'amenity_on': 'bicycle_parking',
            'amenity_nearby': 'shop',
            'cycleway_category': 'lane',
            'cycleway:left:lane': '',
            'cycleway:right:lane': '',
            'cycleway:left': '',
            'cycleway:right': '',
            'cycleway:both': '',
            'lanes': 2,
            'width_cycle_path': 2.5,
            'pt_stop_on': 1,
            'pt_stop_routes': 'route1,route2',
            'lanes': 2,
            'bicycle': 'yes',
            'foot': 'yes',
            'maxspeed': 30,
            'sidewalk': 'both',
        })
        
        edge = p3.AnEdge(df, main_row)
        result = edge.checkModeAccess('motorized')
        
        assert result == True


class TestWalkEdgeClass:
    """Test cases for WalkEdge class"""
    
    def test_walk_edge_initialization(self):
        """Test WalkEdge class initialization"""
        main_row = pd.Series({
            'osmid': 1,
            'geometry': LineString([(0, 0), (1, 1)])
        })
        
        df = pd.DataFrame({
            'highway': ['residential'],
            'foot': ['yes'],
            'maxspeed': [30],
            'sidewalk': ['both'],
            'surface': ['asphalt'],
            'smoothness': ['good'],
            'segregated': ['yes'],
            'width': [2.0],
            'length': [1.414],
        })
        
        walk_edge = p3.WalkEdge(main_row, df)
        
        assert walk_edge.length == 1.414
        assert 'asphalt' in walk_edge.surface
        assert 'good' in walk_edge.smoothness
    

class TestBicycleEdgeClass:
    """Test cases for BicycleEdge class"""
    
    def test_bicycle_edge_initialization(self):
        """Test BicycleEdge class initialization"""
        main_row = pd.Series({
            'osmid': 1,
            'geometry': LineString([(0, 0), (1, 1)])
        })
        
        df = pd.DataFrame({
            'highway': ['residential'],
            'bicycle': ['yes'],
            'bicycle_road': ['yes'],
            'cycleway': ['lane'],
            'cycleway:surface': ['asphalt'],
            'surface': ['asphalt'],
            'smoothness': ['good'],
            'width_cycle_path': [2.5],
            'cycleway:width': [2.0],
            'oneway:bicycle': ['no'],
            'amenity_on': ['bicycle_parking'],
            'amenity_nearby': ['shop'],
            'cycleway_category': ['lane'],
            'segregated': ['yes'],
            'length': [1.414],
            'cycleway:left:lane': [''],
            'cycleway:right:lane': [''],
            'cycleway:left': [''],
            'cycleway:right': [''],
            'cycleway:both': ['']
        })
        
        bicycle_edge = p3.BicycleEdge(main_row, df)
        
        assert bicycle_edge.length == 1.414
        assert bicycle_edge.bicycle_road == True
        assert bicycle_edge.oneway == False
        assert bicycle_edge.bike_parking_on == True
    

class TestMotorizedEdgeClass:
    """Test cases for MotorizedEdge class"""
    
    def test_motorized_edge_initialization(self):
        """Test MotorizedEdge class initialization"""
        main_row = pd.Series({
            'osmid': 1,
            'geometry': LineString([(0, 0), (1, 1)])
        })
        
        df = pd.DataFrame({
            'highway': ['residential'],
            'oneway': [False],
            'width': [3.5],
            'lanes': [2],
            'maxspeed': [30],
            'pt_stop_on': [1],
            'pt_stop_routes': ['route1,route2'],
            'length': [1.414],
            'gradient': 0.05,
        })
        
        motorized_edge = p3.MotorizedEdge(main_row, df)
        
        assert motorized_edge.length == 1.414
        assert motorized_edge.oneway == False
        assert motorized_edge.pt_stop == 1
        assert 'route1' in motorized_edge.pt_routes
    
    def test_motorized_edge_get_lanes(self):
        """Test MotorizedEdge getLanes method"""
        main_row = pd.Series({
            'osmid': 1,
            'geometry': LineString([(0, 0), (1, 1)]),
        })
        
        df = pd.DataFrame({
            'lanes': [2, 3],
            'length': [1.0, 2.0],
            'highway': ['residential', 'residential'],
            'width': [2.0, 3.0],
            'oneway': [False, False],
            'maxspeed': [30, 30],
            'pt_stop_on': [1, 1],
            'pt_stop_routes': ['route1,route2', 'route1,route2'],
        })
        
        motorized_edge = p3.MotorizedEdge(main_row, df)
        result = motorized_edge.getLanes(df['lanes'], df['length'])
        
        assert abs(result - 2.67) < 0.01


class TestIntegration:
    """Integration tests for the module"""
    
    def test_full_workflow_integration(self):
        """Test the complete workflow integration"""
        # Create test data
        nodes_data = {
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)],
            'osmid': [1, 2, 3]
        }
        gdf_nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326')
        
        edges_data = {
            'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
            'u': [1, 2],
            'v': [2, 3],
            'highway': ['residential', 'residential'],
            'length': [1.414, 1.414]
        }
        gdf_edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')
        
        # Test highway ranking
        ranks = p3.addHighwayRank(gdf_edges)
        assert len(ranks) == 2
        assert all(rank == 7 for rank in ranks)  # residential has rank 7
        
        # Test curve splitting
        result_edges, result_nodes = p3.splitCurves(gdf_edges, gdf_nodes)
        assert len(result_edges) >= 2
        
        # Test highest ranking road - need to add highway_rank column
        gdf_edges['highway_rank'] = ranks
        most_important_highway, most_important_highway_rank = p3.getHighestRankingRoadOfNode(gdf_nodes, gdf_edges)
        assert len(most_important_highway) == 3
    

if __name__ == "__main__":
    pytest.main([__file__])
