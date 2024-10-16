"""
Scripts for importing osm network
- getOSMGraph_place: get osm graph for a place
- getOSMGraph_coords: get osm graph for a bounding box
- make_plot: plot the graph
- main: main function to run the script

Inputs:
- configFile.py: configuration file with all values 
    - version: version of the project
    - boundary_mode: 'place' or 'coords'
    - location: location name, e.g. 'Munich, Germany'
    - dist_in_meters: radius around location in meters, e.g. 1500
    - coords_upper_left: (lat, lon)
    - coords_lower_right: (lat, lon)
    - used_tags: list of strings, tags to be used for filtering, e.g. ['cycleway', 'cycleway:left']
    - p1_result_filepath: filepath for geopackage file with the osm network
Outputs:
- geopackage file with the osm network (saved to p1_result_filepath)
- plot of the osm network
"""

# %%

import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx

# FUNCTIONS ###################################################################################

def getOSMGraph_place(location, distance, tags):
    """
    Downloads the street network - using place name
    Args:
        @location: str, location name, e.g. 'Munich, Germany'
        @distance: int, radius around location in meters, e.g. 1500
        @tags: list of strings, tags to be used for filtering, e.g. ['cycleway', 'cycleway:left']
    Returns:
        graph: networkx.MultiDiGraph, graph of the city
        gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        gdf_edges: geopandas.GeoDataFrame, edges of the city
    """
    ox.settings.useful_tags_way = tags
    ox.settings.useful_tags_node = tags

    graph = ox.graph_from_address(location, dist=distance, dist_type='bbox',
                                  network_type='all',
                                  simplify=True, retain_all=False, truncate_by_edge=False,
                                  custom_filter=None)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    return graph, gdf_nodes, gdf_edges

def getOSMGraph_coords(coords_upper_left, coords_lower_right, tags):
    """
    Downloads the street network - using bounding box, i.e., coordinates
    Args:
        @coords_upper_left: tuple, (lat, lon)
        @coords_lower_right: tuple, (lat, lon)
        @tags: list of strings, tags to be used for filtering, e.g. ['cycleway', 'cycleway:left']
    Returns:
        graph: networkx.MultiDiGraph, graph of the city
        gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        gdf_edges: geopandas.GeoDataFrame, edges of the city
    """    
    ox.settings.useful_tags_way = tags
    ox.settings.useful_tags_node = tags

    graph = ox.graph_from_bbox(left=coords_upper_left[1], bottom=coords_lower_right[0], right=coords_lower_right[1], top=coords_upper_left[0], 
                               network_type='all',
                               simplify=True, retain_all=False, truncate_by_edge=False, 
                               custom_filter=None)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    return graph, gdf_nodes, gdf_edges

def make_plot(edgeGeometries, nodes, edgeColor='black', title=None):
    """
    Plots the graph
    Args:
        @edgeGeometries: geopandas.GeoSeries, geometries of the edges
        @nodes: geopandas.GeoDataFrame, nodes of the city
        @edgeColor: str, color of the edges
        @title: str, title of the plot
    """
    ax = edgeGeometries.plot(figsize=(10, 10), color=edgeColor, linewidth=0.3)
    plt.title(title)
    try:  # regular nodes
        plt.scatter([a.x for a in nodes.geometry], [a.y for a in nodes.geometry], color='red', s=4)
    except:  # for buffered nodes
        nodes.plot(ax=ax, color='red', linewidth=0.3)
    ax.axes.get_xaxis().set_visible(False);
    ax.axes.get_yaxis().set_visible(False);
    ax.axis('equal')
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    plt.show()

def main(configFile):
    version = configFile.version
    boundary_mode = configFile.boundary_mode
    if boundary_mode == 'place':
        location = configFile.location
        dist_in_meters = configFile.dist_in_meters
    else:
        coords_upper_left = configFile.coords_upper_left
        coords_lower_right = configFile.coords_lower_right
    used_tags = configFile.used_tags
    p1_result_fp = configFile.p1_result_filepath

    if boundary_mode == 'place':
        print(f"Getting OSM data for {location} with a radius of {dist_in_meters}m")
        graph, gdf_nodes, gdf_edges = getOSMGraph_place(location=location, distance=dist_in_meters, tags=used_tags)
    else:
        print(f"Getting OSM data for the area defined by the coordinates {np.round(coords_upper_left,3)} and {np.round(coords_lower_right,3)}")
        graph, gdf_nodes, gdf_edges = getOSMGraph_coords(coords_upper_left, coords_lower_right, used_tags)
    make_plot(gdf_edges.geometry, gdf_nodes)
    # save as one geopackage file
    ox.save_graph_geopackage(graph, directed=True, filepath=p1_result_fp)
    print(f"Saved graph to {p1_result_fp}")

if __name__ == "__main__":
    main()
    
# %%
#######################################
# Notes
#######################################

# NOTE: EXCLUDED FOR NOW AS POORLY AVAILABLE OR NOT CONSIDERED AS USEFUL:
# 'cycleway:both:lane', 'cycleway:left:lane', 'cycleway:right:lane', z.B. advisory, exclusive
# 'cycleway:surface:colour',
# 'parking:lane', 'parking:lane:left', 'parking:lane:right', 'parking:lane:both', z.B. parallel, perpendicular, diagonal
# 'footway:surface'
# 'footway', z.B. sidewalk
# 'crossing:island', 'button_operated'
