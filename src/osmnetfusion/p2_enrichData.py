"""
Script for enriching the osm network with the additional data (OSM and non-OSM)
- get_landuse_ratio: get landuse ratio for the edges
- improve_bike_edges: generate bike_access label to identify where cyclists can / cannot go AND add missing edges where cyclists can go in both directions, but there is not edge in the opposite direction
- add_cycle_paths: add cycleway category to the edges
- merge_similar_columns: merge similar columns
- add_elevation: add elevation to the nodes
- add_gradient: add gradient to the edges
- add_traffic_lights: add traffic lights to the nodes
- add_cycle_path_width: add cycle path width to the edges
- add_bicycle_parking: add bicycle parking to the edges
- add_pt_stops: add public transport stops to the edges
- update_idxs: update the indexes of the nodes and edges
- add_missing_columns: add missing columns to the edges --> necessary for p3_simplification
- save_p2_result_2_file: save the result to a geopackage file
- main: main function to run the script

Inputs
- configFile.py: configuration file with all values
    - network: filepath for geopackage file with the osm network
    - signals_fp: filepath for geopackage file with the traffic signals
    - cyclePathW_fp: filepath for csv file with the cycle path widths
    - bikeAmenities_fp: filepath for geopackage file with the bike amenities
    - elev_fp: filepath for json file with the elevations
    - greenLanduse_fp: filepath for geopackage file with the green landuse
    - retail_fp: filepath for geopackage file with the retail landuse
    - building_fp: filepath for geopackage file with the building landuse
    - ptStops_fp: filepath for geopackage file with the public transport stops
    - p2_result_fp: filepath for geopackage file with the result
Output
- geopackage file with the enriched osm network (saved to p2_result_fp)
"""

# %%

#######################################
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import os
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.find_common_type.*")

# FUNCTIONS ##################################################################################

def get_landuse_ratio(gdf_edges, kind='retail', input_file=None):
    """
    Get the landuse ratio for the edges
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
        @kind: str, kind of landuse, e.g. 'retail'
        @input_file: str, filepath for geopackage file with the landuse
    Returns:
        gdf_edges: geopandas.GeoDataFrame, edges with landuse ratio
    """    
    ########## STEP 0 ############
    # Load input file
    layers = {}
    for layername in fiona.listlayers(input_file):
        # print(layername)
        layers[layername] = gpd.read_file(input_file, layer=layername)

    gdf_nodes_landuse = layers['Point']
    if 'LineString' in fiona.listlayers(input_file):
        gdf_edges_landuse = layers['LineString']
    if 'Polygon' in fiona.listlayers(input_file):
        gdf_poly_landuse = layers['Polygon']
    if 'MultiPolygon' in fiona.listlayers(input_file):
        gdf_mpoly_landuse = layers['MultiPolygon']

    ########## STEP 1 ############
    # Geodataframe to the same Coordinate Reference System (CRS)
    # 4326 = WGS84 Latitude/Longitude
    # EPSG 3043 = UTM for calculate length in Meter
    # Buffer geometry to get cycle path near green
    gdf_edges = gdf_edges.to_crs(4326)

    gdf_nodes_landuse = gdf_nodes_landuse.to_crs(4326)

    if 'LineString' in fiona.listlayers(input_file):
        gdf_edges_landuse = gdf_edges_landuse.to_crs(4326)
        gdf_edges_landuse_buffered = gdf_edges_landuse.copy()
        gdf_edges_landuse_buffered = gdf_edges_landuse_buffered.to_crs("EPSG:3043")
        gdf_edges_landuse_buffered['geometry'] = gdf_edges_landuse_buffered['geometry'].buffer(5)

    if 'Polygon' in fiona.listlayers(input_file):
        gdf_poly_landuse = gdf_poly_landuse.to_crs(4326)
        gdf_poly_landuse_buffered = gdf_poly_landuse.copy()
        gdf_poly_landuse_buffered = gdf_poly_landuse_buffered.to_crs("EPSG:3043")
        gdf_poly_landuse_buffered['geometry'] = gdf_poly_landuse_buffered['geometry'].buffer(5)

    if 'MultiPolygon' in fiona.listlayers(input_file):
        gdf_mpoly_landuse = gdf_mpoly_landuse.to_crs(4326)
        gdf_mpoly_landuse_buffered = gdf_mpoly_landuse.copy()
        gdf_mpoly_landuse_buffered = gdf_mpoly_landuse_buffered.to_crs("EPSG:3043")
        gdf_mpoly_landuse_buffered['geometry'] = gdf_mpoly_landuse_buffered['geometry'].buffer(5)

    ########## STEP 2 ############
    # Buffered network geometry (buffer = 10 m)
    gdf_edges_buffered = gdf_edges.copy()
    gdf_edges_buffered = gdf_edges_buffered.to_crs("EPSG:3043")
    gdf_edges_buffered['geometry'] = gdf_edges_buffered['geometry'].buffer(10)

    ########## STEP 3 ############
    # interate über die buffer-bike, gehe grüne Punkte durch und zähle nach oben, wenn Punkt im Polygon ist
    gdf_edges[f'{kind}_points'] = 0.0
    gdf_edges[f'{kind}_ratio_point'] = 0.0
    gdf_edges_buffered = gdf_edges_buffered.to_crs(4326)
    for edge in gdf_edges_buffered.itertuples():
        r = gdf_nodes_landuse.within(edge.geometry)
        gdf_edges.loc[edge.Index, f'{kind}_points'] = r.sum()
        z = r.sum() * 5 / edge.length
        point_ratio = min(z, 1)
        gdf_edges.loc[edge.Index, f'{kind}_ratio_point'] = point_ratio

    ########## STEP 4 ############
    # intersection from bike edges with buffered green edges, poly and mpoly
    gdf_edges = gdf_edges.to_crs("EPSG:3043")
    if 'LineString' in fiona.listlayers(input_file):
        gdf_edges_landuse_buffered = gdf_edges_landuse_buffered.to_crs("EPSG:3043")
    if 'Polygon' in fiona.listlayers(input_file):
        gdf_poly_landuse = gdf_poly_landuse.to_crs("EPSG:3043")
    if 'MultiPolygon' in fiona.listlayers(input_file):
        gdf_mpoly_landuse = gdf_mpoly_landuse.to_crs("EPSG:3043")

    gdf_edges[f'{kind}_ratio_poly'] = 0.0

    # Intersection buffered edges/polygons/multipolygons with edges_bike
    dfs = []
    if 'LineString' in fiona.listlayers(input_file):
        dfs.append(gdf_edges_landuse_buffered)
    if 'Polygon' in fiona.listlayers(input_file):
        dfs.append(gdf_poly_landuse_buffered)
    if 'MultiPolygon' in fiona.listlayers(input_file):
        dfs.append(gdf_mpoly_landuse_buffered)
    for df in dfs:
        for edge in df.itertuples():
            r = gdf_edges.sindex.query(edge.geometry, predicate='intersects')
            for each in r:
                inside = gdf_edges.loc[each]['geometry'].intersection(edge.geometry).length
                ratio = inside / gdf_edges.loc[each, 'length']
                gdf_edges.loc[each, f'{kind}_ratio_poly'] = min(gdf_edges.loc[each, f'{kind}_ratio_poly'] + ratio,1)
    
    ########## STEP 5 ############
    # combine green-ratio_point with green_ratio_poly to green_ratio
    gdf_edges[f'{kind}_ratio'] = 0.0
    for edge in gdf_edges.itertuples():
        gdf_edges.loc[edge.Index, f'{kind}_ratio'] = min(
            gdf_edges.loc[edge.Index, f'{kind}_ratio_point'] + gdf_edges.loc[edge.Index, f'{kind}_ratio_poly'], 1)
    
    # Back to correct crs
    gdf_edges = gdf_edges.to_crs('EPSG:4326')
    print(f'gdf_edges shape after adding {kind} ratio {gdf_edges.shape}')
    return gdf_edges

def improve_bike_edges(gdf_edges):
    """
    Generate bike_access label to identify where cyclists can / cannot go
    AND add missing edges where cyclists can go in both directions, but there is not edge in the opposite direction
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
    Returns:
        gdf_edges: geopandas.GeoDataFrame, edges with bike_access label
    """
    # EDGES
    gdf_edges['bike_access'] = 'yes'  # yes, no, bike only
    
    # where cyclists cannot go
    edges_not_allowed = gdf_edges[
        (gdf_edges.highway.str.contains('trunk') == True) |
        (gdf_edges.bicycle.str.contains('use_sidepath') == True) |
        (gdf_edges.bicycle.str.contains('no') == True)
        ]
    gdf_edges.loc[edges_not_allowed.index, 'bike_access'] = 'no'
    
    # if oneway for cars, add opposite link for bikes
    if ('oneway:bicycle' in gdf_edges.columns) and ('cycleway' in gdf_edges.columns):
        edges_to_add = gdf_edges[
            # this must be True
            (gdf_edges['oneway'] == True) & (
                # one of these must be True
                    (gdf_edges['oneway:bicycle'].str.contains('no') == True) |
                    (gdf_edges.cycleway.str.contains('opposite') == True)
            )]
    elif 'cycleway' in gdf_edges.columns:
        edges_to_add = gdf_edges[(gdf_edges['oneway'] == True) & (gdf_edges.cycleway.str.contains('opposite') == True)]
    else:
        edges_to_add = gdf_edges[(gdf_edges['oneway'] == True)]
    to_add = []
    for itr, row in edges_to_add.iterrows():
        # if there is no edge in the opposite direction
        if len(gdf_edges[(gdf_edges.u == row.v) & (gdf_edges.v == row.u)]) < 1:
            new_row = row
            # swap start / end
            start, end = row.v, row.u  # reversed
            new_row.u = start
            new_row.v = end
            new_row.bike_access = 'bike_only'
            new_row.reversed = not row.reversed
            to_add.append(new_row.values)
        else:  # opposite edges already exists, so only update bike_access
            gdf_edges.loc[(gdf_edges.u == row.v) & (gdf_edges.v == row.u), 'bike_access'] = 'yes'
    print('Edges added (bc only oneway for cars and no edge in opposite direction)', len(to_add))
    to_add = gpd.GeoDataFrame(to_add, index=range(len(gdf_edges), len(gdf_edges) + len(to_add)),
                              columns=gdf_edges.columns)
    gdf_edges = gpd.GeoDataFrame(pd.concat([gdf_edges, to_add]))
    print(f'gdf_edges shape after adding building ratio {gdf_edges.shape}')
    
    return gdf_edges

def add_cycle_paths(gdf_edges):
    """
    Add cycleway category - i.e. what type of cycle path is on an edge based on the tags,
    e.g., bidirectional cycle path or only an advisory lane on the road
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
    Returns:
        @gdf_edges: geopandas.GeoDataFrame, edges with cycleway category column
    """
    # NOTE: an error is thrown if the tag does not exist for any of the considered edges, hence, the corresponding lines are commented
    # in the large network of all of Munich, all of the tags are present
    # TODO: make this section more robust

    # APPROACH
    # Schutzstreifen (advisory_lane) : cycleway:(right/left):lane = advisory; cycleway = opposite_lane
    # Radfahrstreifen (exclusive_lane): cycleway:lane = exclusive, cycleway:right/left/both = lane and cycleway:right/left/both:bicycle = designated
    # Auf Busspur (shared_lane):cycleway = shared_busway
    # Einrichtungsradweg (one_direction_cycle_path (one_track)): highway = cycleway (and bicycle = designated), cycleway = track/ opposite_track, cycleway:backward/forward = track, cycleway:right/left = track (and cycleway:right/left:bicycle=designated and cycleway:right/left:segregate=yes), highway = path and bicycle = designated and segregated = yes, cycleway:right/left:oneway = yes
    # Einseitiger Zweirichtungsradweg(two_direction_cycle_path(two_track)): highway = cycleway (cycleway:right/left = track (?)) and oneway = no
    # Gemeisamer Geh- und Radweg(foot_and_cycle_path): highway = path/residential/service/track and bicycle = designated and segregated = no, sidewalk:right/left:bicylce = yes, highway = footway and bicycle = yes
    # Fahrradstraße (bicycle_road): bicycle_road = yes
    # Mischverkehr (mixed_traffic): Rest

    # check if column exists and apply condition
    def contains_condition(df, column, substr):
        return df[column].str.contains(substr) if column in df.columns else pd.Series([False] * len(df))

    # calculation of cycleway_category bases on osm key/tags
    gdf_edges['cycleway_category'] = ''
    # advisory_lane
    cc_advisory_lane = (
        contains_condition(gdf_edges, 'cycleway', 'lane') |
        contains_condition(gdf_edges, 'cycleway', 'opposite') |
        contains_condition(gdf_edges, 'cycleway:lane', 'advisory') |
        contains_condition(gdf_edges, 'cycleway:left:lane', 'advisory') |
        contains_condition(gdf_edges, 'cycleway:right:lane', 'advisory')
    )
    gdf_edges.loc[cc_advisory_lane, 'cycleway_category'] = 'advisory_lane'
    # exclusive lane 
    cc_exclusive_lane = (
        (contains_condition(gdf_edges, 'cycleway', 'lane') & contains_condition(gdf_edges, 'bicycle', 'designated')) |
        contains_condition(gdf_edges, 'cycleway:lane', 'exclusive') |
        contains_condition(gdf_edges, 'cycleway:left:lane', 'exclusive') |
        (contains_condition(gdf_edges, 'cycleway:left', 'lane') & contains_condition(gdf_edges, 'cycleway:left:bicycle', 'designated')) |
        (contains_condition(gdf_edges, 'cycleway:both', 'lane') & contains_condition(gdf_edges, 'cycleway:both:bicycle', 'designated')) |
        (contains_condition(gdf_edges, 'cycleway:right', 'lane') & contains_condition(gdf_edges, 'cycleway:right:bicycle', 'designated')) |
        contains_condition(gdf_edges, 'cycleway:right:lane', 'exclusive')
    )
    gdf_edges.loc[cc_exclusive_lane, 'cycleway_category'] = 'exclusive_lane'

    # shared_lane
    cc_shared_lane = (
        contains_condition(gdf_edges, 'cycleway', 'shared_busway'))
    gdf_edges.loc[cc_shared_lane, 'cycleway_category'] = 'shared_lane'

    # bicycle_road
    cc_bicycle_road = (
        contains_condition(gdf_edges, 'bicycle_road', 'yes'))
    gdf_edges.loc[cc_bicycle_road, 'cycleway_category'] = 'bicycle_road'

    # one_direction_cycle_path
    cc_one_track = (
        contains_condition(gdf_edges, 'highway', 'cycleway') |
        contains_condition(gdf_edges, 'cycleway', 'track') |
        contains_condition(gdf_edges, 'cycleway:left', 'track') |
        contains_condition(gdf_edges, 'cycleway:right', 'track') |
        contains_condition(gdf_edges, 'cycleway:both', 'track') |
        contains_condition(gdf_edges, 'bicycle:backward', 'track') |
        contains_condition(gdf_edges, 'bicycle:forward', 'track') |
        contains_condition(gdf_edges, 'cycleway:right:oneway', 'yes|-1') |
        contains_condition(gdf_edges, 'cycleway:left:oneway', 'yes|-1') |
        (contains_condition(gdf_edges, 'highway', 'path') & contains_condition(gdf_edges, 'bicycle', 'designated') & contains_condition(gdf_edges, 'segregated', 'yes'))
    )
    gdf_edges.loc[cc_one_track, 'cycleway_category'] = 'one_direction_cycle_path'

    # two_direction_cycle_path
    cc_two_track = (
        (contains_condition(gdf_edges, 'cycleway:right', 'track') & ((gdf_edges['oneway'] == False) | contains_condition(gdf_edges, 'cycleway:right:oneway', 'no'))) |
        (contains_condition(gdf_edges, 'cycleway:left', 'track') & ((gdf_edges['oneway'] == False) | contains_condition(gdf_edges, 'cycleway:left:oneway', 'no'))) |
        contains_condition(gdf_edges, 'cycleway:right:oneway', 'no') |
        contains_condition(gdf_edges, 'cycleway:left:oneway', 'no') |
        (contains_condition(gdf_edges, 'highway', 'cycleway') & (gdf_edges['oneway'] == False))
    )
    gdf_edges.loc[cc_two_track, 'cycleway_category'] = 'two_direction_cycle_path'

    # separately list streets with cycle lane and cycle track (i.e. left and right side different)
    cc_track_or_lane = (cc_advisory_lane | cc_exclusive_lane) & (cc_one_track | cc_two_track)
    gdf_edges.loc[cc_track_or_lane, 'cycleway_category'] = 'track_or_lane'

    # foot_and_cycle_path
    cc_fac_path = (
        (contains_condition(gdf_edges, 'highway', 'path') & contains_condition(gdf_edges, 'bicycle', 'designated') & contains_condition(gdf_edges, 'segregated', 'no')) |
        (contains_condition(gdf_edges, 'highway', 'footway') & contains_condition(gdf_edges, 'bicycle', 'yes'))
    )
    gdf_edges.loc[cc_fac_path, 'cycleway_category'] = 'foot_and_cycle_path'

    # pedestrian street with cycling allowed
    cc_pedestrian_street = (
            gdf_edges['highway'].str.contains('pedestrian') & gdf_edges['bicycle'].str.contains('yes'))
    gdf_edges.loc[cc_pedestrian_street, 'cycleway_category'] = 'pedestrian_street'

    # remaining streets (mixed traffic)
    idcs = gdf_edges[gdf_edges['cycleway_category'] == 0].index
    cc_mixed_traffic = (
        gdf_edges['cycleway_category'] == 0)
    gdf_edges.loc[cc_mixed_traffic, 'cycleway_category'] = gdf_edges.loc[cc_mixed_traffic, 'highway']
    
    print('Total number of edges:', len(gdf_edges))
    print('Number of cycleways identified:', len(gdf_edges) - len(gdf_edges.loc[idcs, :]))
    # print("Examples from gdf_edges")
    # print(gdf_edges.sample(2))
    return gdf_edges

def merge_similar_columns(gdf_edges, column1, column2, newName=None):
    """
    Merge similar columns into one
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
        @column1: str, column name 1
        @column2: str, column name 2
        @newName: str, new column name
    Returns:
        @gdf_edges: geopandas.GeoDataFrame, edges with merged columns
    """
    newName = column1 if newName is None else newName

    # combine the columns
    gdf_edges['new_column'] = ''
    for itr, edge in gdf_edges.iterrows():
        if column1 in gdf_edges.columns:
            if edge[column1]:
                gdf_edges.loc[itr, 'new_column'] = edge[column1]
        if column2 in gdf_edges.columns:
            if edge[column2]:
                gdf_edges.loc[itr, 'new_column'] = edge[column2]
        
    if column1 in gdf_edges.columns:
        gdf_edges = gdf_edges.drop(columns=[column1])
    if column2 in gdf_edges.columns:
        gdf_edges = gdf_edges.drop(columns=[column2])
    gdf_edges = gdf_edges.rename(columns={'new_column': newName})
    
    return gdf_edges

def add_elevation(gdf_nodes, input_file):
    """
    Add elevation to the nodes
    Args:
        @gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        @input_file: str, filepath for json file with the elevations
    Returns:
        @gdf_nodes: geopandas.GeoDataFrame, nodes with elevation
    """
    if not os.path.isfile(input_file):
        print('Input file {} containing elevation doesn''t exist.'.format(input_file))
        gdf_nodes['elevation'] = None
        # gdf_nodes['elevation'] = gdf_nodes['elevation'].fillna('')
        return gdf_nodes, False
    
    with open(input_file, 'r') as fp:
        # Open json with elevation data
        elevations = eval(fp.readline())
    if elevations is None or len(elevations) == 0:
        raise ValueError('Missing elevation data')
    for e in elevations:
        gdf_nodes.loc[e['idx'], 'elevation'] = e['elevation']

    rows_added = len(elevations)  # gdf_nodes['elevations'].count(axis=1)
    print(f'Added elevation to {rows_added} rows')
    return gdf_nodes, True

def add_gradient(gdf_nodes, gdf_edges, elev_fp):
    """
    calculates severity of slope from gradient
    Args:
        @gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
    Returns:
        @gdf_nodes: geopandas.GeoDataFrame, nodes with elevation
        @gdf_edges: geopandas.GeoDataFrame, edges with gradient and severity
    """
    # get elevation
    gdf_nodes, elevAdded = add_elevation(gdf_nodes, elev_fp)
    if elevAdded==False:
        gdf_edges['height_difference'] = None
        gdf_edges['gradient'] = None
        gdf_edges['severity'] = None
        # gdf_edges['height_difference'] = gdf_edges['height_difference'].fillna('')
        # gdf_edges['gradient'] = gdf_edges['gradient'].fillna('')
        # gdf_edges['severity'] = gdf_edges['severity'].fillna('')
        return gdf_nodes, gdf_edges

    for edge in gdf_edges.itertuples():
        height1 = gdf_nodes.loc[gdf_nodes['osmid'] == edge[1], 'elevation'].values[0]
        height2 = gdf_nodes.loc[gdf_nodes['osmid'] == edge[2], 'elevation'].values[0]
        if height1 >= height2:
            hdifference = height1 - height2
        else:
            hdifference = height2 - height1

        gdf_edges.loc[edge.Index, 'height_difference'] = hdifference
        length = gdf_edges.loc[edge.Index, 'length']
        gradient = hdifference / length
        severity = hdifference ** 2 / length
        gdf_edges.loc[edge.Index, 'gradient'] = round(gradient, 4)
        gdf_edges.loc[edge.Index, 'severity'] = round(severity, 4)
    return gdf_nodes, gdf_edges

def add_traffic_lights(gdf_nodes, input_file):
    """
    Add traffic lights to the nodes
    Args:
        @gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        @input_file: str, filepath for geopackage file with the traffic signals
    Returns:
        @gdf_nodes: geopandas.GeoDataFrame, nodes with traffic lights
    """
    # some (BUT NOT ALL) traffic lights are already included in the OSMNX data.
    # add the remaining traffic lights here
    # gdf_nodes = gdf_nodes.drop(index=gdf_nodes[gdf_nodes.highway=='traffic_signals'].index)

    # LOAD OSM TRAFFIC SIGNALS DATA
    if not os.path.isfile(input_file):
        raise ValueError('Input file {} containing traffic signal data doesn''t exist.'.format(input_file))
    traffic_signals_df = gpd.read_file(input_file)
    if traffic_signals_df.shape[0] == 0:
        raise ValueError('traffic lights file is empty')

    if 'osmid' not in traffic_signals_df.columns:
        traffic_signals_df[['geomtype', 'osmid']] = traffic_signals_df['id'].str.split("/", expand=True)


    # PLAN:
    # traffic lights are not part of network nodes but they are individual nodes (hence the label highway=traffic_signal)
    # add relevant traffic signal information to the closest node in our network
    # nodes are clustered later on, hence the information will be included then (once per traffic signal)
    osm_nodes = gdf_nodes.copy().to_crs('EPSG:3857')  # metre-based crs
    traffic_signals_df = traffic_signals_df.to_crs('EPSG:3857')  # metre-based crs

    # do this for each traffic signal (approx. 4400)
    ts_df = gpd.sjoin_nearest(traffic_signals_df[['osmid', 'geometry']], osm_nodes[['osmid', 'geometry']], how='left',
                              distance_col="distances", lsuffix='ts', rsuffix='node', max_distance=20)
    # idcs in ts_df and osm_nodes
    mask = ts_df.osmid_ts == ts_df.osmid_node
    print('Previously included because of osmnx: %.2f%%' % (len(ts_df[mask]) / len(ts_df) * 100))
    # keep only smallest distance of each
    ts_df = ts_df.loc[~mask, ['osmid_ts', 'distances', 'index_node']].groupby(by=['osmid_ts'], as_index=False).min().sort_values(
        by='distances')
    # match information to node network
    ts_df = ts_df[['osmid_ts', 'index_node']].groupby('index_node', as_index=False).agg({'osmid_ts': lambda x: ','.join(str(x))})
    gdf_nodes['traffic_signals'] = ''
    gdf_nodes.loc[ts_df.index_node, 'traffic_signals'] = ts_df.osmid_ts.values
    
    return gdf_nodes

def add_cycle_path_width(gdf_edges, input_file):
    """
    Add cycle path width to the edges
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
        @input_file: str, filepath for csv file with the cycle path widths
    Returns:
        @gdf_edges: geopandas.GeoDataFrame, edges with cycle path width
    """
    # Open csv with cycle path widths data
    if not os.path.isfile(input_file):
        print('Input file {} containing cycle path widths data doesn''t exist.'.format(input_file))
        gdf_edges['width_cycle_path'] = None
        return gdf_edges
    widths_df = pd.read_csv(input_file)  # columns=['osmid','width','distances','location']
    if widths_df.shape[0] == 0:
        raise ValueError('cycle path widths file is empty')

    widths_df.columns = ['osmid', 'width_cycle_path', 'distances', 'location']
    widths_df = widths_df[['osmid','width_cycle_path']].groupby('osmid').min().reset_index()  # .mean()

    gdf_edges = gdf_edges.merge(widths_df, on='osmid', how='left')
    return gdf_edges

def add_bicycle_parking(gdf_edges, input_file):
    """
    Add bicycle parking to the edges
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
        @input_file: str, filepath for geopackage file with the bicycle parking
    Returns:
        @gdf_edges: geopandas.GeoDataFrame, edges with bicycle parking
    """
    # 4mins --> 1.5% --> 4-5hrs for all
    amenity_dist_lim = 200  # meters
    if not os.path.isfile(input_file):
        print('Input file {} containing bicycle parking data doesn''t exist.'.format(input_file))
        gdf_edges['amenity_on'] = ''
        gdf_edges['amenity_nearby'] = ''
        return gdf_edges
    
    gdf_cycle_parking = gpd.read_file(input_file, driver="GPKG")  # , layer='accidents'
    if gdf_cycle_parking.shape[0] == 0:
        raise ValueError('bicycle parking file is empty')

    gdf_cycle_parking.set_crs('EPSG:4326')
    # count_values = gdf_cycle_parking.amenity.value_counts()
    # uniques = gdf_cycle_parking.bicycle_parking.unique()
    # --- LABEL 1 --- amenity_on
    # assign each amenity to the closest osm link
    # where bike_access='yes' --> faster and more sensible
    gdf_edges['amenity_on'] = ''
    # --- LABEL 2 --- amenity_nearby
    # assign each amenity to the osm links within amenity_dist_lim (200m)
    # where bike_access='yes' --> faster and more sensible
    gdf_edges['amenity_nearby'] = ''
    osm_edges = gdf_edges[gdf_edges.bike_access == 'yes'].copy()
    osm_edges = osm_edges.to_crs('EPSG:3857')  # metre-based crs
    gdf_cycle_parking = gdf_cycle_parking.to_crs('EPSG:3857')  # metre-based crs
    # --- LABEL 3 --- dist to nearest bike amenity
    # NOT IMPLEMENTED, as there are too few amenities

    # for each amenity (approx. 3000)
    no_nodes_nearby_count = 0
    has_nodes_nearby_count = 0
    for itr, row in gdf_cycle_parking.iterrows():
        pt = row.geometry
        # FASTER
        dists = gpd.sjoin_nearest(gdf_cycle_parking.loc[itr:itr, ['geometry', 'amenity']],
                                  osm_edges[['osmid', 'geometry']], how='inner', distance_col="distances",
                                  lsuffix='amenity', rsuffix='edge', max_distance=amenity_dist_lim)
        if len(dists) < 1:
            no_nodes_nearby_count += 1
            continue
        has_nodes_nearby_count += 1
        dists = dists.set_index(dists.index_edge)
        gdf_edges.loc[dists.index.values[0], 'amenity_on'] = (
                gdf_edges.loc[dists.index.values[0], 'amenity_on'] + ', ' + row.amenity) if (
                gdf_edges.loc[dists.index.values[0], 'amenity_on'] != '') else (row.amenity)
        gdf_edges.loc[dists.index.values, 'amenity_nearby'] = gdf_edges.loc[
            dists.index.values, 'amenity_nearby'].apply(
            lambda x: (x + ', ' + row.amenity) if (x != '') else row.amenity)

    # too many nodes do not have nearby nodes, added a counter
    print(f'{no_nodes_nearby_count} nodes have no nodes within {amenity_dist_lim}m')
    print(f'{has_nodes_nearby_count} nodes have at lest 1 node within {amenity_dist_lim}m')
    return gdf_edges

def add_pt_stops(gdf_edges, input_file):
    """
    Add public transport stops to the edges
    Args:
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
        @input_file: str, filepath for geopackage file with the public transport stops
    Returns:
        @gdf_edges: geopandas.GeoDataFrame, edges with public transport stops
            - pt_stop_on: 0/1, if there is a public transport stop on the edge
            - pt_stop_count: number of public transport stops on the edge
            - pt_stop_routes: names of the public transport stops on the edge
    """
    # TEST
    # use the 'Point' layer of the public transport data
    if not os.path.isfile(input_file):
        print('Input file {} containing public transport data doesn''t exist.'.format(input_file))
        gdf_edges['pt_stop_on'] = 0
        gdf_edges['pt_stop_count'] = 0
        gdf_edges['pt_stop_routes'] = ''
        return gdf_edges
    
    gdf_pt_stops = gpd.read_file(input_file, driver="GPKG", layer='Point')
    if gdf_pt_stops.shape[0] == 0:
        raise ValueError('public transport stops file is empty')
    
    # match the public transport stops to the closest osm link which is accessible by car
    gdf_edges['pt_stop_on'] = 0 # no
    subset_edges = gdf_edges[gdf_edges.highway.isin(['residential', 'service', 'tertiary', 'secondary', 'primary', 'trunk', 'motorway'])].copy()
    subset_edges = subset_edges.to_crs('EPSG:3857')  # metre-based crs
    gdf_pt_stops = gdf_pt_stops.to_crs('EPSG:3857')  # metre-based crs
    # use sjoin_nearest to find the closest link for each PT stops
    dists = gpd.sjoin_nearest(gdf_pt_stops[['member_ref', 'geometry','name']], subset_edges[['osmid', 'geometry']], how='inner', distance_col="distances", lsuffix='pt', rsuffix='edge', max_distance=30)
    # dists = dists.set_index(dists.index_edge)
    print(len(gdf_pt_stops), len(dists))
    gdf_edges.loc[dists.index_edge.values, 'pt_stop_on'] = 1 # yes

    # now count the number of different PT stops on each edge i.e. unique stop names
    gdf_edges['pt_stop_count'] = 0
    gdf_edges['pt_stop_routes'] = ''
    for idx, group in dists.groupby(by='index_edge'):
        gdf_edges.loc[idx, 'pt_stop_count'] = len(group)
        gdf_edges.loc[idx, 'pt_stop_routes'] = ', '.join(group['name'].values)
    
    return gdf_edges

def update_idxs(gdf_nodes, gdf_edges):
    """
    Update the indices of the nodes and edges
    Args:
        @gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
    Returns:
        @gdf_nodes: geopandas.GeoDataFrame, nodes with updated indices
        @gdf_edges: geopandas.GeoDataFrame, edges with updated indices
    """
    # Figure out & generate new idcs - as some edges were added
    # 1. check if any irrelevant idcs in gdf_nodes
    # print(len(gdf_nodes.osmid.unique()))  # all unique nodes
    # print(len(np.unique(gdf_edges[['u', 'v']].values)))  # all nodes associated with edges
    # SAME! GREAT!
    # print(len(gdf_nodes))
    # 2. make new nodes idcs
    gdf_nodes['new_node_idx'] = gdf_nodes.index.values
    # match new node idcs to edges' u and v
    gdf_edges['new_u'] = \
        pd.merge(gdf_edges[['u']], gdf_nodes[['new_node_idx', 'osmid']], how='left', left_on='u',
                 right_on='osmid')['new_node_idx']
    gdf_edges['new_v'] = \
        pd.merge(gdf_edges[['v']], gdf_nodes[['new_node_idx', 'osmid']], how='left', left_on='v',
                 right_on='osmid')['new_node_idx']
    # 3. make new edge idcs
    gdf_edges['new_edge_idx'] = gdf_edges.index.values
    # print(gdf_edges.head())
    # match new edge idcs to nodes' refs
    # --> not applicable
    # 4. rename all
    gdf_edges = gdf_edges.rename(columns={
        "u": "old_u",
        "v": "old_v",
        "osmid": "old_osmid"}
    )
    gdf_edges = gdf_edges.rename(columns={
        "new_u": "u",
        "new_v": "v",
        "new_edge_idx": "osmid"  ###
    })
    gdf_nodes = gdf_nodes.rename(columns={
        "osmid": "old_osmid"
    })
    gdf_nodes = gdf_nodes.rename(columns={
        "new_node_idx": "osmid"  ###
    })
    print("New idxs were generated")
    return gdf_nodes, gdf_edges

# make sure all columns are in the data
def add_missing_columns(gdf_nodes, gdf_edges):
    """
    Add missing columns to the nodes and edges
    Args:
        @gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
    Returns:
        @gdf_nodes: geopandas.GeoDataFrame, nodes with all required columns
        @gdf_edges: geopandas.GeoDataFrame, edges with all required columns
    """

    # some columns are not in the dataframe, as there is no osm information for these columns
    # NOTE: maybe use configFile.used_tags to get all possible tags

    # NODES
    cols = ['osmid','old_osmid','y','x','street_count','highway','crossing','bicycle','foot',\
            'barrier','lit','width','public_transport','bicycle_parking']
    for c in cols:
        if c not in gdf_nodes.columns:
            gdf_nodes[c] = ''

    # EDGES
    cols = ['sidewalk', 'cycleway', 'bicycle_road', 'oneway:bicycle', 'cycleway:right', 'cycleway:left', \
            'cycleway:both', 'cycleway:right:lane', 'cycleway:left:lane', 'ramp:bicycle', 'public_transport', \
            'cycleway:surface', 'cycleway:width', 'crossing', 'width', 'smoothness', 'gradient', 'elevation', 'height_difference', 'severity']
    for c in cols:
        if c not in gdf_edges.columns:
            gdf_edges[c] = ''
    # gdf_edges['width_cycle_path'] = '' --> manually added
    
    return gdf_nodes, gdf_edges

def save_p2_result_2_file(gdf_nodes, gdf_edges, output_file):
    """
    Save the nodes and edges to a geopackage file
    Args:
        @gdf_nodes: geopandas.GeoDataFrame, nodes of the city
        @gdf_edges: geopandas.GeoDataFrame, edges of the city
        @output_file: str, filepath for the geopackage file
    """
    # Save as graph as Geopackage
    gdf_nodes2 = gdf_nodes.set_index('osmid')
    gdf_edges['key'] = gdf_edges['key'].astype(int)
    gdf_edges2 = gdf_edges.set_index(['u', 'v', 'key'])
    assert gdf_nodes2.index.is_unique and gdf_edges2.index.is_unique
    # graph_attrs = {'created_with': 'OSMnx 1.2.2', 'crs': 'epsg:4326', 'simplified': True}
    # TODO: NOTE: gradient missing in gpkg when empty --> bc empty columns saved to the gpkg?
    graph = ox.graph_from_gdfs(gdf_nodes2, gdf_edges2) # , graph_attrs
    ox.save_graph_geopackage(graph, directed=True, filepath=output_file)
    print(f'Graph saved to {output_file}')

def main(configFile, public_transport=True, accidents=True, cycle_path_width=True, elevation=True):
    network = configFile.p1_result_filepath
    signals_fp = configFile.signals_filepath
    cyclePathW_fp = configFile.cycle_path_w_filepath
    bikeAmenities_fp = configFile.bike_amenities_filepath
    elev_fp = configFile.elev_filepath
    greenLanduse_fp = configFile.green_landuse_filepath
    retail_fp = configFile.retail_filepath
    building_fp = configFile.building_filepath
    ptStops_fp = configFile.pt_stops_filepath
    p2_result_fp = configFile.p2_result_filepath

    # load base network
    gdf_edges = gpd.read_file(network, layer='edges')
    gdf_nodes = gpd.read_file(network, layer='nodes')
    print(f'loaded nodes data of shape {gdf_nodes.shape}, edges data of shape {gdf_edges.shape}')

    # add landuse ratios
    gdf_edges = get_landuse_ratio(gdf_edges, kind='green', input_file=greenLanduse_fp)
    gdf_edges = get_landuse_ratio(gdf_edges, kind='retail', input_file=retail_fp)
    gdf_edges = get_landuse_ratio(gdf_edges, kind='building', input_file=building_fp)
    
    # clean data and add some information
    gdf_edges = improve_bike_edges(gdf_edges)
    gdf_edges = add_cycle_paths(gdf_edges)
    if elevation:
        gdf_nodes, gdf_edges = add_gradient(gdf_nodes, gdf_edges, elev_fp)

    # merge similar columns
    gdf_edges = merge_similar_columns(gdf_edges, 'surface', '_30', newName='surface')
    gdf_edges = merge_similar_columns(gdf_edges, 'smoothness', '_40', newName='smoothness')
    gdf_edges = merge_similar_columns(gdf_edges, 'smoothness', '_36', newName='width')

    # add additional information
    gdf_nodes = add_traffic_lights(gdf_nodes, signals_fp)
    if cycle_path_width: # manual input file
        gdf_edges = add_cycle_path_width(gdf_edges, cyclePathW_fp)
    gdf_edges = add_bicycle_parking(gdf_edges, input_file=bikeAmenities_fp)
    if public_transport:
        gdf_edges = add_pt_stops(gdf_edges, input_file=ptStops_fp)
    else:
        gdf_edges['pt_stop_on'] = ''
        gdf_edges['pt_stop_count'] = ''
        gdf_edges['pt_stop_routes'] = ''

    # update idxs and add missing columns
    gdf_nodes, gdf_edges = update_idxs(gdf_nodes, gdf_edges)
    gdf_nodes, gdf_edges = add_missing_columns(gdf_nodes, gdf_edges)

    # save result
    save_p2_result_2_file(gdf_nodes, gdf_edges, output_file=p2_result_fp)



if __name__ == "__main__":
    main(accidents=False, cycle_path_width=False)

