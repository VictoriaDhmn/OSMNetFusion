"""
Functions and classes for p3_simplification.py

Functions
- addHighwayRank: Add a highway ranking (float) column to a pandas DataFrame of street edges.
- splitCurves: Splits curved edges in network into smaller, straighter segments based on the maximum angle between sub-segments.
- getHighestRankingRoadOfNode: returns the most important highway and its rank for each node, considering all edges connected to it.
- getGeomBuffered: Returns a list of polygon geometries that are buffered according to the highway type for each edge in the input DataFrame.
- getBufferIntersections: Takes a pandas.DataFrame of edges and finds all intersections between edges' geometry.
- clusterNodes: This function clusters nodes in a pandas DataFrame and returns the modified DataFrame.
- splitEdgeIfInNodeBuffer: Splits edges that intersect with the buffer of a node in the input GeoDataFrame and updates the nodes and edges GeoDataFrames accordingly.
- reassignNodes: This function updates the location of the "u" and "v" nodes for each edge, as well as the geometry of each edge.
- mergeNodes: Merges nodes based on the column 'merged_by' and returns a list of merged node objects.
- mergeEdgesWithSameNodes: Merges edges that have the same (reassigned) u and v (or v and u), and returns a list of merged link objects.
- removeDeg2Nodes: Removes nodes with degree 2 from the network.

Helper and parallel functions
- run_mergeEdgesWithSameNodes_in_parallel: Runs the mergeEdgesWithSameNodes function in parallel.
- run_splitCurves_in_parallel: Runs the splitCurves function in parallel.
- run_splitEdgeIfInNodeBuffer_in_parallel: Runs the splitEdgeIfInNodeBuffer function in parallel.

Helper functions
- clean: Cleans the input data by removing unnecessary columns and setting the index.
- getNodeDict: Returns a dictionary of nodes with their osmid as the key.
- listify: Converts a string of comma-separated values into a list of values.
- getEdgeDict: Returns a dictionary of edges with their osmid as the key.
- makePlot: Creates a plot of the input data.
- reverse_geom: Reverses the geometry of the input edge.
- getAngle: Returns the angle between two points.

Classes
- Node: Class for node objects.
- Link: Class for link objects.
- AnEdge: Class for edge objects.
- WalkEdge: Class for walk edge objects.
- BicycleEdge: Class for cycle edge objects.
- MotorizedEdge: Class for motorized edge objects.
"""

import ast
import math
import numpy as np
import shapely as sh
from shapely.ops import nearest_points,transform
import pandas as pd
import geopandas as gpd
import pyproj 
import matplotlib.pyplot as plt
import multiprocessing as mp
import psutil
import contextily as ctx
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import functools
import re

############################################
# MAIN FUNCTIONS
############################################

def addHighwayRank(edges,highway_ranking_custom=None,col='highway'):
    """
    Add a highway ranking (float) column to a pandas DataFrame of street edges.

    Inputs:
    - edges (pandas.DataFrame): DataFrame of street edges with a 'highway' column
    - highway_ranking_custom (dict, optional): Custom highway ranking to use instead of default
    - col (str, optional): Name of the column to use for ranking (default is 'highway')

    Returns:
    - pandas.Series: Series of highway rankings for each edge in `edges`
    """

    # make road type specific ranking - higher values will be considered first during the merging process
    if highway_ranking_custom is None:
        highway_ranking = {'trunk':10,
                'trunk_link':9.5,
                'primary':9,
                'secondary':9,
                'secondary_link':8.5,
                'tertiary':8,
                'residential':7,
                'cycleway':6,
                'path':5,
                'service':5,
                'footway':4,
                'pedestrian':4,
                'steps':3,
                'bridleway':3
        }
    else:
        highway_ranking = highway_ranking_custom 

    # map to 'highway', take max value
    ranks = edges.loc[:,col].apply(lambda x: highway_ranking[x] if x in highway_ranking.keys() else 0)
    return ranks

def splitCurves(edgesIn,nodesIn,maxAngleInitial=75,maxAnglePrev=60,addNodes=True):
    """
    Splits curved edges in network into smaller, straighter segments based on the maximum angle between sub-segments.

    Inputs:
    - edgesIn (GeoDataFrame): the input edges to split
    - nodesIn (GeoDataFrame): the node points corresponding to the input edges
    - idxStart (int): the starting index for new edge and node IDs
    - maxAngleInitial (float, optional): the maximum angle between the first and second segments of a curve to trigger a split (default 75)
    - maxAnglePrev (float, optional): the maximum angle between subsequent segments of a curve to trigger a split (default 60)
    - addNodes (bool, optional): whether to add new nodes at split points (default True)

    Returns:
    - edges (GeoDataFrame): the original input edges with any curved segments split into straighter segments
    - nodesNew (GeoDataFrame): the new nodes added at split points, if addNodes is True
    """
    
    # edgesIn remains unchanged
    edges = edgesIn.copy() # will be modified
    nodes = nodesIn #.copy() # will be modified

    crs = edges.crs
    baseIdx = 10000000
    while baseIdx<np.max([nodes.index.max(),edges.index.max()]):
        baseIdx*=100
        if baseIdx>100000000000000000000:
            print(xxx)
    if (baseIdx<np.max(edgesIn.index.values)) | (baseIdx<np.max(nodesIn.index.values)):
        print('ERROR: input edge/node indices are too high. Update the newEdgeIdx and newNodeIdx in the splitCurves function.')
        print(xxx)
    # make a transformation for later
    geod = pyproj.Geod(ellps="WGS84")

    allNewNodeIdcs = []
    allNewEdgeIdcs = []
    edgesNew = []
    nodesNew = []

    idcsToDrop = []
    for itr,row in edgesIn[['geometry']].iterrows():
        
        # check if there can be a curve i.e. > 2 segments
        if len(row.geometry.coords)==2:
            continue
        
        # else, do detailed check
        else:
            line = row.geometry
            newNodeIdcs = []
            
            # GATHER SPLIT INFORMATION
            coordslist = line.coords
            angleInitial = getAngle(coordslist[0],coordslist[1])   
            splits = []
            anglePrev = getAngle(coordslist[0],coordslist[1]) 
            for i in range(len(coordslist)-1):
                angle = getAngle(coordslist[i],coordslist[i+1])
                if abs(angle-angleInitial)>maxAngleInitial:
                    splits.append(i)
                    angleInitial = angle
                elif abs(angle-anglePrev)>maxAnglePrev:
                    splits.append(i)
                anglePrev = angle
            if len(splits)==0:
                continue

            # ADD EDGES
            # first segment
            newedgeidx = baseIdx+itr*100 
            newnodeidx = baseIdx+itr*100 
            edgeNew = edges.loc[itr].copy()
            edgeNew.name = newedgeidx
            edgeNew['geometry'] = sh.geometry.LineString(coordslist[:splits[0]+1]) # to include first split point
            edgeNew['v'] = newnodeidx 
            edgeNew['length'] = geod.geometry_length(edgeNew['geometry'])
            edgesNew.append(edgeNew)
            allNewEdgeIdcs.append(newedgeidx)
            newedgeidx+=1

            # middle segments
            if len(splits)>1:
                for i,split in enumerate(splits[:-1]):
                    edgeNew = edges.loc[itr].copy()
                    edgeNew.name = newedgeidx
                    edgeNew['geometry'] = sh.geometry.LineString(coordslist[split:splits[i+1]+1])  # to include last point of segment
                    edgeNew['u'] = newnodeidx
                    newNodeIdcs.append(newnodeidx)
                    newnodeidx+=1
                    edgeNew['v'] = newnodeidx
                    edgeNew['length'] = geod.geometry_length(edgeNew['geometry'])
                    edgesNew.append(edgeNew)
                    allNewEdgeIdcs.append(newedgeidx)
                    newedgeidx+=1
            
            # last segments
            edgeNew = edges.loc[itr].copy()
            edgeNew.name = newedgeidx
            edgeNew['geometry'] = sh.geometry.LineString(coordslist[splits[-1]:])
            edgeNew['u'] = newnodeidx
            newNodeIdcs.append(newnodeidx)
            newnodeidx+=1
            edgeNew['length'] = geod.geometry_length(edgeNew['geometry'])
            edgesNew.append(edgeNew)
            allNewEdgeIdcs.append(newedgeidx)
            newedgeidx+=1

            # DROP OLD GEOMETRY
            idcsToDrop.append(itr)
            
            # ADD NODES
            if not addNodes:
                continue
            cols = ['osmid','old_osmid','y','x','street_count','highway','crossing','bicycle','foot',\
                            'barrier','bicycle_parking','lit','width','public_transport']
            cols = [col for col in cols if col in nodes.columns]
            for i,split in enumerate(splits):
                newidx = newNodeIdcs[i] 
                nodeNew = nodes.iloc[0].copy()
                nodeNew.name = newidx
                nodeNew[cols] = None
                # nodeNew['highway_conn'] = edgeNew['highway'] # NOTE:EDIT
                nodeNew['geometry'] = sh.geometry.Point(coordslist[split])  # to include last point of segment
                nodeNew['osmid'] = newidx
                nodesNew.append(nodeNew)
            allNewNodeIdcs.extend(newNodeIdcs)
    
    if len(edgesNew)>0:
        edgesNew = pd.concat(edgesNew,axis=1).T
        nodesNew = gpd.GeoDataFrame(pd.concat(nodesNew,axis=1).T)
        edges = gpd.GeoDataFrame(pd.concat([edges,edgesNew]))
    else:
        nodesNew = nodes.loc[allNewNodeIdcs,:]
    edges.crs = crs
    nodesNew.crs = crs
    edges = edges[~edges.index.isin(idcsToDrop)]
    return edges,nodesNew 

def getHighestRankingRoadOfNode(nodesIn,edgesIn):
    """
    Returns the most important highway and its rank for each node, considering all edges connected to it.

    Inputs:
    - nodesIn (pd.DataFrame): DataFrame containing nodes data.
    - edgesIn (pd.DataFrame): DataFrame containing edges data.

    Returns:
    - Tuple[List[str], List[int]]: A tuple containing two lists - mostImportantHighway and mostImportantHighway_rank.
    """
    # take mean of first and second most important road
    mostImportantHighway = []
    mostImportantHighway_rank = []
    edges = pd.DataFrame(edgesIn,columns=edgesIn.columns)[['u','v','highway_rank','highway']]
    nodes = pd.DataFrame(nodesIn,columns=nodesIn.columns)
    nodes['nodeIdx'] = nodes.reset_index().index.values
    
    # find out which nodes are connected to which edges
    df1 = pd.merge(edges,nodes[['osmid','nodeIdx']], how='inner', left_on='u', right_on='osmid')
    df2 = pd.merge(edges,nodes[['osmid','nodeIdx']], how='inner', left_on='v', right_on='osmid')
    df = pd.concat([df1,df2]).drop_duplicates()
    
    # sort such that the most important road (highest highway_rank) is first 
    df = df.sort_values(by=['osmid','highway_rank'], ascending=[True,False])
    # select the first two for each osm id
    df_top2 = df.groupby('osmid').nth([0,1]).reset_index()

    # merge into one row - df highway is a string like "trunk,primary"
    # NOTE: need to do 'first' for highway (rather than lambda x: ','.join(x)), as this value will be used to buffer the nodes
    df = df_top2[['nodeIdx','highway_rank','highway','osmid']].groupby('osmid').agg({'nodeIdx':'first', 'highway_rank':'mean','highway': 'first'}).reset_index()
    
    # ensure same order as before
    df.sort_values(by='nodeIdx', inplace=True, ascending=True)
    
    mostImportantHighway = df['highway'].tolist()
    mostImportantHighway_rank = df['highway_rank'].tolist()
    return mostImportantHighway,mostImportantHighway_rank
    
def getGeomBuffered(edges,highway_buffers_custom=None):
    """
    Returns a list of polygon geometries that are buffered according to the highway type for each edge in the input DataFrame.

    Inputs:
    - edges (pd.DataFrame): A DataFrame containing edges with a geometry column representing the edge's geometry and a highway_conn column representing the type of highway.
    - highway_buffers_custom (dict, optional): A dictionary containing custom buffer distances for each highway type. If not provided, default buffer distances will be used.

    Returns:
    - List[Polygon]: A list of polygon geometries representing the buffered edges.
    """
    
    geom_orig = edges.geometry
    try:
        # assign meter-based crs
        crs = edges.crs
        edges = edges.to_crs("EPSG:3043")
    except AttributeError:
        print('ERROR: DataFrame has no crs. Please set a crs.')
        
    
    # make road type specific buffer - in meters - to either side
    if highway_buffers_custom is None:
        highway_buffers = {'trunk':20,
            'trunk_link':20,
            'primary':20, 
            'secondary':20, 
            'secondary_link':20,
            'tertiary':14,
            'residential':14,
            'cycleway':4,
            'path':4,
            'service':4,
            'footway':4,
            'pedestrian':4,
            'steps':4,
            'bridleway':4,
            'all_others': 4
        }
    else:
        highway_buffers = highway_buffers_custom 
    
    # do buffering
    buffered = []
    for _,row in edges.iterrows():
        buffered.append( row.geometry.buffer(highway_buffers[row.highway_conn] \
            if row.highway_conn in highway_buffers.keys() else highway_buffers['all_others']) )
    
    # convert back to initial crs
    edges.geometry = buffered
    edges = edges.to_crs(crs)
    buffered = edges.geometry
    edges.geometry = geom_orig
    return buffered

def getBufferIntersections(edges,col='geom_buffered'):
    """
    Takes a pandas.DataFrame of edges and finds all intersections between edges' geometry.
    NOTE: the values a row belong to 'osmid', not 'osmid_i'. 'osmid_i' belongs to the intersecting node.
    
    Inputs:
    - edges (pandas.DataFrame): the input DataFrame of edges
    - col (str): the name of the column to find intersections with. Default is 'geom_buffered'.

    Returns:
    - pandas.DataFrame: a DataFrame containing all intersections between edges' geometry
    """    
    
    edges['geometry_orig'] = edges.geometry

    # check if polygon crosses or contains an edge - faster - one row per intersection, thus gdf2 longer than gdf
    edges.geometry = edges[col]
    gdf_intersections = gpd.sjoin(edges[['osmid','geometry']], edges, predicate='intersects', how='right',lsuffix='i')
    gdf_intersections = gdf_intersections.drop(columns=['index_i']).rename(columns={'osmid_right':'osmid'})
    
    edges.geometry = edges.geometry_orig
    gdf_intersections.geometry = gdf_intersections.geometry_orig
    gdf_intersections = gdf_intersections.drop_duplicates()
    
    # the values of the data entry belong to 'osmid', not osmid_i
    # osmid_i belong to the intersecting node
    return gdf_intersections

def clusterNodes(nodes,again=False, clusterThreshold=40):
    """
    This function clusters nodes in a pandas DataFrame and returns the modified DataFrame. 
    The clustering is done by finding intersections between the buffered nodes and merging 
    them together based on their overlapping areas. The function first looks for intersections 
    between the original nodes, then between the buffered nodes. The resulting clustered nodes 
    have a new geometry and buffered geometry, as well as new columns indicating which nodes 
    were merged together.

    If the number of overlapping nodes is greater than the clusterThreshold, the function uses
    KMeans clustering to group the nodes into clusters based on their proximity to each other.
    The number of clusters is determined by dividing the number of overlapping nodes by the 
    clusterThreshold. This only applies to <2% of the resulting merged nodes.

    Inputs:
    - nodesIn: a pandas DataFrame containing node data
    
    Returns:
    - nodes: a modified pandas DataFrame with clustered nodes
    """
    
    # new geometries
    if again==False:
        nodes['geom_merged'] = nodes.geometry
        nodes['geom_buff_merged'] = nodes.geom_buffered
        nodes['merged_by'] = nodes.osmid.astype(int)
    nodes['merged'] = '' 

    # Check if more buffers overlap with merged buffer
    # Cluster again --> merge geom_buff_merged instead of geom_buffered

    # 1. Perform a spatial join to find overlapping areas
    nodes = nodes.reset_index(drop=True)
    nodes = nodes.set_geometry('geom_buffered')
    overlapping = gpd.sjoin(nodes, nodes, predicate="intersects",how='inner')

    # 2. Identify clusters of overlapping areas using connected components
    # ADJ. MATRIX - using SPARSE matrices
    rows = overlapping.index
    cols = overlapping['index_right']
    data = np.ones(len(rows), dtype=int)
    adjacency_matrix = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    # get connected components
    n_components, labels = connected_components(adjacency_matrix)

    # 3. Keep one polygon per cluster and merge their shapes into one big polygon
    bigIntersectionsCounter = 0
    for component_id in range(n_components):
        cluster_indices = np.where(labels == component_id)[0]
        areas = nodes.loc[cluster_indices, ['geom_buffered','merged','osmid','highway_rank','geometry_orig']]
        if len(areas)<2: 
            # only self, no overlapping buffer areas
            continue
        if len(cluster_indices)<clusterThreshold: 

            # remove/keep lines
            nodes.loc[areas.index,'merged'] = 'r' # remove
            nodes.loc[areas.index.values[0],'merged'] = 'k' # keep (set again, as itr included in areas)
            
            # find centroid of overlapping areas
            # union of node and its overlapping nodes
            overlappingArea = sh.ops.unary_union(areas['geom_buffered'])
            mostImportantAreas = areas[areas.highway_rank==areas.highway_rank.max()]
            nodes.loc[areas.index,'geom_merged'] = sh.ops.unary_union(mostImportantAreas['geom_buffered']).centroid
            nodes.loc[areas.index,'geom_buff_merged'] = overlappingArea
            nodes.loc[areas.index,'merged_by'] = int(areas.osmid.values[0])
        
        else:
            bigIntersectionsCounter+=1
            # USING KMEANS
            points = np.column_stack(([p.x for p in areas.geometry_orig], [p.y for p in areas.geometry_orig]))
            n = len(cluster_indices)//clusterThreshold + 1
            kmeans = KMeans(n_clusters=n, random_state=42)
            kmeans.fit(points)
            centers = kmeans.cluster_centers_
            # for each node, find the closest center
            dists = np.zeros((len(points),len(centers)))
            for i,center in enumerate(centers):
                center = np.array(center)
                dists[:,i] = np.linalg.norm(points-center,axis=1)
            closestCenter = np.argmin(dists,axis=1)
            # assign geom to the centroids using osmid
            for i,center in enumerate(centers):
                subareas = areas.iloc[np.where(closestCenter==i)[0]]
                nodes.loc[subareas.index,'merged'] = 'r'
                nodes.loc[subareas.index.values[0],'merged'] = 'k'
                # NOTE: TODO
                mostImportantAreas = subareas[subareas.highway_rank==subareas.highway_rank.max()]
                nodes.loc[subareas.index,'geom_merged'] = sh.ops.unary_union(mostImportantAreas['geom_buffered']).centroid
                # nodes.loc[subareas.index,'geom_merged'] = sh.ops.unary_union(subareas['geom_buffered']).centroid
                nodes.loc[subareas.index,'geom_buff_merged'] = sh.ops.unary_union(subareas['geom_buffered'])
                nodes.loc[subareas.index,'merged_by'] = int(subareas.osmid.values[0])
            if 'plotIntersections'=='NO_plotIntersections':
                # plot polygon and centroid and new centroids
                fig, ax = plt.subplots()
                gpd.GeoSeries(areas['geom_buffered']).plot(ax=ax,color='blue',alpha=0.2)
                gpd.GeoSeries(areas['geometry_orig']).plot(ax=ax,c=closestCenter)
                gpd.GeoSeries([sh.geometry.Point(center[0],center[1]) for center in centers]).plot(ax=ax,color='red')
                ctx.add_basemap(ax,crs='EPSG:4326',source=ctx.providers.OpenStreetMap.Mapnik)
                plt.title('Large cluster')
                plt.show()
    print('\tn:',len(nodes),'-->',len(nodes[nodes.merged=='k']),f'of which {bigIntersectionsCounter} are big intersections.')
    nodes = nodes.set_geometry('geometry')

    return nodes

def splitEdgeIfInNodeBuffer(edgesIn,nodesIn,allNodes=0,idxStart=0):
    """
    Splits edges that intersect with the buffer of a node in the input GeoDataFrame
    and updates the nodes and edges GeoDataFrames accordingly.

    Inputs
    - edgesIn (GeoDataFrame): Input GeoDataFrame representing the edges.
    - nodesIn (GeoDataFrame): Input GeoDataFrame representing the nodes.
    - allNodes (int, optional): Index of all nodes. Default is 0.
    - idxStart (int, optional): Starting index. Default is 0.

    Returns
    - updatedEdges (GeoDataFrame):  Updated GeoDataFrame representing the edges.
    - newEdges (GeoDataFrame): New GeoDataFrame representing the newly created edges.
    - newNodes (GeoDataFrame): New GeoDataFrame representing the newly created nodes.
    """

    geod = pyproj.Geod(ellps="WGS84")

    if type(allNodes)==int:
        allNodes = nodesIn.copy()

    # to not change the input data
    edges = edgesIn.copy()
    nodes = nodesIn.copy()
    
    # keep track of progress
    i = 10000000+10000000
    while i<np.max([nodes.index.max(),edges.index.max()]):
        i*=10
        if i>100000000000000000000:
            print(xxx)
    newedgeidx = int(i) 
    newnodeidx = int(i)
    counterSIPT = 0

    allNewNodeIdcs = []
    allNewEdgeIdcs = []
    allUpdatedEdgeIdcs = []
    edgesNew = []
    nodesNew = []

    # only consider each merged geometry once
    splitCounter = 0 + idxStart*1000 # NOTE: RECENT CHANGE
    unique_geom_buff_merged = nodesIn.geom_buff_merged.drop_duplicates().index
    for itr,n in nodesIn.loc[unique_geom_buff_merged,['geom_merged','geom_buff_merged','osmid']].iterrows():
        for i,e in edgesIn[edgesIn.intersects(n.geom_buff_merged)].iterrows():
            # if neither end of the edge is within the merged buffer
            if e.geometry.coords[0]==e.geometry.coords[-1]:
                # geometry is a linear ring / circle
                continue
            start,end = e.geometry.boundary.geoms
            if not ( (start.within(n.geom_buff_merged)) | (end.within(n.geom_buff_merged)) ):
                p5,_ = nearest_points(e.geometry, n.geom_merged) 
                if not ( (p5==start) | (p5==end) ):
                    seg1,buff,seg2 = [a for a in sh.ops.split(e.geometry,p5.buffer(0.000000001)).geoms]
                    # calculate the distance in meters between the point "start" and the linestrings "seg1" and "seg2"
                    # if the distance between "start" and "seg1" is smaller than the distance between "start" and "seg2", then the correct order is seg1,seg2
                    l = sh.geometry.LineString(list(seg1.coords) + list(p5.coords) + list(seg2.coords))
                    seg1,seg2 = [a for a in sh.ops.split(l,p5).geoms]
                    newEIdx = newedgeidx+splitCounter # + i
                    newNIdx = newnodeidx+splitCounter # + i
                    splitCounter+=1
                    counterSIPT+=1
                    edgeNew = edges.loc[i].copy()
                    edgeNew.name = newEIdx
                    start = allNodes[allNodes.osmid==e.u].geometry.values[0]
                    correctOrder = seg1.distance(start)<seg2.distance(start)
                    edges.loc[i,'geometry'] = seg1 if correctOrder else seg2
                    edges.loc[i,'length'] = geod.geometry_length(seg1 if correctOrder else seg2) # seg1.length
                    edges.loc[i,'v'] = newNIdx # n['osmid']
                    edgeNew['geometry'] = seg2 if correctOrder else seg1
                    edgeNew['length'] = geod.geometry_length(seg2 if correctOrder else seg1) # seg2.length
                    edgeNew['u'] = newNIdx # n['osmid']
                    allNewEdgeIdcs.append(newEIdx)
                    allUpdatedEdgeIdcs.append(i)
                    edgesNew.append(edgeNew)

                    # ADD NODE
                    nodeNew = nodes.loc[itr].copy()
                    nodeNew.name = newNIdx
                    nodeNew['osmid'] = newNIdx
                    nodeNew['geometry_orig'] = sh.geometry.Point(seg1.coords[-1])
                    nodeNew['geometry'] = nodeNew['geometry_orig']
                    nodesNew.append(nodeNew)
                    allNewNodeIdcs.append(newNIdx)
                    # geom_buffered must be updated as a whole
                    # geom_merged, geom_buff_merged, merged_by, merged must be updated by reclustering
    
    edgesNew = pd.concat(edgesNew,axis=1).T
    nodesNew = pd.concat(nodesNew,axis=1).T
    nodes = gpd.GeoDataFrame(nodes,crs=nodesIn.crs,geometry='geometry')
    return edges.loc[allUpdatedEdgeIdcs,:],edgesNew,nodesNew 

def reassignNodes(edges,mergeIdx):
    """
    This function updates the location of the "u" and "v" nodes for each edge, as well as the geometry of each edge. The updated 
    geometry consists of a simplified version of the original geometry, consisting only of the line connecting the new "u" and "v" 
    nodes.

    Takes two inputs:
    - edges: a pandas DataFrame containing information about edges in a graph, such as the nodes they connect and their geometry.
    - mergeIdx: a pandas DataFrame containing information about nodes that have been merged, including their new location.

    Returns a pandas DataFrame with four columns:
    - new_u: the new location of the "u" node for each edge, based on the information in mergeIdx.
    - new_v: the new location of the "v" node for each edge, based on the information in mergeIdx.
    - geom_reassigned: the geometry of each edge, with the nodes reassigned based on the information in mergeIdx.
    - geom_linear: a simplified version of the geometry for each edge, consisting only of the line connecting the new "u" and "v" nodes.
    """
    geod = pyproj.Geod(ellps="WGS84")

    edges.loc[:,'new_u'] = edges.u.map(lambda x: int(mergeIdx.loc[x,'merged_by']) if x in mergeIdx.index else x)
    edges.loc[:,'new_v'] = edges.v.map(lambda x: int(mergeIdx.loc[x,'merged_by']) if x in mergeIdx.index else x)
    edges.loc[:,'geom_reassigned'] = edges.geometry
    edges.loc[:,'geom_linear'] = edges.geometry.map(lambda a: sh.geometry.LineString([a.coords[0],a.coords[-1]]))
    edges.loc[:,'length'] = edges.geom_linear.map(lambda x: geod.geometry_length(x))
    
    new_u_pts = mergeIdx.loc[edges.u,'geom_merged'].values
    new_v_pts = mergeIdx.loc[edges.v,'geom_merged'].values
    mids = []
    for itr,row in edges.iterrows():
        # mid = row.geometry.coords[:]
        if (row.u==row.new_u):
            mid = row.geometry.coords[1:]
        elif (row.v==row.new_v):
            mid = row.geometry.coords[:-1]
        elif (row.v==row.new_v) & (row.u==row.new_u):
            mid = row.geometry.coords[1:-1]
        else:
            mid = row.geometry.coords[:]
        mids.append(mid)

    ################################
    # IDEA: only keep the points closest to new_u and new_v, and those in between
    
    # near_new_u = [sh.nearest_points(sh.geometry.Point(new_u_pts[i]),sh.geometry.LineString(mids[i]))[0] for i in range(len(edges))]
    # near_new_v = [sh.nearest_points(sh.geometry.Point(new_v_pts[i]),sh.geometry.LineString(mids[i]))[0] for i in range(len(edges))]
    # interpolate where the new_u and new_v points are on the line
    # between the two points
    # near_new_u = [sh.geometry.LineString([mids[i][0],mids[i][-1]]).interpolate(near_new_u[i].distance(sh.geometry.LineString([mids[i][0],mids[i][-1]]))) for i in range(len(edges))]
    # near_new_v = [sh.geometry.LineString([mids[i][0],mids[i][-1]]).interpolate(near_new_v[i].distance(sh.geometry.LineString([mids[i][0],mids[i][-1]]))) for i in range(len(edges))]
    # only keep the points in between near_new_u and near_new_v for each line in mids
    
    # find the coordinate (or index) in mids[i] that is closest to new_u_pts[i]
    # and new_v_pts[i], and keep only those points in between --> mids[i][idx1:idx2+1]
    near_new_u = [min(mids[i], key=lambda x: new_u_pts[i].distance(sh.geometry.Point(x))) for i in range(len(edges))]
    near_new_v = [min(mids[i], key=lambda x: new_v_pts[i].distance(sh.geometry.Point(x))) for i in range(len(edges))]
    idx1 = [mids[i].index(near_new_u[i]) for i in range(len(edges))]
    idx2 = [mids[i].index(near_new_v[i]) for i in range(len(edges))]
    
    # keep only the points in between
    print('Average len of mids:',np.mean([len(mids[i]) for i in range(len(edges))]))
    mids = [mids[i][idx1[i]:idx2[i]+1] for i in range(len(edges))]
    print('Average len of mids:',np.mean([len(mids[i]) for i in range(len(edges))]))
    
    #################################

    edges['geom_linear'] = [sh.geometry.LineString(new_u_pts[i].coords[:] + new_v_pts[i].coords[:]) for i in range(len(edges))]
    initReassigned = [sh.geometry.LineString(new_u_pts[i].coords[:] + mids[i] + new_v_pts[i].coords[:]) for i in range(len(edges))]
    reverseReassigned = [sh.geometry.LineString(new_u_pts[i].coords[:] + list(reversed(mids[i])) + new_v_pts[i].coords[:]) for i in range(len(edges))]
    correctOrder = [geod.geometry_length(initReassigned[i])<geod.geometry_length(reverseReassigned[i]) for i in range(len(edges))]
    edges['geom_reassigned'] = [initReassigned[i] if correctOrder[i] else reverseReassigned[i] for i in range(len(edges))]
    # edges['geom_reassigned'] = [sh.geometry.LineString(new_u_pts[i].coords[:] + mids[i] + new_v_pts[i].coords[:]) for i in range(len(edges))]
    edges['length'] = edges.geom_linear.map(geod.geometry_length)

    return edges[['new_u','new_v','geom_reassigned','geom_linear']]

def mergeNodes(nodes): 
    """
    Merges nodes based on the column 'merged_by' and returns a list of merged node objects.

    Inputs:
    - nodes (pd.DataFrame): a pandas DataFrame containing node data, including osmid, merged_by, and geom_merged columns

    Returns:
    - gatherNodeObjects (list): a list of merged node objects
    """
    
    mergeIdx = nodes[['osmid','merged_by','geom_merged','old_osmid','highway_conn','highway_rank','crossing']].drop_duplicates(subset=['osmid','merged_by','geom_merged'])
    nodes['merged'] = ''
    gatherNodeObjects = []
    
    for itr,row in nodes.loc[:,:].iterrows(): 
    
        if nodes.loc[itr,'merged'] == 'r':
            continue
        else:
            nodes.loc[itr,'merged'] = 'k' # keep
        
        # find edges intersecting the buffer zone of the considered edge
        lines = mergeIdx[mergeIdx.merged_by==row.osmid]
        if len(lines)<2: # ==0
            nodesToMerge = None
            #continue
        else:
            nodesToMerge = []
            for i,r in lines.iterrows():
                if nodes.loc[i,'merged'] == 'k': # or == 'r'?
                    continue
                nodesToMerge.append(r)
            nodesToMerge = pd.DataFrame(nodesToMerge)
            
            for i in nodesToMerge.index.values:
                nodes.loc[i,'merged'] = 'r' # remove
                nodes.loc[i,'merged_by'] = itr 

        # merge tags and stuff
        mergedNode = Node(nodesToMerge,row)
        
        gatherNodeObjects.append(mergedNode)

    return gatherNodeObjects #, referenceDict

def mergeEdgesWithSameNodes(edges): 
    """
    Merges edges that have the same (reassigned) u and v (or v and u), and returns a list of merged link objects.

    Inputs:
    - edges (pd.DataFrame): a pandas DataFrame containing edge data

    Returns:
    - gatherLinkObjects (list): a list of merged link objects
    """
    
    edges = edges.sort_values(by='highway_rank',ascending=False)
    edges['merged'] = ''
    gatherLinkObjects = []
    deleted_edges = [] # some edges are simply deleted
    
    for itr,row in edges.loc[:,:].iterrows(): 
    
        # disregard edges that start and end at the same node
        if row.new_u==row.new_v:
            # NOTE: SOME EDGES ARE REMOVED HERE
            # drop edges where newu==newv (very short segments where both ends were merged to the same new cluster/node)
            edges.loc[itr,'merged'] == 'r'
            deleted_edges.append(row.osmid)
        if edges.loc[itr,'merged'] == 'r':
            continue
        else:
            edges.loc[itr,'merged'] = 'k' # keep
        
        # find edges intersecting the buffer zone of the considered edge
        lines = edges[((edges.new_u==row.new_u) & (edges.new_v==row.new_v)) | \
                      ((edges.new_u==row.new_v) & (edges.new_v==row.new_u))]
        if len(lines)<2: # ==0
            linesToMerge = None
        else:
            linesToMerge = []
            for i,r in lines.iterrows():
                if edges.loc[i,'merged'] == 'k': # or == 'r'?
                    continue
                # disregard edges that stat and end at the same node
                if r.new_u==r.new_v:
                    edges.loc[i,'merged'] == 'r'
                    deleted_edges.append(r.osmid)
                    continue
                # NOTE: check for length, as 2 roads like this 'D' shouldn't be merged
                l1, l2 = row['length'], r['length']
                if (l1<0.0000001) | (l2<0.0000001):
                    linesToMerge.append(r)
                elif (max(l1/l2,l2/l1) < 1.5):
                    linesToMerge.append(r)
            if len(linesToMerge)<1:
                continue
            linesToMerge = pd.DataFrame(linesToMerge)
            
            # NOTE: SOME EDGES ARE REMOVED HERE
            # drop edges where newu==newv (very short segments where both ends were merged to a new cluster)
            # to fix 'zigzag' issue e.g. at Giselastr. or near Freiheit
            # skip if u==v (i.e. previously on osm, maybe a roundabout)
            if (len(linesToMerge[linesToMerge.new_u!=linesToMerge.new_v])==0) & \
                (len(linesToMerge[linesToMerge.u==linesToMerge.v])==0) & \
                (row.new_u==row.new_v) & \
                (row.u!=row.v):
                edges.loc[itr,'merged'] = 'r'
                deleted_edges.append(row.osmid)
                continue

            for i in linesToMerge.index.values:
                edges.loc[i,'merged'] = 'r' # remove
                edges.loc[i,'merged_by'] = itr 
        
        # merge tags and stuff
        mergedLink = Link(linesToMerge,row)
        if (mergedLink.edgeUV is None) & (mergedLink.edgeVU is None):
            print('Something is wrong here! UV and VU are None. Idx/u/v/length:',row.name,row.new_u,row.new_v,row['length'])
            continue
        gatherLinkObjects.append(mergedLink)

    return gatherLinkObjects, set(deleted_edges)
 
# UNUSED
def removeDeg2Nodes(edgesIn,nodesIn):
    """
    Removes nodes with degree 2 and merges edges with the same approach angle and highway type.
    NOTE: This function is not used in the current implementation.
    NOTE: This function is currently erronerous.

    Inputs:
    - edgesIn (pd.DataFrame): a pandas DataFrame containing edge data
    - nodesIn (pd.DataFrame): a pandas DataFrame containing node data

    Returns:
    - edges (pd.DataFrame): a pandas DataFrame containing updated edge data
    - nodes (pd.DataFrame): a pandas DataFrame containing updated node data
    """

    nodes = nodesIn.copy()
    edges = edgesIn.copy()
    nodesToDrop = []
    edgesToDrop = []

    for itr,row in nodes.iterrows():

        # check if node has degree == 2
        us = edges[edges.u==row.osmid]
        vs = edges[edges.v==row.osmid]
        segments = pd.concat([us,vs])
        toNodes = set(list(segments.u.values)+list(segments.v.values))
        if not len(toNodes)==3: # self, plus two other nodes
            continue
        
        # if degree == 2
        # if same highway type
        if len(segments.highway_conn.unique())!=1: # .explode().unique()
            continue

        # if oneway to and fro, merge as 'reverse False' and 'reverse True'
        # do later... part of the tag restructuring

        # merge links if they have the same approach angle
        # e.g. idx 241 and 244
        # check if angle similar
        toNodes.remove(row.osmid)
        node1,node3 = toNodes
        for a,c in [[node1,node3],[node3,node1]]:
            subset = segments[(segments.u==a) | (segments.v==c)]
            if len(subset)<2:
                continue
            # should have two rows now
            colvals = []
            for i,col in enumerate(segments.columns):
                if col=='u':
                    colvals.append(a)
                elif col=='v':
                    colvals.append(c)
                elif col=='length':
                    colvals.append(float(subset.iloc[0,i])+float(subset.iloc[1,i]))
                elif col in ['geometry','geom_reassigned','geom_linear']:                    
                    # merge lines
                    if subset.iloc[0,i].type=='MultiLineString':
                        # NOTE: FIX THIS ------------------------------------------
                        subset.iloc[0,i] = subset.iloc[0,i].geoms[0]
                    if subset.iloc[1,i].type=='MultiLineString':
                        # NOTE: FIX THIS ------------------------------------------
                        subset.iloc[1,i] = subset.iloc[1,i].geoms[0]
                    multiLine = sh.geometry.MultiLineString([subset.iloc[0,i],subset.iloc[1,i]])
                    mergedLine = sh.ops.linemerge(multiLine)
                    # if lines are the same, no need to merge
                    if mergedLine.type=='GeometryCollection':
                        mergedLine = subset.iloc[0,i]
                    colvals.append(mergedLine)
                else:
                    colvals.append(subset.iloc[0,i] if subset.iloc[0,i]==subset.iloc[1,i] else str([subset.iloc[0,i],subset.iloc[1,i]]) )
            
            # need to take care of previously merged tags (will be str)
            # or to not loose them   

            # update edge 1
            edges.loc[subset.index[0],:] = colvals
            # remove edge 2
            edgesToDrop.append(subset.index[1])
            
        # check if node now removed
        us = edges[edges.u==row.osmid]
        vs = edges[edges.v==row.osmid]
        segments = pd.concat([us,vs])
        toNodes = set(list(segments.u.values)+list(segments.v.values))
        #if len(segments)>2:
        if len(toNodes)>3: # self, plus two other nodes
            print('ERROR. Node degree:',len(toNodes),'Node:',row.osmid)
        else:
            nodesToDrop.append(itr)
            
    nodes = nodes.drop(nodesToDrop)
    edges = edges.drop(edgesToDrop)
    
    return edges,nodes
   

############################################
# PARALLELIZATION FUNCTIONS
############################################

def run_mergeEdgesWithSameNodes_in_parallel(df):
    """
    This function takes a pandas dataframe as input, splits it into multiple dataframes based on the sum of the u and v columns,
    such that each u-v pair is in only of the resulting splits. It then applies the mergeEdgesWithSameNodes function to each of 
    the split dataframes in parallel using multiprocessing. It returns a list of results.

    Inputs:
    - df: pandas dataframe

    Returns:
    - results: list
    """
    # 1. Create a Pool of processes
    logical = False
    num_processes  = psutil.cpu_count(logical=logical)-2
    pool = mp.Pool(processes=num_processes)
    # 2. Split the dataframes into equal parts for each process
    #    - later we filter by u=v and u=u, so need to consider both...
    tmp = df[['new_u','new_v']].copy()
    tmp['sum'] = tmp.new_u+tmp.new_v
    tmp = tmp.sort_values(by='sum')
    split_vals = [tmp.iloc[int(len(tmp)*q/num_processes)-1,:]['sum'] for q in range(1,num_processes+1)]
    split_df = []
    iprev = -1
    for i in split_vals:
        split_df.append(df.loc[tmp[(tmp['sum']>iprev)&(tmp['sum']<=i)].index.values,:])
        iprev = i
    assert len(df)==np.sum([len(a) for a in split_df])
    # 3. Apply the update_df function to each pair of dataframes in parallel
    result_link_objects = []
    result_deleted_edges = []
    print('\tNumber of processors used:',len(pool._pool))
    for links,delEdges in pool.map(mergeEdgesWithSameNodes, split_df):
        result_link_objects.extend(links)
        result_deleted_edges.extend(delEdges)
    # 4. Combine the results back into a single dataframe
    pool.close()
    pool.join()
    return result_link_objects,result_deleted_edges

def run_splitCurves_in_parallel(df, nodes, maxAngleInitial=75,maxAnglePrev=60):
    """
    This function splits a dataframe into multiple chunks and runs the splitCurves function in parallel on each chunk.

    Inputs:
    - df (pandas.DataFrame): The dataframe to be split
    - nodes (list): List of nodes to be passed to the splitCurves function

    Returns:
    - tuple: A tuple containing two lists - es and new_ns
    """
    # 1. Create a Pool of processes
    logical = False
    num_processes  = psutil.cpu_count(logical=logical)-2
    pool = mp.Pool(processes=num_processes)
    # 2. Split the dataframes into equal parts for each process
    split_df = np.array_split(df, num_processes)
    nodes_np = np.array_split(nodes, 1)
    # 3. Apply the update_df function to each pair of dataframes in parallel
    results = []
    print('\tNumber of processors used:',len(pool._pool))
    for result in pool.starmap(splitCurves, zip(split_df, nodes_np*len(split_df), [maxAngleInitial]*len(split_df), [maxAnglePrev]*len(split_df))):
        results.append(result)
    # 4. Combine the results back into a single dataframe
    es = [a[0] for a in results]
    new_ns = [a[1] for a in results]
    pool.close()
    pool.join()
    return es,new_ns

def run_splitEdgeIfInNodeBuffer_in_parallel(edges, nodes):
    """
    This function splits the node dataframe into multiple chunks and runs the splitEdgeIfInNodeBuffer function 
    in parallel on each chunk. Each unique merged node geometry is only in one of these chunks. 
    
    If an edge is split in more than one of these parallel processes, it is reprocessed i.e. split under 
    consideration of all merged node geometries in a sequential manner. The new edges created from this process are added to the 
    new_es DataFrame, while the previously created new edges are dropped.
    
    Inputs:
    - edges (pd.DataFrame): DataFrame of edge data.
    - nodes (pd.DataFrame): DataFrame of node data.

    Returns:
    - Tuple (pd.DataFrame, pd.DataFrame, pd.DataFrame, List[int]): A tuple containing the updated edge 
    dataframe, the new edge dataframe, the new node dataframe, and a list of edges split multiple times.
    """
    # 1. Create a Pool of processes
    logical = False
    num_processes  = psutil.cpu_count(logical=logical)-2
    pool = mp.Pool(processes=num_processes)
    # 2. Split the dataframes into equal parts for each process
    # Create an empty dictionary to store the arrays
    unique_node_geoms = nodes.geom_buff_merged.unique()
    unique_splits = np.array_split(unique_node_geoms, num_processes)
    split_df = [nodes[nodes.geom_buff_merged.isin(split)] for split in unique_splits]
    edges['tmp_idx_init'] = edges.index.values
    edgesnp = np.array_split(edges, 1)
    nodesnp = np.array_split(nodes, 1) # for reference only
    # 3. Apply the update_df function to each pair of dataframes in parallel
    results = []
    print('\tNumber of processors used:',len(pool._pool))
    for result in pool.starmap(splitEdgeIfInNodeBuffer, zip(edgesnp*len(split_df), split_df, nodesnp*len(split_df), range(len(split_df)))):
        results.append(result)
    # 4. Combine the results back into a single dataframe
    updated_es = [a[0] for a in results]
    new_es = [a[1] for a in results]
    new_ns = [a[2] for a in results]
    
    
    ##########################################
    # Deal with edges that were split multiple times
    ##########################################
    geod = pyproj.Geod(ellps="WGS84")
    updated_idcs = []
    for u in updated_es:
        updated_idcs.extend(u.index.values)
    from collections import Counter
    duplicate_updates = [elem for elem, count in Counter(updated_idcs).items() if count>1]
    print('\tEdges split multiple times:',len(duplicate_updates))
    updated_es = pd.concat(updated_es)
    new_es = pd.concat(new_es)
    new_ns = pd.concat(new_ns)
    new_es_2ndIteration = []
    newedgeidx = new_es.index.values.max()+1
    nodes_unique = nodes.loc[nodes.geom_buff_merged.drop_duplicates(keep='first').index.values]
    for itr in duplicate_updates:
        # find split points based on buffer nodes
        e = edges.loc[itr,:]
        intersection_idcs = gpd.GeoSeries(nodes_unique.geom_buff_merged).intersects(e.geometry)
        split_points = nodes_unique.loc[intersection_idcs,'geom_merged'].values
        split_points_label = nodes_unique.loc[intersection_idcs,'osmid'].values
        # clostest_to_line = [...]
        # geometry does not consider uv vs vu... so need to manually set geometry
        start = nodes[nodes.osmid==e.u].geometry.values[0]
        end = nodes[nodes.osmid==e.v].geometry.values[0]
        geom = sh.geometry.LineString([start,end])
        distances = [geom.project(point) for point in split_points] # 1st = u, last = v
        sort_indices = sorted(range(len(distances)), key=lambda k: distances[k],reverse=False)
        split_points = [split_points[i] for i in sort_indices]
        split_points_label = [split_points_label[i] for i in sort_indices]
        # ADD EDGES
        # first segment
        for i,split in enumerate(split_points[:-1]):
            new_e = e.copy()
            e.name = newedgeidx
            new_e['geometry'] = sh.geometry.LineString(split_points[i:i+2])  # to include last point of segment
            new_e['u'] = split_points_label[i] 
            new_e['v'] = split_points_label[i+1] 
            new_e['tmp_idx_init'] = np.nan
            new_e['length'] = geod.geometry_length(new_e['geometry'])
            new_es_2ndIteration.append(new_e)
            newedgeidx+=1
        if 'plot'=='plotNO':
            plt.plot(*e.geometry.coords.xy)
            if len(updated_es.loc[itr,:])>1:
                for iu,u in updated_es.loc[itr,:].iterrows():
                    plt.plot(*u.geometry.coords.xy)
            else:
                plt.plot(*updated_es.loc[itr,:].geometry.coords.xy)
            if len(new_es[new_es.tmp_idx_init==itr])>1:
                for iu,u in new_es[new_es.tmp_idx_init==itr].iterrows():
                    plt.plot(*u.geometry.coords.xy)
            else:
                plt.plot(*new_es[new_es.tmp_idx_init==itr].geometry.coords.xy)
            plt.scatter([a.x for a in split_points],[a.y for a in split_points])
            plt.show()
            print(new_es[['u','v','geometry']])
    if len(new_es_2ndIteration)>0:
        new_es_2ndIteration = pd.concat(new_es_2ndIteration,axis=1).T
        new_es = pd.concat([new_es,new_es_2ndIteration])
    # ---- added 2 lines -----
    remove_prev_added_nodes = sum( new_es[new_es.tmp_idx_init.isin(duplicate_updates)][['u','v']].values.tolist(), [])
    new_ns = new_ns[~new_ns.osmid.isin(remove_prev_added_nodes)]
    updated_es = updated_es[~updated_es.tmp_idx_init.isin(duplicate_updates)]
    new_es = new_es[~new_es.tmp_idx_init.isin(duplicate_updates)]
    updated_es = updated_es.drop(columns=['tmp_idx_init'])
    new_es = new_es.drop(columns=['tmp_idx_init'])

    pool.close()
    pool.join()
    return updated_es,new_es,new_ns,duplicate_updates 


############################################
# HELPER FUNCTIONS
############################################

def clean(a,asFloat=False,keep='all'):
    """
    Cleans input data and returns it in a standardized format.

    Inputs:
    - a (Union[str, float, int, List[Union[str, float, int]]]): The input data to be cleaned.
    - asFloat (bool): If True, converts all valid numeric strings to float.
    - keep (str): If 'all', returns all cleaned data. If 'min', 'mean', or 'max', returns the corresponding statistic.

    Returns:
    - Union[None, float, List[Union[None, float, str]]]: Returns cleaned data as a float, list of floats, or list of strings.
    """
    # TODO: make this function nicer and neater

    l = np.nan
    if type(a)!=list:
        if (a=='') | (pd.isna(a)):
            l = np.nan
        elif (type(a)==float) | (type(a)==int) | (type(a)==np.float64) | (type(a)==np.int64):
            l = float(a) if asFloat else a
        elif (a[0]=="["):
            tmp = a[1:-1].split(", ")
            tmp2 = [b[1:-1] if b[0]=="'" else b for b in tmp]
            #l = [float(b) if asFloat else b for b in tmp2]
            l = []
            for b in tmp2:
                b_isNum = False
                if (type(b)==float) | (type(b)==int) | (type(b)==np.float64) | (type(b)==np.int64):
                    b_isNum = True
                if type(b)==str:
                    b_isNum = b.replace(',','').replace('.','').isdigit()
                if asFloat & b_isNum:
                    l.append(float(b))
                elif asFloat & (not b_isNum):
                    continue
                elif b=='':
                    continue
                else:
                    l.append(float(b) if b_isNum else b)
        elif (a[1:-1].isnumeric()):
            l = float(a[1:-1]) if asFloat else a[1:-1]
        else:
            try: # error for: '1.35;3.6'
                l = float(a) if asFloat else a
            except:
                l = np.nan
    else:
        l = []
        for b in a:
            b_isNum = False
            if (type(b)==float) | (type(b)==int) | (type(b)==np.float64) | (type(b)==np.int64):
                b_isNum = True
            if type(b)==str:
                b_isNum = b.replace(',','').replace('.','').isdigit()
            if asFloat & b_isNum:
                l.append(float(b))
            elif asFloat & (not b_isNum):
                continue
            elif b=='':
                continue
            else:
                l.append(float(b) if b_isNum else b)
    
    if (type(l)==list):
        if len(l)==0:
            l = np.nan
        elif len(l)==1:
            l = l[0]
        
    if (keep!='all'):
        if (asFloat==False):
            print('Min/Mean/Max. only applicable if asFloat==True.')
        else:
            if keep=='min':
                return np.min(l) if type(l)==list else l
            elif keep=='mean':
                return np.mean(l) if type(l)==list else l
            elif keep=='max':
                return np.max(l) if type(l)==list else l
    return l

def getNodeDict(nodes):
    """
    Create a GeoDataFrame object from a list of node objects.

    Input:
    - nodes (list[class.Node()]): A list of nodes to convert to a GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: The resulting GeoDataFrame object.
    """
    es = []
    for l in nodes:
        es.append(l.to_dict())
    keys = sorted(list(set([e for e in es for e in e])))
    d = {}
    for e in es:
        for k in keys:
            if k in d:
                d[k].append(e[k] if k in e else None)
            else:
                d[k] = [e[k]] if k in e else [None]
    df = pd.DataFrame.from_dict(d)
    return gpd.GeoDataFrame(df,geometry='g_geometry')

def listify(theList,asInt=False):
    """
    Convert a pandas Series into a list of flattened, unique elements

    Args:
    - theList (pd.Series): a pandas Series containing the data to be converted into a list
    - asInt (bool): a boolean flag indicating whether to convert all elements to integers

    Returns:
    - l (list): a list of the flattened, unique elements from the input Series
    """
    # takes a pandas series
    l = []
    for a in theList.explode().unique(): 
        if type(a)==list:
            l.extend(a)
        if (a=='') | (pd.isna(a)): # | (np.isnan(a))
            continue
        elif (type(a)==float) | (type(a)==int) | (type(a)==np.float64) | (type(a)==np.int64):
            l.append(a)
        elif type(a)==str:
            tmp = re.sub(r"[^\w,]", "", a).split(',')
            tmp = [b for b in tmp if b!='']
            l.extend(tmp)
        else:
            l.append(a)
    if len(l)==0:
        return l
    if asInt==True:
        l = [int(a) for a in l] # [int(a) if a!=0 else None for a in l]
    l = sorted(list(set(l)))
    return l

def getEdgeDict(links, modes=['walk','bike','motorized']):
    """
    This function takes in a list of link objects and optional modes, and returns a GeoDataFrame containing all edges 
    that are accessible to the specified modes. The modes parameter defaults to ['walk', 'bike', and 'motorized'] if not 
    specified, i.e. all available modes.

    Inputs:
    - links (list[class.Link()]): a list of link objects
    - modes (list, optional): a list of modes to filter on

    Returns:
    - gpd.GeoDataFrame: a GeoDataFrame containing edges accessible to the provided modes
    """
    es = []
    for l in links:
        if l.edgeUV is not None:
            if any([getattr(l.edgeUV, 'access_%s'%(m)) for m in modes]):
                es.append(l.edgeUV.to_dict(listOfModes=modes))
        if l.edgeVU is not None:
            if any([getattr(l.edgeVU, 'access_%s'%(m)) for m in modes]):
                es.append(l.edgeVU.to_dict(listOfModes=modes))
    keys = sorted(list(set([e for e in es for e in e])))
    
    d = {}
    for e in es:
        for k in keys:
            if k in d:
                d[k].append(e[k] if k in e else None)
            else:
                d[k] = [e[k]] if k in e else [None]
    df = pd.DataFrame.from_dict(d)
    return gpd.GeoDataFrame(df,geometry='g_geo_lin') # g_geo_reassigned

def make_plot(edges,nodes,edgeColor='white',basemap=True,title=None,bounds=None, edge_col=None, node_col=None):
    """
    This function takes in edges and nodes data and plots them on a graph.

    Inputs:
    - edges (pd.DataFrame): dataframe containing edges data.
    - nodes (pd.DataFrame): dataframe containing nodes data.
    - edgeColor (str): color of the edges on the plot.
    - title (str): title of the plot.
    """
    if basemap:
        edgeColor = 'black'
    if bounds is not None:
        try:
            edges_subarea = edges.cx[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        except:
            edges_subarea = gpd.GeoDataFrame(edges,geometry=edge_col).cx[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        try:
            nodes_subarea = nodes.cx[bounds[0]:bounds[1],bounds[2]:bounds[3]]
        except:
            nodes_subarea = gpd.GeoDataFrame(nodes,geometry=node_col).cx[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    else:
        edges_subarea = gpd.GeoDataFrame(geometry=edges)
        nodes_subarea = gpd.GeoDataFrame(geometry=nodes)
    ax = edges_subarea.plot(figsize=(10,10),color=edgeColor,linewidth=0.3)
    try: # regular nodes
        plt.scatter([a.x for a in nodes_subarea],[a.y for a in nodes_subarea],color='red',s=6)
    except: # for buffered nodes
        nodes_subarea.plot(ax=ax,color='red',linewidth=0.3,markersize=7)
    plt.title(title); ax.axes.get_xaxis().set_visible(False); ax.axes.get_yaxis().set_visible(False); ax.axis('equal')
    if basemap:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)   
    
    # TESTING
    if title=='After step 8 - Linear':
        for i, txt in edges_subarea.iterrows():
            ax.annotate(int(txt.length), (edges_subarea.loc[i,edge_col].centroid.x, edges_subarea.loc[i,edge_col].centroid.y),color='darkgreen')     
    if title=='node_importance':
        for i, txt in nodes_subarea.iterrows():
            ax.annotate(txt.highway_rank, (nodes_subarea.loc[i,node_col].centroid.x, nodes_subarea.loc[i,node_col].centroid.y),color='darkgreen', fontsize=6)
        for i, txt in edges_subarea.iterrows():
            ax.annotate(int(txt.highway_rank), (edges_subarea.loc[i,edge_col].centroid.x, edges_subarea.loc[i,edge_col].centroid.y),color='blue', fontsize=6)
    plt.show()

def reverse_geom(geom):
    """
    Reverses the order of the x and y inputs of a given geometry object.

    Inputs:
    - geom (shapely goemetry): a geometry object to be reversed

    Returns:
    - the reversed geometry object
    """
    def _reverse(x, y):
        return x[::-1], y[::-1]
    return sh.ops.transform(_reverse, geom)
    
def getAngle(pt1, pt2): 
    """
    Calculates the angle between two points.
    
    Inputs:
    - pt1 (tuple): a tuple containing the x and y coordinates of the first point
    - pt2 (tuple): a tuple containing the x and y coordinates of the second point
    
    Returns:
    - float: the angle between the two points (in degrees)
    """
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]    
    return math.degrees(math.atan2(y_diff, x_diff))

    # MAYBE MORE ACCURATE:

    # lon1, lat1 = map(math.radians, pt1)
    # lon2, lat2 = map(math.radians, pt2)
    # d_lon = lon2 - lon1
    # x = math.sin(d_lon) * math.cos(lat2)
    # y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))
    
    # initial_bearing = math.atan2(x, y)
    # initial_bearing = math.degrees(initial_bearing)
    
    # compass_bearing = (initial_bearing + 360) % 360
    # return compass_bearing

# UNUSED
def isLeft(a,b,c):
    """
    Determines if point c is to the left of the line formed by points a and b.
    
    Inputs:
    - a (tuple): a tuple containing the x and y coordinates of the first point
    - b (tuple): a tuple containing the x and y coordinates of the second point
    - c (tuple): a tuple containing the x and y coordinates of the third point
    
    Returns:
    - float: the determinant of the matrix formed by the three points
    """
    return ((b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0]))

# UNUSED
def inBtw(line,p3):
    # perpendicular line
    # approach 1
    # check if line btw perpendicular line of the ends of the line
    #     |                |        ---: perpendicular lines at either end of the OSM edge
    #     |__________OSM___|        x: LHM point next to OSM edge
    #     |    x           |  o     o: LHM point not next to OSM edge
    '''
    x1,y1 = np.array(line)[0]
    x2,y2 = np.array(line)[-1]
    k = (y2-y1)/(x2-x1)
    x3,y3 = p3[0],p3[1]
    inBtw = False
    if (y1+k*x1)>(y2+k*x2):
        if (y3 < -k*x3+(y1+k*x1)) and (y3 > -k*x3+(y2+k*x2)):
            inBtw = True 
    else:
        if (y3 > -k*x3+(y1+k*x1)) and (y3 < -k*x3+(y2+k*x2)):
            inBtw = True 
    return inBtw'''

    # approach 2
    # make perpendicular line at centre
    # check that dist of p3 to line < length of line / 2
    x1,y1 = np.array(line)[0]
    x2,y2 = np.array(line)[-1]
    #//find the center
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    #//move the line to center on the origin
    x1-=cx; y1-=cy
    x2-=cx; y2-=cy
    #//rotate both points
    xtemp = x1; ytemp = y1
    x1=-ytemp; y1=xtemp
    xtemp = x2; ytemp = y2
    x2=-ytemp; y2=xtemp
    #//move the center point back to where it was
    x1+=cx; y1+=cy
    x2+=cx; y2+=cy
    # check that dist of p3 to line < length/2
    p1 = np.asarray([x1,y1]) 
    p2 = np.asarray([x2,y2])
    p3 = np.asarray(p3)
    d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    if d < np.linalg.norm(p2-p1)/2:
        return True
    else: 
        return False


############################################
# NODE AND EDGE CLASSES
############################################

class Node():
    """
    A class to represent a node object.
    """
    def __init__(self,otherRows,mainRow):
        """
        Initializes a Node object with given data.

        Input:
        - mainRow: a pandas Series containing the main row of data to be used to initialize the Node object
        - otherRows: a pandas DataFrame containing other rows of data to be concatenated with the mainRow
        """
        if otherRows is not None:
            self.df = pd.concat([otherRows,pd.DataFrame([mainRow])],ignore_index=True)
        else:
            self.df = pd.DataFrame([mainRow])
            
        # key information
        self.id = int(mainRow.osmid)
        self.x = mainRow.geom_merged.x
        self.y = mainRow.geom_merged.y
        self.geometry = mainRow.geom_merged
        # self.crossing = [x for x in self.df.highway.explode().unique() if ((x!='') & (x==x) & (x is not None))]
        self.infra = list(set( listify(self.df.highway) + listify(self.df.crossing) ))
        self.crossing = True if 'crossing' in self.infra else False
        self.traffic_signals = True if 'traffic_signals' in self.infra else False
        
        self.l_id = listify(self.df.osmid,asInt=True)
        self.df['old_osmid'] = self.df['old_osmid'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        self.df["old_osmid"] = self.df["old_osmid"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
        self.l_osmid = listify(self.df.old_osmid,asInt=True) if (len(self.df.old_osmid.dropna())>0) else None
        self.l_highway_conn = listify(self.df.highway_conn)
        self.l_highway_rank = listify(self.df.highway_rank,asInt=True)

        # These have fewer than 5 entries...
        # 'street_count', 'bicycle', 'foot', 'barrier', 'lit', 'width', 'bicycle_parking', 'public_transport', 
        
        # Don't have to be merged
        # 'geometry_orig', 'geom_buffered', 'geometry', 'geom_buff_merged', 'merged_by', 'merged', 'newNode'
    
        self.df = None

    def to_dict(self):
        """
        Returns a dictionary representation of the Node object.
        """
        base = {
            # general single values
            'g_id': self.id,
            'g_x': self.x,
            'g_y': self.y,
            'g_geometry': self.geometry,
            'g_infra': self.infra,
            'g_crossing': self.crossing,
            'g_signals': self.traffic_signals,
            # multiple values
            'l_id': self.l_id,
            'l_osmid': self.l_osmid,
            #'l_highway': self.l_highway,
            'l_hw_conn': self.l_highway_conn,
            'l_hw_rank': self.l_highway_rank,
        }
        return base

class Link():
    """
    A class to represent a link object. A link object is a collection of edges that are parallel to each other.
    Links can have a two edges, one in each direction (UV and VU), or only one edge in one direction.
    """
    def __init__(self,otherRows,mainRow):
        """
        Initializes a Link object with a given set of rows (otherRows) and a primary row (mainRow).

        Inputs:
        - mainRow (pd.DataFrame): a single row represented as a pandas DataFrame
        - otherRows (pd.DataFrame): a pandas DataFrame containing additional rows to be concatenated with mainRow
        
        Returns:
        - None
        """
        if otherRows is not None:
            otherRows['osmid'] = otherRows.osmid_i
            # NOTE: previously had 'ignore index = True'
            self.df = pd.concat([otherRows,pd.DataFrame([mainRow])],ignore_index=False)
            angle_l1 = mainRow.bearing
        else:
            self.df = pd.DataFrame([mainRow])
            seg = np.array(mainRow.geom_linear.coords)
            angle_l1 = np.angle(complex(*(seg[1] - seg[0])), deg=True)
            self.df['bearing'] = angle_l1

        self.df = self.df.replace('',np.nan)
        
        # key information
        self.u = mainRow.new_u
        self.v = mainRow.new_v
        try:
            self.geometry = sh.geometry.LineString([mainRow.geometry.coords[0],mainRow.geometry.coords[-1]]) # generate a simple line
        except:
            print('Error in generating geometry for edge:',mainRow.osmid)
            print(mainRow.geometry)
            for a in mainRow.geometry.geoms:
                plt.plot(*a.coords.xy)
            plt.show()

        # determine direction of the parallel links
        self.df['direction'] = False
        if not ((self.df.bearing.max()-self.df.bearing.min()) < 90):
            # edges in both directions
            a = (angle_l1-90) if (angle_l1-90)>-180 else angle_l1+270 # °
            b = (angle_l1+90) if (angle_l1+90)<180 else angle_l1-270  # °
            lower,upper = min([a,b]), max([a,b])
            idcs = self.df[(self.df.bearing<upper) & (self.df.bearing>lower)].index.values
            self.df.loc[idcs,'direction'] = True

        direction_mainRow = self.df.loc[mainRow.name,'direction']
        if len(self.df[self.df.direction==direction_mainRow])>0:
            self.edgeUV = AnEdge(self.df,mainRow,True,reversed=direction_mainRow) # 1
        else:
            self.edgeUV = None
        if len(self.df[self.df.direction==(not direction_mainRow)])>0:
            self.edgeVU = AnEdge(self.df,mainRow,False,reversed=(not direction_mainRow)) # -1
        else:
            self.edgeVU = None
            
        # TODO: match information across UV and VU 
        # e.g. if have cycleway:both, then add cycleway information to other edge
        # --> look for any 'left_XXX' in b_attribut
        # self.enrichInformation

class AnEdge():
    """
    A class to represent an edge object. An edge object is a collection of edges that are parallel to each other, AND in the same direction.
    """
    def __init__(self,df,mainRow,uv=True,reversed=False):
        """
        This function takes in all currently considered edges (i.e. parallel to the edge currently 
        considered in the 'mergeEdges' function).

        Inputs:
        df (pd.DataFrame): dataframe containing all considered edges (all modes, all directions)
        mainRow (pd.DataFrame): used for general infos (geometry, u, v, id)
        uv (bool): indicates whether edge uv or vu is currently considered
        reversed (bool): indicated where currently the edges with similar or opposite 
        heading compared to the mainRow are considered
        """
        self.df = df
        self.subset = self.df[self.df.direction==reversed]

        # single values
        self.u = int(mainRow.new_u) #if not reversed else mainRow.new_v
        self.v = int(mainRow.new_v) #if not reversed else mainRow.new_u
        self.id = int(mainRow.osmid)
        self.lit = self.isLit()
        self.reversed = not uv # reversed # (as u and v are swapped)
        # self.reversed = False 
        self.crossing = self.getParam(self.df,'crossing') if 'crossing' in self.df.columns else None # ['' 'unmarked' 'traffic_signals' 'marked']
        self.geometry = mainRow.geometry # path != road geometry, but do we need detailed geometry here? No, but we do need the rough geometry for the map matching.
        self.geom_linear = mainRow.geom_linear 
        self.geom_reassigned = mainRow.geom_reassigned 
        self.incline = [str(a) for a in self.subset.incline if str(a) in ['up','down']] # {'0%', '15%', "['up', 'down']", 'down', 'nan', 'up'}
        self.parking_left,self.parking_right = self.hasParking()
        
        self.gradient = np.mean([float(a) for a in self.subset.gradient if (a is not None) and (a==a)]) 
        self.height_difference = np.mean([float(a) for a in self.subset.height_difference if (a is not None) and (a==a)]) 
        self.severity = np.mean([float(a) for a in self.subset.severity if (a is not None) and (a==a)]) 
        self.green_ratio = np.mean([float(a) for a in self.df.green_ratio if (a is not None) and (a==a)])
        self.retail_ratio = np.mean([float(a) for a in self.df.retail_ratio if (a is not None) and (a==a)])
        self.building_ratio = np.mean([float(a) for a in self.df.building_ratio if (a is not None) and (a==a)])
        
        # multiple values
        self.l_id = listify(self.subset.osmid,asInt=True)
        self.df['old_osmid'] = self.df['old_osmid'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        self.df["old_osmid"] = self.df["old_osmid"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
        self.l_osmid = listify(self.subset.old_osmid,asInt=True)
        self.l_old_u = listify(self.subset.old_u,asInt=True)
        self.l_old_v = listify(self.subset.old_v,asInt=True)
        self.l_highway = listify(self.subset.highway)
        #self.l_highway = self.getHighways(self.subset)
        self.l_highway_rank = listify(self.subset.highway_rank)

        # general mode information
        self.access_bike = self.checkModeAccess('bike') # self.isTrue(self.subset.bike_access)
        self.access_walk = self.checkModeAccess('walk') 
        self.access_motorized = self.checkModeAccess('motorized') 
        # self.access_PT = self.checkModeAccess('PT') 

        self.walkEdge = WalkEdge(mainRow,self.df) if self.access_walk else None
        self.bicycleEdge = BicycleEdge(mainRow,self.df) if self.access_bike else None
        self.motorizedEdge = MotorizedEdge(mainRow,self.df) if self.access_motorized else None
        # seld.ptEdge = PTEdge(mainRow,self.df) if self.access_PT else None

        self.df = None # remove df after setting all params, to reduce data size        

    def hasParking(self):
        """
        Check if the edge has parking information.

        Returns:
        - bool: True if the edge has parking information, False otherwise
        """
        parking_left = []
        parking_right = []
        if 'parking:lane:left' in self.subset.columns:
            parking_left += [a for a in self.subset['parking:lane:left'].explode().unique() if a not in ['no','no_stopping','fire_lane',np.nan,''] and (a is not None) and a==a]
        if 'parking:lane:right' in self.subset.columns:
            parking_right += [a for a in self.subset['parking:lane:right'].explode().unique() if a not in ['no','no_stopping','fire_lane',np.nan,''] and (a is not None) and a==a]
        if 'parking:lane:both' in self.subset.columns:      
            parking_left += [a for a in self.subset['parking:lane:both'].explode().unique() if a not in ['no','no_stopping','fire_lane',np.nan,''] and (a is not None) and a==a]
            parking_right += [a for a in self.subset['parking:lane:both'].explode().unique() if a not in ['no','no_stopping','fire_lane',np.nan,''] and (a is not None) and a==a]
        parking_left = list(set(parking_left)) 
        parking_right = list(set(parking_right))
        return parking_left,parking_right
        
    def checkModeAccess(self,m):
        """
        Checks if a given mode of transportation is allowed based on certain conditions.

        Inputs:
        - self: reference to the object calling this method
        - m (str): mode of transportation ('walk', 'bike' or 'motorized')

        Returns:
        - access (bool): True if the given mode of transportation is allowed, False otherwise
        """
        access = False
        
        if m=='walk':
            walkHighways = ['path','footway','pedestrian','steps','bridleway','track']
            walkHighways += ['residential','tertiary','agricultural','service']
            walkHighways += ['tertiary_link']
            cond0 = any(x in walkHighways for x in self.subset.highway.explode().unique())
            cond1 = any(x in ['designated','yes'] for x in self.subset.foot.explode().unique())
            cond2 = any(x<=50 for x in self.subset.maxspeed.explode().unique() if x is not None)
            cond3 = any(x not in ['no',''] for x in self.subset.sidewalk.explode().unique() if ((x is not None) & (x==x)))
            #cond3 = nonWalkHighways = ['trunk',]
            #cond4 = if unclassified
            #cond5 = any(x>0 for x in self.subset.busstops.unique())
            if cond0 | cond1 | cond2 | cond3:
                access = True
            return access
        
        if m=='bike':
            cols = ['bicycle_road','oneway:bicycle','cycleway:both','cycleway:right',\
                    'cycleway:left','cycleway:right:lane','ramp:bicycle']
            bikeHighways = ['cycleway']
            bikeHighways += ['residential','tertiary','secondary','agricultural','service']
            bikeHighways += ['tertiary_link','secondary_link']
            cond0 = any(x in bikeHighways for x in self.subset.highway.explode().unique())
            cond1a = any(x in ['yes','designated'] for x in self.subset.bicycle.explode().unique())
            cond1b = all(x=='no' for x in self.subset.bicycle.explode().unique())
            cond2 = False
            for c in cols:
                if any(str(x) not in ['no',''] for x in list(set(self.subset.loc[:,c].explode().values)) if ((x is not None) & (x==x))):
                    cond2 = True
                    break
            cond3 = any(str(x)!='' for x in self.subset.cycleway.explode().unique() if ((x is not None) & (x==x))) 
            # cond4 = nonWalkHighways = ['trunk',]
            # cond5 = if unclassified
            if cond0 | cond1a | cond2:
                access = True
            if cond1b:
                access = False
            if cond3:
                access = True
            # TODO: incl. oneway roads for cyclists without cycleway='opposite'
            #if any(self.subset['oneway:bicycle'].isin(['True',True])) & any(self.subset.reversed.isin(['True',True])):
            #    access = False
            return access
        
        if m=='motorized':
            nonMotorizedHighways = ['path','footway','pedestrian','steps','bridleway','cycleway','track']
            access = any(x not in nonMotorizedHighways for x in self.subset.highway.explode().unique())
            if any(self.subset.oneway.isin(['True',True])):
                if any(self.subset.bike_access=='bike_only'):
                    access = False
                # if self.reversed:
                #     access = False
            return access

    def getParam(self,df,col):
        """
        Extracts unique non-null values from the specified column in a pandas dataframe.

        Inputs:
        - df (pd.DataFrame): The pandas dataframe to extract values from.
        - col (str): The name of the column to extract values from.

        Returns:
        - Union[str, List[str], None]: 
          If there is only one unique non-null value in the column, returns the value as a string. 
          If there are multiple unique, non-null values in the column, returns a list of the values.
          If there are no unique non-null values in the column, returns None.
        """
        a = df[col].dropna().explode().unique().tolist()
        if '' in a:
            a.remove('')
        if len(a)==0:
            return None
        # if len(a)==1:
        #     return a[0]
        return a

    def isTrue(self,values):
        """
        Check if a list or pandas Series contains the strings 'yes' or the boolean value True.

        Inputs:
        - values (list or pandas Series): A list or pandas Series of values to check.

        Returns:
        - bool: True if 'yes' or True is present in the input values, False otherwise.
        """
        values = values.unique().tolist()
        if ('yes' in values) or (True in values):
            return True
        else:
            return False
    
    def isLit(self):
        """
        This function checks if a location is lit or not based on the values in the 'lit' column of a dataframe.

        Inputs:
        - self - the class instance containing the dataframe with the 'lit' column

        Returns:
        - (bool) True if the location is lit, False otherwise
        """
        # ["['yes', '24/7']" 'yes' '' 'no' "['yes', 'no']"]
        litValues = listify(self.df.lit)
        if litValues is None:
            return False
        elif ('yes' in litValues) or ('24/7' in litValues):
            return True
        else:
            return False

    def showAccess(self):
        """
        Displays whether this edge is accessible for walking, biking, motorized vehicles, and public transportation.

        Inputs: 
            self - object instance

        Returns: 
            None
        """
        print('Walk access:\t\t',self.access_walk)
        print('Bicycle access:\t\t',self.access_bike)
        print('Motorized access:\t',self.access_motorized)
        # print('PT access:\t\t',self.access_PT)
    
    def to_dict(self,listOfModes=['walk','bike','motorized']):
        """
        Converts the attributes of the edge object and it's modespecific sub-edges into a dictionary.
        Inputs:
        - listOfModes (optional): a list of modes for which to include additional key-value pairs in 
        the returned dictionary. Default value is ['walk','bike','motorized'].
        
        Returns:
        - a dictionary representation of the edge object based on the values in listOfModes.
        """
        # max 10 characters for key names (.gpkg requirements)
        base = {
            # general single values
            #'x': self.link.x,
            #'y': self.link.y,
            'g_u': self.u,
            'g_v': self.v,
            'g_id': self.id,
            'g_lit': self.lit,
            'g_incline': self.incline,
            'g_gradient': self.gradient,
            'g_height_d': self.height_difference,
            'g_severity': self.severity,
            'g_reversed': self.reversed,
            'g_crossing': self.crossing,
            'g_greenR': self.green_ratio,
            'g_retailR': self.retail_ratio,
            'g_buildR': self.building_ratio,
            'g_geometry': self.geometry,
            'g_geo_lin': self.geom_linear,
            'g_geo_rea': self.geom_reassigned,
            'g_parkingL': self.parking_left,
            'g_parkingR': self.parking_right,

            # multiple values
            'l_id': self.l_id,
            'l_osmid': self.l_osmid,
            'l_old_u': self.l_old_u,
            'l_old_v': self.l_old_v,
            'l_highway': self.l_highway,
            'l_hw_rank': self.l_highway_rank,
        
            # general mode information
            'access_bik': self.access_bike,
            'access_wal': self.access_walk, 
            'access_mot': self.access_motorized,
            # 'access_PT': self.access_PT,
        }
        if ('walk' in listOfModes) & (self.walkEdge is not None):
            base.update({
                'w_length': self.walkEdge.length,
                'w_surface': self.walkEdge.surface,
                'w_smoothne': self.walkEdge.smoothness,
                'w_width': self.walkEdge.width,
                'w_segregat': self.walkEdge.segregated,
            })
        if ('bike' in listOfModes) & (self.bicycleEdge is not None):
            base.update({
                'b_length': self.bicycleEdge.length,
                'b_surface': self.bicycleEdge.surface,
                'b_smoothne': self.bicycleEdge.smoothness,
                'b_width': self.bicycleEdge.width,
                'b_bikeRoad': self.bicycleEdge.bicycle_road,
                'b_oneway': self.bicycleEdge.oneway,
                'b_category': self.bicycleEdge.cycleway_category,
                'b_attribut': self.bicycleEdge.cycleway_attributes,
                'b_segregat': self.bicycleEdge.segregated,
                'b_amntyOn': self.bicycleEdge.amenity_on,
                'b_amntyNea': self.bicycleEdge.amenity_nearby,
                'b_bikerack': self.bicycleEdge.bike_parking_on,
            })
        if ('motorized' in listOfModes) & (self.motorizedEdge is not None):
            base.update({
                'm_length': self.motorizedEdge.length,
                'm_width': self.motorizedEdge.width,
                'm_lanes': self.motorizedEdge.lanes,
                'm_oneway': self.motorizedEdge.oneway,
                'm_maxspeed': self.motorizedEdge.maxspeed,
                # 'm_public_transport': self.motorizedEdge.public_transport,
                'm_ptStop': self.motorizedEdge.pt_stop,
                'm_ptRoutes': self.motorizedEdge.pt_routes,
            })

        return base

class WalkEdge():
    """
    A class to represent a walk edge object, i.e., an edge that is accessible for walking.
    """
    def __init__(self,mainRow,df):
        """
        Initializes a WalkEdge object with the specified mainRow and df inputs.

        Inputs:
        - mainRow: A row of data representing the main feature of the WalkEdge.
        - df: A pandas dataframe containing data about the WalkEdge.

        Returns:
        - None
        """
        # when creating a child edge, the init of the parent edge is ran, so no need to reset it all
        # but then i have 4 times the same information...
        walkEdges = self.getWalkEdges(df)
        self.length = walkEdges.length.max() # .sum()
        self.surface = listify(walkEdges.surface) 
        self.smoothness = listify(walkEdges.smoothness)  
        self.segregated = listify(walkEdges.segregated)  
        self.width = self.getLanes(walkEdges.width,walkEdges.length) 
        self.df = None # remove df after setting all params, to reduce data size        
        
    def getWeightedWidth(self,df):
        """
        Calculates the weighted width of the WalkEdge based on the specified dataframe.

        Input:
        - df (pd.DataFrame): A pandas dataframe containing data about the WalkEdge.

        Returns:
        - (float) The weighted width of the WalkEdge.
        """
        tmp = df[['width','length']][df.width!=''].astype(float).dropna()
        if len(tmp)<1:
            return None
        elif len(tmp)==1:
            return tmp.width.values[0]
        else:
            return np.average(tmp['width'],weights=tmp['length']) if (np.sum(tmp['length'])>0) else np.average(tmp['width'])
        
    def getWalkEdges(self,df):
        """
        Filters the dataframe to find the WalkEdges based on certain conditions.
        
        Input:
        - df (pd.DataFrame): A pandas dataframe containing data about the WalkEdge.

        Returns:
        - A filtered dataframe containing the WalkEdges.
        """
        walkHighways = ['path','footway','pedestrian','steps','bridleway','track']
        cond0 = any(x in walkHighways for x in df.highway.explode().unique())
        if cond0:
            return df[df.highway.isin(walkHighways)]
        cond1 = any(x=='designated' for x in df.foot.explode().unique())
        if cond1:
            return df[(df.foot.isin(['designated','yes']))]
        walkHighways += ['residential','tertiary','agricultural','service','tertiary_link']
        cond2 = any(x in walkHighways for x in df.highway.explode().unique())
        if cond2:
            return df[df.highway.isin(walkHighways)]
        
        return df[(df.maxspeed<=30) | ~df.sidewalk.isin(['no',''])] # consider nan...
            
    def getLanes(self,laneVals,lengthVals):
        """
        Calculates the weighted average of a list of values.
        Similar to getWeightedWidth(self,df)

        Input:
        - lanevals (list or array): a list or array containing a set of values.

        Returns:
        - (float) The weighted width of the input variable.
        """
        # takes a pandas series
        lanes = []
        lengths = []
        for i,a in enumerate(laneVals): 
            if (a=='') | (pd.isna(a)):
                continue
            elif (type(a)==float) | (type(a)==int) | (type(a)==np.float64) | (type(a)==np.int64):
                lanes.append(a)
            elif (a[0]=="["):
                tmp = a[2:-2].split("', '")
                lanes.append(np.mean([float(b) for b in tmp]))
            else:
                lanes.append(float(a))
            lengths.append(lengthVals.values[i])
        if len(lanes)==0:
            return None
        elif len(lanes)==1:
            return lanes[0]
        else:
            return np.average(lanes,weights=lengths) if (np.sum(lengths)>0) else np.average(lanes)

class BicycleEdge():
    """
    A class to represent a bicycle edge object, i.e., an edge that is accessible for biking.
    """
    def __init__(self,mainRow,df):
        """
        Initializes a BikeEdge object with the specified mainRow and df inputs.

        Inputs:
        - mainRow: A row of data representing the main feature of the BikeEdge.
        - df: A pandas dataframe containing data about the BikeEdge.

        Returns:
        - None
        """

        # when creating a child edge, the init of the parent edge is ran, so no need to reset it all
        # but then i have 4 times the same information...
        bikeEdges = self.getBikeEdges(df)
        self.length = bikeEdges.length.max() # .sum()
        if len(bikeEdges['cycleway:surface'].explode().unique())>1:
            self.surface = listify(bikeEdges['cycleway:surface']) # if known, for wheelchair-users for instance    
        else:
            self.surface = listify(bikeEdges.surface) # if known, for wheelchair-users for instance
        self.smoothness = listify(bikeEdges.smoothness)  # if known, for wheelchair-users for instance
        if 'width_cycle_path' in bikeEdges.columns:
            self.width = self.getWeightedWidth(bikeEdges) # of what? prob of path...
        else:
            self.width = listify(bikeEdges['cycleway:width'])
        self.bicycle_road = any(bikeEdges.bicycle_road=='yes')
        self.oneway = any(bikeEdges['oneway:bicycle']=='yes') 
        self.amenity_on = listify(bikeEdges.amenity_on)  # if known, for wheelchair-users for instance
        self.amenity_nearby = listify(bikeEdges.amenity_nearby)  # if known, for wheelchair-users for instance
        self.bike_parking_on = True if any('bicycle_parking' in a for a in self.amenity_on) else False  # if known, for wheelchair-users for instance
        
        self.cycleway_category = listify(bikeEdges[bikeEdges.cycleway_category!=bikeEdges.highway].cycleway_category) 
        
        # TODO: differentiate more between category and attributes?
        # start with the most specific and then go to the more general
        def getBikeAttr(osmTag, prefix):
            attr = [x for x in bikeEdges[osmTag] if ((x not in ['','pictogram']) & (x is not None) & (x==x))]
            if ('no' in attr) and (len(attr)>1):
                attr.remove('no')
            # if attr like this: ['track', "['no', 'lane']"] --> remove 'no' from the sublist --> ['track', 'lane']
            rank = ['no', 'track', 'lane', 'advisory', 'exclusive.', 'exclusive']
            # pick the one with the highest rank
            for i,a in enumerate(attr):
                if type(a)==list:
                    attr[i] = sorted(a,key=lambda x: rank.index(x) if x in rank else 0)[-1]
            return ['%s%s'%(prefix,x) for x in attr]
        # 1. first check cycleway:left:lane
        attr1 = getBikeAttr('cycleway:left:lane','left_lane_')
        attr2 = getBikeAttr('cycleway:right:lane','right_lane_')
        attr3, attr4, attr5, attr6 = [],[],[],[]
        # 2. if no cycleway:left:lane, check cycleway:left
        if len(attr1)==0:
            attr3 = getBikeAttr('cycleway:left','left_')
        if len(attr2)==0:
            attr4 = getBikeAttr('cycleway:right','right_')        
        # 3. if no cycleway:left, check cycleway
        if (len(attr1)==0) and (len(attr3)==0) or (len(attr2)==0) and (len(attr4)==0):
            attr5 = getBikeAttr('cycleway','')
            attr5 += getBikeAttr('cycleway:both','')
            if ('no' in attr5) and (len(attr5)>1):
                attr5.remove('no')
        gatherCyclewayAttributes = [item for sublist in [attr1,attr2,attr3,attr4,attr5] for item in sublist]
        gca = []
        for a in gatherCyclewayAttributes:
            if type(a)==list:
                gca.extend([b for b in a])
            else:
                gca.append(a)
        self.cycleway_attributes = sorted(list(set(gca)))
        self.segregated = listify(bikeEdges.segregated)
        
        self.df = None # remove df after setting all params, to reduce cache size        

    def getWeightedWidth(self,df):
        """
        Calculates the weighted width of the BikeEdge based on the specified dataframe.

        Input:
        - df (pd.DataFrame): A pandas dataframe containing data about the BikeEdge.

        Returns:
        - (float) The weighted width of the BikeEdge.
        """
        tmp = df[['width_cycle_path','length']].dropna()
        if len(tmp)<1:
            return None
        elif len(tmp)==1:
            return tmp.width_cycle_path.values[0]
        else:
            return np.average(tmp['width_cycle_path'],weights=tmp['length']) if (np.sum(tmp['length'])>0) else np.average(tmp['width_cycle_path'])
        
    def getBikeEdges(self,df):
        """
        Filters the dataframe to find the BikeEdges based on certain conditions.
        
        Input:
        - df (pd.DataFrame): A pandas dataframe containing data about the BikeEdge.

        Returns:
        - A filtered dataframe containing the BikeEdges.
        """
        bikeHighways = ['cycleway']
        cond0 = any(x in bikeHighways for x in df.highway.explode().unique())
        cond1a = any(x in ['yes','designated'] for x in df.bicycle.explode().unique())
        if cond0 | cond1a:
            return df[(df.highway.isin(bikeHighways)) | (df.bicycle=='yes') | (df.bicycle=='designated') ]
        bikeHighways += ['residential','tertiary','secondary','agricultural','service']
        bikeHighways += ['tertiary_link','secondary_link']
        cond0 = any(x in bikeHighways for x in df.highway.explode().unique())
        cols = ['bicycle_road','oneway:bicycle','cycleway:both','cycleway:right',\
                    'cycleway:left','cycleway:right:lane','ramp:bicycle',]
        cond2 = False
        for c in cols:
            if any(str(x) not in ['no',''] for x in list(set(df.loc[:,c].explode().values)) if ((x is not None) & (x==x))):
                cond2 = True
                break
        cond3 = any(str(x)!='' for x in df.cycleway.explode().unique() if ((x is not None) & (x==x))) 
        if cond0 | cond2 | cond3:
            return df[(df.highway.isin(bikeHighways)) | (df.cycleway!='')]
        
class MotorizedEdge():
    """
    A class to represent a motorized edge object, i.e., an edge that is accessible for motorized vehicles.
    """
    def __init__(self,mainRow,df):
        motorizedEdges = self.getMotorizedEdges(df)
        self.length = motorizedEdges.length.max() # .sum()
        self.width = self.getLanes(motorizedEdges.width,motorizedEdges.length) # of what? prob of path...
        self.lanes = self.getLanes(motorizedEdges.lanes,motorizedEdges.length)
        self.oneway = True in motorizedEdges['oneway'].unique() 
        self.maxspeed = self.getLanes(motorizedEdges.maxspeed,motorizedEdges.length)
        self.pt_stop = motorizedEdges.pt_stop_on.max()
        self.pt_routes = listify(motorizedEdges.pt_stop_routes)

        self.df = None # remove df after setting all params, to reduce data size        

    def getWeighted(self,df,col):
        """
        Calculates the weighted average of the values.

        Input:
        - df (pd.DataFrame): A pandas dataframe containing data about the BikeEdge.

        Returns:
        - (float) The weighted average of the BikeEdge attribute.
        """
        tmp = df[[col,'length']].dropna()
        if len(tmp)<1:
            return None
        elif len(tmp)==1:
            return df.width.values[0]
        else:
            return np.average(df[col].astype(float),weights=df['length']) if (np.sum(df['length'])>0) else np.average(df[col].astype(float))
        
    def getMotorizedEdges(self,df):
        """
        Filters the dataframe to find the MotorizedEdges based on certain conditions.
        
        Input:
        - df (pd.DataFrame): A pandas dataframe containing data about the MotorizedEdge.

        Returns:
        - A filtered dataframe containing the MotorizedEdges.
        """
        nonMotorizedHighways = ['path','footway','pedestrian','steps','bridleway','cycleway','track']
        if any(x not in nonMotorizedHighways for x in df.highway.explode().unique()):
            return df[~df.highway.isin(nonMotorizedHighways)]
        return df

    def getLanes(self,laneVals,lengthVals):
        """
        Calculates the weighted average of a list of values.
        Similar to getWeightedWidth(self,df)

        Input:
        - lanevals (list or array): a list or array containing a set of values.

        Returns:
        - (float) The weighted width of the input variable.
        """
        lanes = []
        lengths = []
        for i,a in enumerate(laneVals): 
            if (a=='') | (pd.isna(a)):
                continue
            elif (type(a)==float) | (type(a)==int) | (type(a)==np.float64) | (type(a)==np.int64):
                lanes.append(a)
            elif (a[0]=="["):
                tmp = a[2:-2].split("', '")
                lanes.append(np.mean([float(b) for b in tmp]))
            else:
                lanes.append(float(a))
            lengths.append(lengthVals.values[i])
        if len(lanes)==0:
            return None
        elif len(lanes)==1:
            return lanes[0]
        else:
            return np.average(lanes,weights=lengths) if (np.sum(lengths)>0) else np.average(lanes)
       

