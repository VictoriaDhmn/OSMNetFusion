# %%
"""
Script for simplifying the osm network

Inputs
- configFile.py: configuration file with all values
    - visualize: boolean for visualizing the steps
    - plot_bounds: bounds for the plot
    - crs: crs for the geodataframes
    - parallelized: boolean for parallelizing the code
    - maxAngleInitial: maximum angle for splitting the curves
    - maxAnglePrev: maximum angle for splitting the curves
    - HIGHWAY_RANKING: dictionary with the highway ranking
    - HIGHWAY_BUFFERS_1: buffer for the nodes
    - HIGHWAY_BUFFERS_2: buffer for the nodes
    - clusterThreshold: threshold for clustering the nodes
    - geom_col: geometry column
    - version: version of the network
    - p2_result_filepath: filepath for the p2 result
    - p3_result_filepath: filepath for the p3 result
- p3_functions.py: functions for simplifying the network
Outputs
- geopackage file with the simplified osm network (saved to p3_result_filepath)
"""

#######################################
# Import required python packages
#######################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import time
import sys
import os
sys.path.append(os.getcwd())
from p3_functions import *

import warnings
warnings.filterwarnings("ignore", message=".*'GeoDataFrame.swapaxes' is deprecated and will be removed in a future version.*", category=FutureWarning)

# because of a geopandas warning
import os
os.environ['USE_PYGEOS'] = '0'
pd.set_option('future.no_silent_downcasting', True)

# load all values in configFile.py
import configFile

############################################
# CONTENT
############################################

# Steps   Nodes     Edges   Name
#         (added/dropped)
# 1.      +         +       split curves - parallel
# 2.                        assign ranks to highway types
# 3.                        determine how 'important' a node is 
# 4.                        buffer nodes
# 5.                        find buffer intersections & cluster nodes (1st iteration) 
# 6.      +         +       split edges if they pass through a buffer - SLOW
# 7.                        cluster nodes again
# 8.                -       reassign edges
# 9.      -         -       (DISABLED) remove degree 2 nodes
# 10.     -                 merge nodes
# 11.               -       merge edges
# 12.                       convert merged objects to dict to dataframes
# 13.                       convert & save

# NOTE: g_id is unique for nodes and edges

# RUNTIME (outdated, but gives an idea)
# VERSION   SIZE in km2        num. nodes  num. edges   runtime
# small     2*2=4km2       --> n:   6,300  e:   2,500   45+20=47s
# medium    4*4=16km2      --> n:  29,000  e:  11,100   45*5+2*70=370s


############################################
# NETWORK SIMPLIFICATION
############################################

CFvisualize = configFile.visualize
CFplot_bounds = configFile.plot_bounds
CFcrs = configFile.crs
CFparallelized = configFile.parallelized
CFp2_result_filepath = configFile.p2_result_filepath
CFmaxAngleInitial = configFile.maxAngleInitial
CFmaxAnglePrev = configFile.maxAnglePrev
CFhighwayRanking = configFile.HIGHWAY_RANKING
CFhighwayBuffers1 = configFile.HIGHWAY_BUFFERS_1
CFhighwayBuffers2 = configFile.HIGHWAY_BUFFERS_2
CFclusterThreshold = configFile.clusterThreshold
CFgeom_col = configFile.geom_col
CFversion = configFile.version
CFp3_result_filepath = configFile.p3_result_filepath

def main():
    ############################################
    # 0. Preparation - INPUTS & PARAMETERS
    startX = time.time()
    ############################################
    
    # load parameters
    crs = CFcrs
    parallelized = CFparallelized
    plot_bounds = CFplot_bounds
    visualize = CFvisualize

    # load network data and set crs
    gdf_edges = gpd.read_file(CFp2_result_filepath, layer = 'edges',crs=crs)
    gdf_nodes = gpd.read_file(CFp2_result_filepath, layer = 'nodes',crs=crs)
    make_plot(gdf_edges.geometry,gdf_nodes.geometry,bounds=None)

    ####################################
    # 1. split curves
    # ----------------------------------
    # split into subsegments where node has degree > 2 AND where not straight
    start = time.time()
    ####################################

    lnprev,leprev = int(len(gdf_nodes)),int(len(gdf_edges))
    # gdf_edges['highway_conn'] = np.nan # NOTE:EDIT

    if parallelized:
        es,new_ns = run_splitCurves_in_parallel(gdf_edges, gdf_nodes,maxAngleInitial=CFmaxAngleInitial,maxAnglePrev=CFmaxAnglePrev)
        edges1 = es
        gdf_edges = pd.concat(edges1)
        new_ns = [n for n in new_ns if (not n.empty) and (not n.isnull().all().all())]
        if len(new_ns)>0:
            nodes1 = [gdf_nodes]
            nodes1.extend(new_ns)
            gdf_nodes = pd.concat(nodes1)
    else:
        gdf_edges,new_ns = splitCurves(gdf_edges,gdf_nodes,maxAngleInitial=CFmaxAngleInitial,maxAnglePrev=CFmaxAnglePrev)
        gdf_nodes = pd.concat([gdf_nodes,new_ns])

    print('\te:',leprev,'-->',len(gdf_edges))
    print('\tn:',lnprev,'-->',len(gdf_nodes))
    print("Completed step 1 in %s"%(round(time.time()-start,2)))
    if visualize:
        make_plot(gdf_edges.geometry,gdf_nodes.geometry,bounds=plot_bounds)

    ####################################
    # 2. assign ranks to highway types
    start = time.time()
    ####################################

    gdf_edges['highway_rank'] = addHighwayRank(gdf_edges,highway_ranking_custom=CFhighwayRanking).fillna(0)
    
    print("Completed step 2 in %s"%(round(time.time()-start,2)))
    
    ####################################
    # 3. determine how 'important' a node is 
    # ----------------------------------
    # determine road with the highest rank for each node 
    start = time.time()
    ####################################

    # make sure data is sorted by rank (length not necessary)
    gdf_edges = gdf_edges.sort_values(by=['highway_rank','length'],ascending=False)
    gdf_nodes['highway_conn'],gdf_nodes['highway_rank'] = getHighestRankingRoadOfNode(gdf_nodes,gdf_edges)
    gdf_nodes = gdf_nodes.sort_values(by=['highway_rank'],ascending=False)
    
    # plot
    if visualize:
        make_plot(gdf_edges[['geometry','highway_rank']],gdf_nodes[['geometry','highway_rank']],bounds=plot_bounds,title='node_importance',edge_col='geometry',node_col='geometry')
    
    print("Completed step 3 in %s"%(round(time.time()-start,2)))
    
    ####################################
    # 4. buffer nodes
    # ----------------------------------
    # buffer nodes by Xm based on most highest connected edge i.e. highway_rank
    start = time.time()
    ####################################

    gdf_nodes['geometry_orig'] = gdf_nodes.geometry
    gdf_nodes['geom_buffered'] = getGeomBuffered(gdf_nodes,highway_buffers_custom=CFhighwayBuffers1)
    
    print("Completed step 4 in %s"%(round(time.time()-start,2)))
    if visualize: 
        make_plot(gdf_edges.geometry,gdf_nodes.geom_buffered,bounds=plot_bounds)

    ####################################
    # 5. cluster nodes (1st iteration) 
    # ----------------------------------
    # find all intersections of the previously buffered nodes
    # then cluster nodes if they overlap
    # new columns: 'geom_merged','geom_buff_merged','merged_by','merged'
    start = time.time()
    ####################################

    # should still be sorted by highway rank
    lnprev = int(len(gdf_nodes))
    gdf_nodes = clusterNodes(gdf_nodes,clusterThreshold=CFclusterThreshold)
    
    print('\tn:',lnprev,'-->',len(gdf_nodes[gdf_nodes.merged=='k']))
    print("Completed step 5 in %s"%(round(time.time()-start,2)))
    if visualize: 
        make_plot(gdf_edges.geometry,gdf_nodes.geom_merged,bounds=plot_bounds) 

    ####################################
    # 6. split edges if they pass through a buffer - SLOW
    # ----------------------------------
    # split edges if they pass though a buffer
    # NOTE: linear rings are currently skipped/ignored, as edges are split in step 1
    start = time.time()
    ####################################

    lnprev,leprev = int(len(gdf_nodes)),int(len(gdf_edges))
    if parallelized:
        updated_es,new_es,new_ns,duplicates = run_splitEdgeIfInNodeBuffer_in_parallel(gdf_edges, gdf_nodes)
        nodes1 = [new_ns, gdf_nodes]
        edges1 = [new_es, updated_es, gdf_edges[~gdf_edges.index.isin(updated_es.index.values.tolist()+duplicates)]]
    else: 
        updated_es,new_es,new_ns = splitEdgeIfInNodeBuffer(gdf_edges,gdf_nodes)
        nodes1 = [new_ns, gdf_nodes]
        edges1 = [new_es, updated_es, gdf_edges[~gdf_edges.index.isin(updated_es.index.values.tolist())]]
    gdf_edges = gpd.GeoDataFrame(pd.concat(edges1),crs=crs)
    gdf_nodes = gpd.GeoDataFrame(pd.concat(nodes1),crs=crs)
    
    # update geom_buffered
    gdf_nodes['geom_buffered'] = getGeomBuffered(gdf_nodes,highway_buffers_custom=CFhighwayBuffers2)
    # sort values again (including new nodes)
    gdf_edges = gdf_edges.sort_values(by=['highway_rank','length'], ascending=False)
    gdf_nodes = gdf_nodes.sort_values(by=['highway_rank'], ascending=False)
    
    # NOTE: now node osmids are no longer unique, as some edges were split!
    print('\te:',leprev,'-->',len(gdf_edges))
    print('\tn:',lnprev,'-->',len(gdf_nodes))
    lnprev = int(len(gdf_nodes))
    print("Completed step 6 in %s"%(round(time.time()-start,2)))
    if visualize: 
        make_plot(gdf_edges.geometry,gdf_nodes.geom_merged,bounds=plot_bounds, edge_col='geometry', node_col='geom_merged') 

    ####################################
    # 7. cluster nodes again
    # ----------------------------------
    # cluster nodes again, considering new nodes from previously split edges (and new nodes)
    # new/updated columns = 'geom_merged','geom_buff_merged','merged_by','merged'
    start = time.time()
    ####################################

    # should still be sorted by highway rank
    gdf_nodes = clusterNodes(gdf_nodes,again=True, clusterThreshold=CFclusterThreshold)
    
    print('\tn:',lnprev,'-->',len(gdf_nodes[gdf_nodes.merged=='k']))
    print("Completed step 7 in %s"%(round(time.time()-start,2)))
    if visualize: 
        make_plot(gdf_edges.geometry,gdf_nodes.geom_merged,bounds=plot_bounds, edge_col='geometry', node_col='geom_merged') 

    ####################################
    # 8. reassign edges
    # ----------------------------------
    # reassign edges to nodes
    # if node X merged by/to node Y, change X-->Y for all 'u' and 'v' values 
    # update geometries
    #     - add geom_linear --> line from u to v
    #     - add geom_reassigned --> change X-->Y i.e. first/last point updated
    start = time.time()
    ####################################

    mergeIdx = gdf_nodes[['osmid','merged_by','geom_merged']].drop_duplicates()
    mergeIdx = mergeIdx.set_index('osmid')
    gdf_edges[['new_u','new_v','geom_reassigned','geom_linear']] = reassignNodes(gdf_edges,mergeIdx)
    # NOTE - no need to discard old nodes! will automatically be removed when merging nodes
    # gdf_nodes = gdf_nodes[(gdf_nodes.osmid.isin(gdf_edges.new_u)) | (gdf_nodes.osmid.isin(gdf_edges.new_v))]
    
    print("Completed step 8 in %s"%(round(time.time()-start,2)))
    if visualize: 
        make_plot(gdf_edges.geom_reassigned,gdf_nodes.geom_merged,bounds=plot_bounds, edge_col='geom_reassigned', node_col='geom_merged') 
        make_plot(gdf_edges[['geom_linear','length']],gdf_nodes.geom_merged,bounds=plot_bounds, edge_col='geom_linear', node_col='geom_merged',title='After step 8 - Linear') 
    
    ####################################
    # 9. (DISABLED) remove degree 2 nodes
    # ----------------------------------
    # remove nodes of degree 2 --> simplifies the network further
    # would have to merge tags accordingly
    # NOTE: ERRONOUS... CHECK BEFORE ENABLEING
    start = time.time()
    ####################################

    removeDeg2NodesVal = False
    if removeDeg2NodesVal:
        gdf_edges,gdf_nodes = removeDeg2Nodes(gdf_edges,gdf_nodes)
        # make_plot(gdf_edges.geom_linear,gdf_nodes.geom_merged) 
    else:
        gdf_edges,gdf_nodes = gdf_edges.copy(),gdf_nodes.copy()

    print("Completed step 9 in %s"%(round(time.time()-start,2)))
    
    ####################################
    # 10. merge nodes
    # ----------------------------------
    # remove/merge unused nodes
    start = time.time()
    ####################################

    # preparation
    # convert strings and lists to numbers, where applicable
    gdf_nodes.old_osmid = gdf_nodes.old_osmid.apply(lambda x: clean(x,asFloat=True))
    for c in gdf_nodes.columns:
        if set(type(a) for a in gdf_nodes[c])=={str}:
            gdf_nodes[c] = [clean(a) for a in gdf_nodes[c].values] #gdf_edges[c].apply(lambda x: clean(x))
    # convert None to np.nan (type float)
    gdf_nodes.fillna(value=np.nan,inplace=True)    

    # perform merging
    nodes = mergeNodes(gdf_nodes)
    
    print('\tn:',len(gdf_nodes),'-->',len(nodes))
    print("Completed step 10 in %s"%(round(time.time()-start,2)))
    
    ####################################
    # 11. merge edges
    # ----------------------------------
    # merge edges between a node U and a node V
    start = time.time()
    ####################################

    # preparation
    gdf_edges['osmid_i'] = gdf_edges.osmid
    gdf_edges['bearing'] = gdf_edges.geom_linear.apply(lambda x: np.angle(complex(*(np.array(x.coords)[1] - np.array(x.coords)[0])), deg=True) if x.geom_type=='LineString' else None)
    # convert None to np.nan (type float)
    gdf_edges.fillna(value=np.nan,inplace=True)    
    gdf_edges.replace('none',np.nan,inplace=True,regex=True)    
    # convert strings and lists to numbers, where applicable
    gdf_edges.maxspeed = gdf_edges.maxspeed.apply(lambda x: clean(x,asFloat=True,keep='max'))
    gdf_edges.old_osmid = gdf_edges.old_osmid.apply(lambda x: clean(x,asFloat=True))
    gdf_edges.lanes = gdf_edges.lanes.apply(lambda x: clean(x,asFloat=True,keep='max'))
    gdf_edges.width = gdf_edges.width.apply(lambda x: clean(x,asFloat=True,keep='min'))
    for c in gdf_edges.columns:
        if set(type(a) for a in gdf_edges[c])=={str}:
            gdf_edges[c] = [clean(a) for a in gdf_edges[c].values] #gdf_edges[c].apply(lambda x: clean(x))
    
    # NOTE: RECENT CHANGE
    gdf_edges.reset_index(drop=True,inplace=True)
    # perform merging
    # deleted edges --> edges that are not merged. Their information is 'lost'.
    if parallelized: 
        # links, deleted_edges = run_mergeEdgesWithSameNodes_in_parallel(gdf_edges)
        links, _ = run_mergeEdgesWithSameNodes_in_parallel(gdf_edges)
    else:
        # links, deleted_edges = mergeEdgesWithSameNodes(gdf_edges)
        links, _ = mergeEdgesWithSameNodes(gdf_edges)
    
    print('\te:',len(gdf_edges),'-->',len(links))
    print("Completed step 11 in %s"%(round(time.time()-start,2)))
    
    ####################################
    # 12. convert merged objects to dict to dataframes
    # ----------------------------------
    # convert to dict and then to df for 1+ modes
    start = time.time()
    ####################################

    gdfNodes = getNodeDict(nodes)
    gdfEdges = getEdgeDict(links)
    
    # make edges ids (g_id) unique
    gdfEdges = gdfEdges.reset_index(drop=True)
    gdfEdges['g_id'] = gdfEdges.index

    # Various variants (empty columns are not included)
    print('\tNumber of nodes:')
    print('\tAll\t\t',    len(getNodeDict(nodes)))
    print('\tNumber of edges per network version:')
    print('\tAll\t\t',    len(getEdgeDict(links)))
    print('\tWalk\t\t',   len(getEdgeDict(links, modes=['walk'])))
    print('\tBike\t\t',   len(getEdgeDict(links, modes=['bike'])))
    print('\tWalk+Bike\t',len(getEdgeDict(links, modes=['walk','bike'])))
    print('\tCar\t\t',    len(getEdgeDict(links, modes=['motorized'])))
    
    print("Completed step 12 in %s"%(round(time.time()-start,2)))
    
    ####################################
    # 13. convert & save
    # ----------------------------------
    # convert back to gpkg
    start = time.time()
    ####################################

    # swap u/v and reverse geometries, where reverse==1
    # such that plotting and map matching is accurate later on
    idcs = gdfEdges[gdfEdges.g_reversed==True].index.values
    tmp_u = [int(u) for u in gdfEdges.loc[idcs,'g_u']]
    tmp_v = [int(v) for v in gdfEdges.loc[idcs,'g_v']]
    gdfEdges.loc[idcs,'g_u'] = tmp_v
    gdfEdges.loc[idcs,'g_v'] = tmp_u
    gdfEdges.loc[idcs,'g_geometry'] = gdfEdges.loc[idcs,'g_geometry'].apply(lambda x: reverse_geom(x))
    gdfEdges.loc[idcs,'g_geo_lin'] = gdfEdges.loc[idcs,'g_geo_lin'].apply(lambda x: reverse_geom(x))
    gdfEdges.loc[idcs,'g_geo_rea'] = gdfEdges.loc[idcs,'g_geo_rea'].apply(lambda x: reverse_geom(x))
    gdfEdges.loc[idcs,'g_reverse'] = False
    gdfEdges = gdfEdges.drop(columns=['g_geometry'])
    
    make_plot(gdfEdges.g_geo_lin,gdfNodes.geometry,title='Final - linear',bounds=None, edge_col='g_geo_lin', node_col='geometry') 
    make_plot(gdfEdges.g_geo_rea,gdfNodes.geometry,title='Final - reassigned',bounds=None, edge_col='g_geo_rea', node_col='geometry') 
    
    # update edge column types
    if CFgeom_col == 'geometry_reassigned':
        gdfEdges.set_geometry(col='g_geo_rea',inplace=True)
    cols =  [c for c in gdfEdges.columns if (gdfEdges.dtypes[c]=='object')]
    if CFgeom_col == 'geometry_reassigned':
        cols.extend(['g_geo_lin']) 
    else:
        cols.extend(['g_geo_rea'])
    for c in cols:
        gdfEdges[c] = gdfEdges[c].astype('string')
    
    # update node column types
    cols =  [c for c in gdfNodes.columns if (gdfNodes.dtypes[c]=='object')]
    for c in cols:
        gdfNodes[c] = gdfNodes[c].astype('string')
    
    # SAVE - check if CFp3_result_filepath exists, otherwise create the folder
    if not os.path.exists(CFp3_result_filepath):
        os.makedirs(CFp3_result_filepath)
    print('Files saved to: ') 
    print('\t',CFp3_result_filepath+str(CFversion)+'_edges'+'.shp')
    print('\t',CFp3_result_filepath+str(CFversion)+'_nodes'+'.shp')
    gdfEdges.to_file(CFp3_result_filepath+str(CFversion)+'_edges'+'.shp')
    gdfNodes.to_file(CFp3_result_filepath+str(CFversion)+'_nodes'+'.shp')
    
    print("Completed step 13 in %s"%(round(time.time()-start,2)))
    print("Completed in %s"%(round(time.time()-startX,2)))

if __name__ == '__main__':    
    main()
    
# %%

