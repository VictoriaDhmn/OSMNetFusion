# configFile.py

# ------------------------------------------------------------------------
# ADJUST LOCATION --------------------------------------------------------
# ------------------------------------------------------------------------

# PLEASE ADJUST THE BELOW VARIABLES 

# 1. FOR VERSION CONTROL
version = 'Frankfurt' 

# 2. FOR QUERYING OSM DATA 
# Assign city and country values using a predefined city or custom input
city_dict = {   'Munich': {'city': 'Munich', 'country': 'Germany', 'city_OSM': 'München'},
                'Tegernsee': {'city': 'Tegernsee', 'country': 'Germany', 'city_OSM': 'Tegernsee'},
                'Auckland': {'city': 'Auckland', 'country': 'New Zealand', 'city_OSM': 'Auckland'},
                'Düsseldorf': {'city': 'Düsseldorf', 'country': 'Germany', 'city_OSM': 'Düsseldorf'},
                'Frankfurt': {'city': 'Frankfurt', 'country': 'Germany', 'city_OSM': 'Frankfurt am Main'},
                'Freiburg': {'city': 'Freiburg', 'country': 'Germany', 'city_OSM': 'Freiburg im Breisgau'} }
predefined_city = True
if predefined_city:
    city_info = city_dict['Freiburg']
else:
    city_info = {'city': 'Munich', 'country': 'Germany', 'city_OSM': 'München'}  

# 3. FOR SETTING THE NETWORK BOUNDARIES
# Select 'place' or 'coordinates'
boundary_mode = 'place'
if boundary_mode == 'place':
    location = f'{city_info["city_OSM"]}, {city_info["country"]}' 
    dist_in_meters = 1500 # Radius in meters
else:
    coords_upper_left = 48.15847740556768, 11.556108918739799 
    coords_lower_right = 48.152471559463216, 11.567313793292625

# Only set this to True if the PT stops/routes cannot be retrieved with the OSM API
manual_OSM_PT_query = False

# Use 'network_data' when using the package approach 
# When manually running runSimplification.py, use '../../network_data/'
network_data_dir = 'network_data/'  
regional_data_dir = 'regional_data/'

# ------------------------------------------------------------------------
# ONLY ADJUST THE BELOW WHEN NECESSARY -----------------------------------
# ------------------------------------------------------------------------

# Place information used for OSM queries
place = {"city": city_info['city'], "country": city_info['country']}

# p1_getOSMNetwork.py
p1_result_filepath = network_data_dir + f"{version}/p1_{version}_osmnx.gpkg"
# Define useful OSM tags that should be included in nodes and edges
used_tags = ['osmid',
                     'highway',  # used for cycleway_category | used in network_type = bike
                     'cycleway',  # used for cycleway_category | used in network_type = bike
                     'cycleway:both',  # used for cycleway_category | used in network_type = bike
                     'cycleway:both:bicycle',
                     'cycleway:left',  # used for cycleway_category | used in network_type = bike
                     'cycleway:left:oneway',  # used for cycleway_category | used in network_type = bike
                     'cycleway:left:bicycle',
                     'cycleway:left:segregated',
                     'cycleway:right',  # used for cycleway_category | used in network_type = bike
                     'cycleway:right:oneway',  # used for cycleway_category | used in network_type = bike
                     'cycleway:right:bicycle',
                     'cycleway:right:segregated',
                     'cycleway:lane',  # used in network_type = bike
                     'cycleway:right:lane',
                     'cycleway:left:lane',
                     'bicycle',  # used for cycleway_category | used in network_type = bike
                     'bicycle:forward',  # used for cycleway_category | used in network_type = bike
                     'bicycle:backward',  # used for cycleway_category | used in network_type = bike
                     'cyclestreet',  # used for cycleway_category
                     'bicycle_road',  # used for cycleway_category | used in network_type = bike
                     'oneway',  # used for cycleway_category | used in network_type = bike
                     'oneway:bicycle',  # used for cycleway_category | used in network_type = bike
                     'bicycle_parking',  # used in network_type = bike
                     'class:bicycle',  # used in network_type = bike
                     'lanes',  # used in network_type = bike
                     'maxspeed',  # used in network_type = bike
                     'surface',  # used in network_type = bike
                     'cycleway:surface',  # used in network_type = bike
                     'smoothness',  # used in network_type = bike
                     'cycleway:smoothness',  # used in network_type = bike
                     'tracktype',  # used in network_type = bike
                     'foot',  # used in network_type = bike
                     'segregated',  # used in network_type = bike
                     'crossing',  # used in network_type = bike
                     'width',  # used in network_type = bike
                     'cycleway:width',  # used in network_type = bike
                     'cycleway:buffer',  # n no Data
                     'ramp:bicycle',  # used in network_type = bike
                     'public_transport',  # used in network_type = bike & to track bus stops
                     'route',  # no Data
                     'barrier',  # used in network_type = bike
                     'obstacle',  # no Data
                     'incline',
                     'lit',
                     'traffic_signals',
                     'sidewalk',
                     'parking:lane:both',
                     'parking:lane:left',
                     'parking:lane:right',
                     ]

# p1_getFurtherOSMData.py
# -----------------PT Stops-----------------------
pt_stops_tags = {   "amenity": ['bus_station'],
                    "highway": ['bus_stop', 'platform'],
                    "public_transport": ['station', 'stop_position', 'platform'],
                    "railway": ['station', 'stop', 'tram_stop'], "station": ['subway', 'light_rail', 'tram_stop']}
pt_stops_cols = ['geometry', 'name', 'osmid', 'public_transport', 'highway', 'railway', 'station']
pt_stops_filepath = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_PTStops.gpkg"
manual_OSM_query_fp = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_PTStops_query.xml"
# -----------------bike amenities----------------------
bike_amenities_tags = { "amenity": ['bicycle_parking', 'bicycle_repair_station', 'bicycle_rental'],
                        "vending": ['bicycle_tube']}
bike_amenities_cols = ['amenity', 'bicycle_parking', 'bike_ride', 'capacity', 'covered', 'geometry', 'operator', 'network']
bike_amenities_filepath = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_bicycle_amenities.gpkg"
# -----------------building landuse--------------
building_tags = {"building": True}
building_cols = ['building', 'geometry', 'height', 'est_height']
building_filepath = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_buildings.gpkg"
# -----------------green landuse----------
green_landuse_cols = ['geometry', 'landuse', 'leisure', 'natural']
green_landuse_tags = {  "landuse": ['allotments', 'animal_keeping', 'apiary', 'cemetery', 'farmland', 'farmyard',
                        'flowerbed', 'forest', 'grass', 'green', 'orchard', 'village_green', 'vineyard', 'greenfield',
                        'greenhouse_horticulture', 'landfill', 'meadow'],
                        "leisure": ['garden', 'nature_reserve', 'park', 'pitch', ], "natural": True,
                        'waterway': ['river', 'stream', 'tidal_channel', 'canal', 'ditch']}
green_landuse_filepath = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_landuse_leisure_natural.gpkg"
# -----------------retail landuse----------
retail_cols = ['geometry', 'landuse', 'shop', 'amenity', 'building']
retail_tags = { "landuse": ['commercial', 'retail'],
                "shop": True,  
                "amenity": ['marketplace', 'mall', 'bank', 'restaurant', 'cafe', 'fast_food', 'bar', 'pub', 'cinema', 'theatre'],
                "building": ['retail', 'supermarket', 'shopping_center']
}
retail_filepath = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_landuse_retail.gpkg"
# -----------------traffic signals----------
signals_cols = ['id','@id','highway','crossing','bicycle','traffic_signals','traffic_signals:direction','traffic_signals:operating_times','geometry']
signals_tags = {"highway": "traffic_signals"}
signals_filepath = regional_data_dir + f"{city_info['city']}/osmnx/osmnx_traffic_signals.geojson"

# p1_getOtherData.py
elev_filepath = network_data_dir + f"{version}/add_data/{version}_elevations.json"

# p2_enrichData.py
cycle_path_w_filepath = network_data_dir + f"{version}/add_{version}_cycle_path_widths.csv"
p2_result_filepath = network_data_dir + f"{version}/p2_{version}_enriched.gpkg"

# p3_simplification.py
# 1. Filepath of output network 
# p3_result_filepath = network_data_dir + f"{version}/p3_simplified_data/" 
p3_result_filepath_gpkg = network_data_dir + f"{version}/p3_{version}_simplified.gpkg"
# 2a. Set input crs and output geometry mode
crs = "EPSG:4326"   #  this CRS corresponds to UTM 32N
# geometry_reassigned:  end points of an edge are reassigned to the nearest node cluster
# geometry_linear:      straight line between the end points of an edge
geom_col = 'geometry_reassigned' 
# 2b. Visualise intermediate steps? plot bounds = [minx, maxx, miny, maxy]
visualize = False
plot_bounds = [11.5562674, 11.5605949, 48.1469797, 48.1489210,]
# plot_bounds = [11.592781898558341, 11.589610747767697, 48.12932550391191, 48.13145017242847]
# 3. Should / can the code run in parallel?
#   Use parallelized = True for better simplification results and to speed up the proces.
#   Use parallelized = False if this approach is not compatible with your setup, 
#   or if you need to look at or print intermediate outputs of the parallel functions
parallelized = True 
# 4. Edges / nodes will be sorted based on this ranking
HIGHWAY_RANKING = {
    'trunk':10,
    'trunk_link':9.5,
    'primary':9.25,#9
    'secondary':9,
    'secondary_link':8.5,
    'tertiary':8,
    'residential':7,
    'cycleway':6,
    'path':5.5,
    'footway':5, #4
    'pedestrian':4.5,
    'service':4, #5
    'steps':3.5,#3
    'bridleway':3
}
# 5. Buffers for first clustering --> nodes are buffered based on the highest ranking connected highway type
clusterThreshold = 50
HIGHWAY_BUFFERS_1 = {'trunk':18,
        'trunk_link':18,
        'primary':18,
        'secondary':16, 
        'secondary_link':16,
        'tertiary':14,
        'residential':12,
        'cycleway':12,
        'path':10,
        'footway':10,
        'pedestrian':10,
        'service':6,
        'steps':6,
        'bridleway':6,
        'all_others': 4
    } 
# HIGHWAY_BUFFERS_1 = {'trunk':22,
#         'trunk_link':22,
#         'primary':22,
#         'secondary':20, 
#         'secondary_link':20,
#         'tertiary':15,
#         'residential':14,
#         'cycleway':14,
#         'path':10,
#         'service':8,
#         'footway':8,
#         'pedestrian':8,
#         'steps':8,
#         'bridleway':8,
#         'all_others': 8
#     } 
# 6. Buffers for second clustering 
HIGHWAY_BUFFERS_2 = HIGHWAY_BUFFERS_1
# 7. Split curves angle --> Curves are split into smaller, straighter segments based on the maximum angle between sub-segments.
# maxAngleInitial (float, optional): the maximum angle between the first and second segments of a curve to trigger a split (default 75)
maxAngleInitial=75
# maxAnglePrev (float, optional): the maximum angle between subsequent segments of a curve to trigger a split (default 60)
maxAnglePrev=60

