"""
Extract information from OpemStreetMap (OSM) using the osmnx library.
- get_city_boundary: get the boundary of the city
- generate_pt_stops_and_route_file: get PT stops and routes
- generate_landuse_file: get land use information
- generate_objects_file: get information about objects
- main: main function to run the script

Inputs:
- configFile.py: configuration file with all values
    - city_OSM: city name
    - place: location name, e.g. 'Munich, Germany'
    - stops_tags: tags to filter for PT stops
    - stops_cols: columns to keep for PT stops
    - stops_fp: filepath to save PT stops
    - manual_OSM_PT_query: True if manual query is needed
    - manual_OSM_query_fp: filepath to manual query
    - amenities_tags: tags to filter for bike amenities
    - amenities_cols: columns to keep for bike amenities
    - amenities_fp: filepath to save bike amenities
    - buildings_tags: tags to filter for buildings
    - buildings_cols: columns to keep for buildings
    - buildings_fp: filepath to save buildings
    - green_landuse_tags: tags to filter for green land use
    - green_landuse_cols: columns to keep for green land use
    - green_landuse_fp: filepath to save green land use
    - retail_tags: tags to filter for retail
    - retail_cols: columns to keep for retail
    - retail_fp: filepath to save retail
    - signals_tags: tags to filter for traffic signals
    - signals_cols: columns to keep for traffic signals
    - signals_fp: filepath to save traffic signals
Outputs:
- geopackage files with the extracted information (saved to the respective filepaths)
    - PT stops
    - bike amenities
    - buildings
    - green land use
    - retail
    - traffic signals
- plots of the extracted information
"""

# %%
import osmnx as ox
import overpy
import shapely as sh
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import osmium

# FUNCTIONS ##################################################################################

def get_city_boundary(place):
    """
    Get the boundary of the city.
    Args:
        @place: dict, center location, e.g. {"city": Munich, "country": Germany}
    Returns:
        boundary: shapely.geometry, boundary of the city
    """
    # Download the administrative boundary polygon for the city
    boundary = ox.geocode_to_gdf(place)
    if len(boundary)==0:
        print("No boundary found.")
        # simple doing 10km around the city center
        boundary = ox.geocode_to_gdf(place, buffer_dist=10000)
    if len(boundary)>1:
        print("Multiple boundaries found. Using the first one.")
        print(boundary)
    boundary = sh.geometry.box(*boundary.total_bounds)
    return boundary 

def generate_pt_stops_and_route_file(place, manual_query, output_file_stops):
    """
    Generates files with info on public transport routes and stops.
    Args:
        @place_: dict, center location, e.g. {"city": Munich, "country": Germany}
        @manual_query: str, filepath to manual query (only if data is too large for API, else None)
        @output_file: str, filepath to save the output
    """
    # extract the boundary of the city using the link provided: link_OSM_city
    boundary = get_city_boundary(place)

    # get the PT stops and routes
    print(place)
    modes = ['bus', 'tram', 'trolleybus']
    modes = '|'.join(modes)
    if manual_query is not None:
        # load geojson file such that it has the same format as the overpy result
        print(manual_query)
        # ---------------------
        class OSMDataHandler(osmium.SimpleHandler):
            def __init__(self):
                super(OSMDataHandler, self).__init__()
                self.nodes = {}
                self.ways = {}
                self.relations = {}
            def node(self, n):
                self.nodes[n.id] = {'lat': n.location.lat, 'lon': n.location.lon, 'tags': {tag.k: tag.v for tag in n.tags}}
            def way(self, w):
                self.ways[w.id] = {'nodes': [n.ref for n in w.nodes], 'tags': {tag.k: tag.v for tag in w.tags}}
            def relation(self, r):
                self.relations[r.id] = {
                    'members': [{'type': m.type, 'ref': m.ref, 'role': m.role} for m in r.members],
                    'tags': {tag.k: tag.v for tag in r.tags}
                }
        # the OSM PBF file
        file_path = manual_query
        handler = OSMDataHandler()
        handler.apply_file(file_path)
        result = {
            'nodes': handler.nodes,
            'ways': handler.ways,
            'relations': handler.relations
        }
        # ---------------------
        print("Relations: ", len(result['relations']))
        stops = []
        for key, relation in result['relations'].items():
            tags = relation['tags']
            # members = relation.members
            relation_mode = tags['route']
            # extract and save PT stops and platforms
            for member in relation['members']:
                member_ref = member['ref']
                if member['type']=='n':
                    node = result['nodes'][member_ref]
                    geometry = sh.geometry.Point(node['lon'], node['lat'])
                elif member['type']=='w':
                    way = result['ways'][member_ref]
                    geometry = sh.geometry.LineString([(result['nodes'][node]['lon'], result['nodes'][node]['lat']) for node in way['nodes']])
                else:
                    geometry = None
                stops.append({
                    "relation_id": key,
                    "mode": relation_mode,
                    "stop_type": member['role'],
                    "member_ref": member_ref,
                    "name": tags["name"],
                    "operator": tags["operator"] if "operator" in tags.keys() else None,
                    "ref": tags["ref"],
                    "network": tags["network"] if "network" in tags.keys() else None,
                    "from": tags["from"],
                    "to": tags["to"],
                    "geometry": geometry,
                })
    else:
        try:
            query = f"""area["name"="{place}"]["admin_level"="6"]->.searchArea;
                relation["type"="route"]["route"~"{modes}"](area.searchArea);
                out body;
                >;
                out skel qt;
                """
            api = overpy.Overpass()
            result = api.query(query)
            if len(result.relations) == 0:
                query = f"""area["name"="{place}"]["admin_level"="8"]->.searchArea;
                    relation["type"="route"]["route"~"{modes}"](area.searchArea);
                    out body;
                    >;
                    out skel qt;
                    """
                result = api.query(query)
            print("Relations: ", len(result.relations))
            stops = []
            for relation in result.relations:
                tags = relation.tags
                # members = relation.members
                relation_mode = tags.get("route", None)
                # extract and save PT stops and platforms
                for member in relation.members:
                    member_ref = member.ref
                    if isinstance(member, overpy.RelationNode):
                        node = result.get_node(member_ref)
                        geometry = sh.geometry.Point(node.lon, node.lat)
                    elif isinstance(member, overpy.RelationWay):
                        way = result.get_way(member_ref)
                        geometry = sh.geometry.LineString([(node.lon, node.lat) for node in way.nodes])
                    else:
                        geometry = None
                    stops.append({
                        "relation_id": relation.id,
                        "mode": relation_mode,
                        "stop_type": member.role,
                        "member_ref": member_ref,
                        "name": tags.get("name", None),
                        "operator": tags.get("operator", None),
                        "ref": tags.get("ref", None),
                        "network": tags.get("network", None),
                        "from": tags.get("from", None),
                        "to": tags.get("to", None),
                        "geometry": geometry,
                    })
        except:
            print(f"Overpass query is too large for the API.")
            print('##################################################')
            print('1. Please open this link: https://overpass-turbo.eu/')
            print('2. Copy the following query and run it there:')
            print(query)
            print('Note: you might have to use admin_level 4 or 8 instead of 6.')
            print('3. Download the data and save it here:', output_file_stops)
            print('4. In configFile, set the manual_OSM_query=True.')
            print('5. Rerun this script.')
            print('##################################################')
            return
    
    # process the result
    gdf_stops = gpd.GeoDataFrame(stops, crs="EPSG:4326")
    gdf_stops.set_geometry('geometry', inplace=True)
    
    # PLOT TO CHECK RESULTS
    fig, ax = plt.subplots()
    gdf_stops.plot(ax=ax, color='blue')
    gdf_stops[gdf_stops.geometry.within(boundary)].plot(ax=ax, color='orange')
    x, y = boundary.exterior.xy
    ax.plot(x, y, color='red')
    ax.set_xlim(boundary.bounds[0]-0.05, boundary.bounds[2]+0.05)
    ax.set_ylim(boundary.bounds[1]-0.05, boundary.bounds[3]+0.05)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:4326")
    plt.title('The PT stops/routes within this area are extracted.')
    plt.show()
    
    # only keep stops with 4+ stops within the boundary
    gdf_stops = gdf_stops[gdf_stops.geometry.within(boundary)]
    gdf_stops = gdf_stops[gdf_stops.member_ref.isin(gdf_stops.member_ref.value_counts()[gdf_stops.member_ref.value_counts() > 3].index)]
    print('stop types:', gdf_stops.stop_type.unique())
    
    # Save stops as GeoPackage
    for geom_type in gdf_stops.geom_type.unique():
        gdf_stops[gdf_stops.geom_type == geom_type].to_file(output_file_stops, driver="GPKG", layer=geom_type)
    print("{} stops written to file {}".format(gdf_stops.shape[0], output_file_stops))

def generate_landuse_file(place, output_file, tags, cols):
    """
    get OSM data regarding type of use of land
    Args:
        @place: dict, center location, e.g. {"city": Munich, "country": Germany}
        @tags: dict, tags to filter for, e.g. {"landuse": "residential"}
        @cols: list, columns to keep, e.g. ["name", "landuse", "geometry"]
    """
    # get all "landuse"
    gdf_landuse = ox.features_from_place(place, tags)
    
    # select useful columns that are available in gdf
    gdf_landuse = gdf_landuse[[x for x in cols if x in gdf_landuse.columns]]

    # prepare for saving as geopackage
    gdf_landuse = gdf_landuse.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)

    # save as geopackage
    for geomtype in gdf_landuse.geom_type.unique():
        gdf_landuse[gdf_landuse.geom_type == geomtype].to_file(output_file, driver="GPKG", layer=geomtype)
    print("{} rows of data regarding type of use of land written to file {}".format(gdf_landuse.shape[0], output_file))

def generate_objects_file(place, output_file, tags, cols=None, objects='objects'):
    """
    get OSM data regarding objects
    Args:
        @place: dict, center location, e.g. {"city": Munich, "country": Germany}
        @output_file: str, filepath to save the output
        @tags: dict, tags to filter for, e.g. {"amenity": "bicycle_parking"}
        @cols=None: list, columns to keep, e.g. ["name", "amenity", "geometry"]
        @objects='objects': str, object class, e.g. "bike_amenities"
    """
    # get all objects
    gdf = ox.features_from_place(place, tags)
    
    # select useful columns that are available in gdf
    if cols is not None:
        gdf = gdf[[x for x in cols if x in gdf.columns]]

    # save as geopackage
    gdf = gdf.apply(lambda c: c.astype(str) if c.name != "geometry" else c, axis=0)
    for geomtype in gdf.geom_type.unique():
        gdf[gdf.geom_type == geomtype].to_file(output_file, driver="GPKG", layer=geomtype)
    print("{} rows of data regarding {} written to file {}".format(gdf.shape[0], objects, output_file))

def main(configFile, ptstops=True, amenities=True, buildings=True, landuse=True, retail=True, signals=True):
    city_OSM = configFile.city_info['city_OSM']
    place = configFile.place
    stops_tags = configFile.pt_stops_tags
    stops_cols = configFile.pt_stops_cols
    stops_fp = configFile.pt_stops_filepath
    manual_OSM_query = None
    if configFile.manual_OSM_PT_query:
        manual_OSM_query = configFile.manual_OSM_query_fp
    amenities_tags = configFile.bike_amenities_tags
    amenities_cols = configFile.bike_amenities_cols
    amenities_fp = configFile.bike_amenities_filepath
    buildings_tags = configFile.building_tags
    buildings_cols = configFile.building_cols
    buildings_fp = configFile.building_filepath
    green_landuse_tags = configFile.green_landuse_tags
    green_landuse_cols = configFile.green_landuse_cols
    green_landuse_fp = configFile.green_landuse_filepath
    retail_tags = configFile.retail_tags
    retail_cols = configFile.retail_cols
    retail_fp = configFile.retail_filepath
    signals_tags = configFile.signals_tags
    signals_cols = configFile.signals_cols
    signals_fp = configFile.signals_filepath
    
    import os
    output_directory = os.path.dirname(stops_fp)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if ptstops:
        generate_pt_stops_and_route_file(place=city_OSM, manual_query=manual_OSM_query, output_file_stops=stops_fp)
    if amenities:
        generate_objects_file(place, amenities_fp, amenities_tags, cols=amenities_cols, objects='bike_amenities')
    if buildings:
        generate_landuse_file(place, buildings_fp, buildings_tags, buildings_cols)
    if landuse:
        generate_landuse_file(place, green_landuse_fp, green_landuse_tags, green_landuse_cols)
    if retail:
        generate_landuse_file(place, retail_fp, retail_tags, retail_cols)
    if signals:
        generate_objects_file(place, signals_fp, signals_tags, cols=signals_cols, objects='traffic signals')

if __name__ == "__main__":
    main()

