# file to run all the steps necessary for the simplification
# %%
import os
import sys
# because of a geopandas warning
os.environ['USE_PYGEOS'] = '0'

if __name__ == "__main__":

    # make sure all required packages are installed using the requirements.txt file
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()
    for r in requirements:
        try:
            exec("import " + r.split("==")[0])
        except ImportError:
            print("Package " + r + " not found. Please install all required packages listed in the requirements.txt file.")

    # import the python files
    import p1_getOSMNetwork
    import p1_getFurtherOSMData
    import p1_getOtherData
    import p2_enrichData
    import p3_simplification

    # add the current directory to the path
    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(currentDirectory)
    
    # run the steps
    p1_getOSMNetwork.main()
    p1_getFurtherOSMData.main(ptstops=False, amenities=True, buildings=True, landuse=True, retail=True, signals=True)
    p1_getOtherData.main(elevation=False, bike_traffic=False, accidents=False, district_centers=False)
    p2_enrichData.main(public_transport=False, accidents=False, cycle_path_width=False)
    p3_simplification.main()
    print("\nAll steps have been executed.")

# %%

