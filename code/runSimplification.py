# file to run all the steps necessary for the simplification
# %%
import os
import sys
# shapely / pandas depreciation warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.find_common_type.*")
# because of a geopandas warning
os.environ['USE_PYGEOS'] = '0'

# import the python files
import p1_getOSMNetwork
import p1_getFurtherOSMData
import p1_getOtherData
import p2_enrichData
import p3_simplification


if __name__ == "__main__":
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

