# file to run all the steps necessary for the simplification
# %%
import os
import sys
import importlib.util
# because of a geopandas warning
os.environ['USE_PYGEOS'] = '0'

# NOTE: see setup.py for the required packages

def runSimplification(fp_config=None, 
                      run_getOSMNetwork=True, run_getFurtherOSMData=True, run_getOtherData=True, run_enrichData=True, run_simplification=True,
                      ptstops=False, amenities=True, buildings=True, landuse=True, retail=True, signals=True, elevation=True, public_transport=False, cycle_path_width=False,
                      manual=False):
    
    # load configFile
    if fp_config is None:
        if manual==True:
            import configFile # when running this script manually
        else:
            from . import configFile # when running as a package
    else:
        configFile = load_config(fp_config)
    # check if the configFile is correctly set up
    if manual==True:
        if configFile.network_data_dir[:6]!='../../':
            print('Careful: if you are running runSimplification.py, you need to adjust the network_data_dir (and regional_data_dir) in the configFile when running the script locally: change "network_data/" to "../../network_data/".')
    else:
        if configFile.network_data_dir[:6]=='../../':
            print('Careful: if you are using the package, you may need to adjust the network_data_dir (and regional_data_dir) in the configFile from "../../network_data/" to "network_data/".')
    print('Version:',configFile.version)
    # add the current directory to the path
    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(currentDirectory)
    
    # run the steps
    if run_getOSMNetwork:
        p1_getOSMNetwork.main(configFile)
    if run_getFurtherOSMData:
        p1_getFurtherOSMData.main(configFile, ptstops=ptstops, amenities=amenities, buildings=buildings, landuse=landuse, retail=retail, signals=signals)
    if run_getOtherData:
        p1_getOtherData.main(configFile, elevation=elevation)
    if run_enrichData:
        p2_enrichData.main(configFile, public_transport=public_transport, cycle_path_width=cycle_path_width)
    if run_simplification:
        p3_simplification.main(configFile)
    print("\nAll steps have been executed.")

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("configFile", config_path)
    configFile = importlib.util.module_from_spec(spec)
    sys.modules["configFile"] = configFile
    spec.loader.exec_module(configFile)
    return configFile

if __name__ == "__main__":
    # import the python files
    import p1_getOSMNetwork
    import p1_getFurtherOSMData
    import p1_getOtherData
    import p2_enrichData
    import p3_simplification
    runSimplification(manual=True)
else:
    # import the python files
    from . import p1_getOSMNetwork
    from . import p1_getFurtherOSMData
    from . import p1_getOtherData
    from . import p2_enrichData
    from . import p3_simplification


# %%

