from setuptools import setup, find_packages

setup(
    name='OSMNetFusion',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'contextily==1.6.2',
        'fiona==1.10.1',
        'geopandas==1.0.1',
        'matplotlib==3.9.2',
        'numpy==2.1.2',
        'osmium==4.0.2',
        'osmnx==1.9.1',
        'overpy==0.7',
        'pandas==2.2.3',
        'psutil==6.0.0',
        'pyproj==3.7.0',
        'Requests==2.32.3',
        'scikit_learn==1.5.2',
        'scipy==1.14.1',
        'setuptools==75.2.0',
        'Shapely==2.0.6',
        'pytest==7.4.0',
    ],
    entry_points={
        'console_scripts': [
            'runSimplification=osmnetfusion.runSimplification:main',
        ],
    },
    author='Victoria Dahmen',
    author_email='v.dahmen@tum.de',
    description='A tool to generate a multimodal topologically-simplified OpenStreetMap network. The network is enriched with open-source data and tag-information is preserved.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/VictoriaDhmn/OSMNetFusion',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
