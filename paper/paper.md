---
title: 'OSMNetFusion: Topological Simplification for Multimodal Networks with Attribute-Preservation and Enrichment'
tags:
  - Python
  - network simplification
  - network topology
  - OpenStreetMap
  - spatial analyses
authors:
  - name: Victoria Dahmen
    orcid: 0009-0004-0392-2526
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
#   - name: ...
#     orcid: ...
#     equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
#     affiliation: 1
affiliations:
 - name: Chair of Traffic Engineering and Control, Technical University of Munich, Germnay
   index: 1
date: 27 June 2025
bibliography: paper.bib

---

# Summary

The `OSMNetFusion` package is a tool designed to topologically simplify and enrich OpenStreetMap (OSM) networks to facilitate their use for further mobility or geospatial analyses. It provides researchers, transport analysts and urban planners with a simplified, information-rich, multimodal network that retains essential information for various transport applications. With only the location of interest as required input, this tool reduces network complexity by merging edges (roads) and nodes (intersections) and preserves essential tag information, while also adding additional open-source data. Additionally, the resulting network supports the effortless selection of transport-mode-specific subnetworks (e.g., walking, cycling, motorised), making them easier to analyse and visualise.


# Statement of need

`OSMNetFusion` offers a powerful approach to creating a streamlined yet information-rich network, which is multimodal, versatile, and well-suited for spatial network analyses, transportation planning, and mobility services. By merging different types of OSM data with external geographic data and then simplifying the network topology without reduced loss of attribute-information, this framework enables efficient, transport-mode-specific analyses. For instance, `OSMNetFusion`'s network simplification and high information density has already been used forassessing route choice of pedestrians or cyclists, which is affected by many variables `[@Dahmen:2025]`. Similarly, perceived stress levels has also been assessed `[@Takayasu:2024]`. Furthermore, it can aid mobility service providers to strategically place stations by mapping accessibility based on factors like bike racks and elevation. As the mode-specific networks can be selected with ease, the use cases are versatile.

This Python package that leverages parallel programming to reduce runtime (if supported by the local machine). The package can simply be cloned, built, and then imported. The config file contains key parameters that can be set as desired, and a wide range of additional parameters that can be altered to, e.g., change the degree of simplification.

Several existing open-source tools address network simplification for specific use-cases in geospatial and mobility analysis. For instance, Ballo and Axhausen developed a road space reallocation tool `[@Ballo:2024b]`, \textit{SNMan}, which includes a topological network simplification with a reconstruction and visualisation of the lanes per link `[@Ballo:2024a]`. Fleischmann and Vybornova focus on topological network simplification in a mathematically rigorous manner `[@Fleischmann:2024]` and recently published `neatnet` `[@Fleischmann:2025]`. Tools such as OSMnx `[@Boeing:2024]` also offer simplification capabilities to a limited extent. However, \texttt{OSMNetFusion} specifically aims to simplify multimodal networks (pedestrian, cyclist, motorised traffic) while retaining OSM tag information and integrating additional open-source data, an aspect often absent in current solutions but crucial for comprehensive mobility analysis.


# Methodology

The framework is composed of three key steps: the data retrieval, the network enrichment, and the network simplification. These are described in the subsequent sections.

## Data retrieval

OSM networks contain a wide range of metadata tags covering information from road surface to speed limits and lighting conditions. The OSM platform also contains many items other than the road network, such as information about land-use, public transit, restaurants and mobility-related infrastructure. Additionally, there also exists other relevant open-source data. Hence, data is gathered from all three categories. As each data is either network-specific or valid for the entire region of interest, the data is saved in two folders depending on this characteristic.

## Network enrichment

All the downloaded data is added to the OSM network to creates an enriched, detailed network that captures more than just the road layout. This multi-layered network provides the foundation for the last step, where the network is simplified (consolidated) to produce a model that supports various modes of transport and retains essential topographic and infrastructure characteristics. 

## Network simplification

The topological simplification (and subsequent merging) consists of 8 key steps. The entire process is visualised in Fig. \autoref{fig:simplSteps}, where a simple example of the aim of the topological simplification is shown in Fig. \autoref{fig:example}.

![Simplification steps.\label{fig:simplSteps}](../visualisations/Vis_simplificationSteps.png)

![The resulting network.\label{fig:example}](../visualisations/Vis_ExampleResult.png)

## Information Preservation and Tag Restructuring

It is no trivial task to merge nodes or edges which have a wide range of information associated to them. Hence, the structure visualised in Figure \autoref{fig:attrStructure} has been conceptualised. For each consolidated edge, there is a set of attributes that are associated with a specific mode (bike, walk, PT, car) and a set of general/high-level attributes (like the object ID). Additionally, four tags denote the accessibility of an edge to the key modes of transport, ensuring the easy selection of mode-specific subnetworks. For a given link, each edge is directional, so if the link is bidirectional (i.e., not one-way for all modes that may access the link), \textit{Edge UV} and \textit{Edge VU} are separate entities. % Each edge object retains specific metadata relevant to its allowed modes of travel. The object-oriented approach reflects this design. Each link object has one or two edges, which in turn may have walking, cycling, and/or motorized path objects.

![Structure of the information of the link (black arrow) and edge objects (orange arrows).\label{fig:attrStructure}](../visualisations/Vis_AttrStructure.png){width=20%}

The consolidation process of the node/edge of all attributes (including the geometries) is explained in depth in the documentation. The result is a compact, well-organised network that is both efficient for computational tasks and accurate and information-rich for practical applications.


# Acknowledgements

Victoria Dahmen acknowledges funding by the TUM Georg Nemetschek Institute Artificial Intelligence for the Built World, and support from Margarita Piscenco on refactoring parts of the code.


# References