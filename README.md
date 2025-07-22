# SHERPA-training
Here you find the code to train the Source Receptor Relationship (SRR) used in the SHERPA model.
More details on SHERPA model can be found at: https://aqm.jrc.ec.europa.eu/Section/Sherpa/Background.

SRR (as the ones used in SHERPA) are statistical models that replicate the behavour of a full Chemical Transport Model (CTM).
SRR are used when a speedy version of a full CTM is required, to perform multiple simulations of emission reduction scenarios with a limited amount of time and resources.

This code has been updated in July 2025 to train SRR to be used for the 2025 version of the PM2.5 Atlas (https://publications.jrc.ec.europa.eu/repository/handle/JRC134950) preparation.
The code has been then revised in July 2025, to allow also for the computation of seasonal SRRs (i.e. for the period December-January-February, March-April-May, June-July-August, September-October-November).

The default version of SHERPA uses EU wide data for training.
It works at 0.1x0.05 deg (around 6x6km) spatial resolution, and uses as input:
1) basecase emissions from CAMS v8.0 including condensables, for the year 2022;
2) various emission reduction scenarios, to be used for training and validation of the SRR;
3) basecase concentrations of various pollutants (mainly PM2.5, PM10, O3, NO2, SO2) as produced by the EMEP v4.45 air quality model, using CAMS emissions;
4) various concentration scenarios simulated by the EMEP model on the aforementioned emission reduction scenarios. 

The data to train the European version of SHERPA are available on request.

This code has also been used to train 'local' versions of SHERPA (based on other input dataset), on domains including areas in Poland, Italy, Slovenia and China.

Finally, this code has been recently modified and extended, to allow for a separate treatement of 'low' and 'high' level sources (i.e. a different treatment for ground level emissions, and point source emissions), and to train seasonal SRRs.

# What is SHERPA
SHERPA (Screening for High Emission Reduction Potential on Air) is a Python tool, which allows for a rapid exploration of potential air quality improvements resulting from national/regional/local emission reduction measures. The tool has been developed with the aim of supporting national, regional and local authorities in the design and assessment of their air quality plans. The tool is based on the relationships between emissions and concentration levels, and can be used to answer the following type of questions:
1) What is the potential for local action in my domain?
2) What are the priority activity, sectors and pollutants on which to take action and,
3) What is the optimal dimension that my policy action domain (city, region…) should have to be efficient?"

The SHERPA tool is distributed with EU-wide data on emissions and source-receptor models (spatial resolution of roughly 6x6 km2), so that it is very easy to start working on any region/local domain in Europe.
You can freely access SHERPA at https://jeodpp.jrc.ec.europa.eu/eu/dashboard/voila/render/SHERPA/Sherpa.ipynb (you only need an EU login to do so).

More specifically, SHERPA logical pathway is implemented through the following steps:
1) Source allocation: to understand how the air quality in a given area is influenced by different sources (both sectoral and geographical);
2) Scenario analysis: to simulate the impact on air quality of a specific emission reduction scenario (defined also through the previous two steps)

# Selected publications

For a full and updated list of papers, go to: https://aqm.jrc.ec.europa.eu/Section/Sherpa/Document.

Here below few papers, as example of the produced work:

- Pisoni E., Zauli-Sajani S., Belis C.A., Khomenko S., Thunis P., Motta C., Van Dingenen R., Bessagnet B., Monforti-Ferrario F., Maes J., Feyen L.
High resolution assessment of air quality and health in Europe under different climate mitigation scenarios
(2025) Nature Communications , 16 (1), art. no. 5134
https://www.scopus.com/inward/record.uri?eid=2-s2.0-105007185788&doi=10.1038%2fs41467-025-60449-2&partnerID=40&md5=0e12d55e732355e69b53a7e452185bed

- Thunis P., Pisoni E., Zauli-Sajani S., de Meij A.
Comparison of source apportionment targeting hot-spot concentration and average population exposure
(2025) Science of the Total Environment, 968, art. no. 178857, Cited 0 times.
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85218139948&doi=10.1016%2fj.scitotenv.2025.178857&partnerID=40&md5=c6caa747fd9cbf4b8c677b9150b51ac4

- Zauli-Sajani S., Thunis P., Pisoni E., Bessagnet B., Monforti-Ferrario F., De Meij A., Pekar F., Vignati E.
Reducing biomass burning is key to decrease PM2.5 exposure in European cities
(2024) Scientific Reports, 14 (1), art. no. 10210, Cited 16 times.
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85191971193&doi=10.1038%2fs41598-024-60946-2&partnerID=40&md5=499bd09964bd0983d4e83a4c2882b619

- Pisoni E., De Marchi D., di Taranto A., Bessagnet B., Zauli Sajani S., De Meij A., Thunis P.
SHERPA-Cloud: An open-source online model to simulate air quality management policies in Europe
(2024) Environmental Modelling and Software, 176, art. no. 106031, Cited 6 times.
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85189674063&doi=10.1016%2fj.envsoft.2024.106031&partnerID=40&md5=8c359f063d3f77817707d15a4d99eb47

- Verweij R.W., Pisoni E., van Pul A., Thunis P., van der Swaluw E.
A fast method for ammonia emission reduction scenario studies: Combining EMEP4NL with the SHERPA tool
(2023) Atmospheric Environment, 306, art. no. 119805, Cited 2 times.
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85158873253&doi=10.1016%2fj.atmosenv.2023.119805&partnerID=40&md5=ec95b52abf00139e497a4f0b815b2ded

- Khomenko S., Pisoni E., Thunis P., Bessagnet B., Cirach M., Iungman T., Barboza E.P., Khreis H., Mueller N., Tonne C., de Hoogh K., Hoek G., Chowdhury S., Lelieveld J., Nieuwenhuijsen M.
Spatial and sector-specific contributions of emissions to ambient air pollution and mortality in European cities: a health impact assessment
(2023) The Lancet Public Health, 8 (7), pp. e546 - e558, Cited 38 times.
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85163451629&doi=10.1016%2fS2468-2667%2823%2900106-8&partnerID=40&md5=cbb336454ed4268a2251d142b0680ef5

- Bessagnet B., Pisoni E., Thunis P., Mascherpa M.,
Design and implementation of a new module to evaluate the cost of air pollutant abatement measures
(2022) Journal of Environmental Management, 317, 115486.
https://www.sciencedirect.com/science/article/pii/S0301479722010593

- Degraeuwe, B., Pisoni, E., Thunis, P.
Prioritising the sources of pollution in European cities: Do air quality modelling applications provide consistent responses?
(2020) Geoscientific Model Development, 13 (11), pp. 5725-5736. 
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85096928556&doi=10.5194%2fgmd-13-5725-2020&partnerID=40&md5=d4cb7a3413af75830b782af529db3727

- Pisoni, E., Thunis, P., Clappier, A.
Application of the SHERPA source-receptor relationships, based on the EMEP MSC-W model, for the assessment of air quality policy scenarios
(2019) Atmospheric Environment: X, 4, art. no. 100047, . 
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85072582264&doi=10.1016%2fj.aeaoa.2019.100047&partnerID=40&md5=34146ae8e90b2bc98ed4babfea0991a3

- Pisoni, E., Albrecht, D., Mara, T.A., Rosati, R., Tarantola, S., Thunis, P.
Application of uncertainty and sensitivity analysis to the air quality SHERPA modelling tool
(2018) Atmospheric Environment, 183, pp. 84-93. 
https://www.scopus.com/inward/record.uri?eid=2-s2.0-85045688029&doi=10.1016%2fj.atmosenv.2018.04.006&partnerID=40&md5=9432074fa33ac072995b6076f64cce3c

- Thunis, P., Degraeuwe, B., Pisoni, E., Ferrari, F., Clappier, A.
On the design and assessment of regional air quality plans: The SHERPA approach
(2016) Journal of Environmental Management, 183, pp. 952-958. 
https://www.scopus.com/inward/record.uri?eid=2-s2.0-84994012263&doi=10.1016%2fj.jenvman.2016.09.049&partnerID=40&md5=1561546680304fcf57e914bdf441d452

