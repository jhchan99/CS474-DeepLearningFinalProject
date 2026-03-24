## Folders are organized as follows:

* **1_SecondaryData** contains all secondary information collected for each site in a single file (sites.csv) with the following columns:
	1. **SiteID**: Identifier assigned to each participant in the study. Type: Integer. 
	2. **City**: Name of the city where the participant lives. Type: Logan / Providence.
	3. **N_Residents**: Total number of residents in the site. Type: Integer.
	4. **N_Residents_0-10**: Number of residents in the age range listed. Type: Integer.
	5. **N_Residents_10-25**: Number of residents in the age range listed. Type: Integer.
	6. **N_Residents_25-40**: Number of residents in the age range listed. Type: Integer.
	7. **N_Residents_40-60**: Number of residents in the age range listed. Type: Integer.
	8. **N_Residents_Over60**: Number of residents in the age range listed. Type: Integer.
	9. **IrrigationType**: Descriptor of the irrigation system in the property. Type: Hose / SprinklerSystem
	10. **MeterBrand**: Manufacturer of the meter used by the utility company to record water use at the site. Type: Neptune / MasterMeter
	11. **MeterSize**: Size of the meter used by the utility company to record water use at the site. Units: Inches. 
	12. **MeterResolution**: Volumetric meter resolution for the meter located at the site. Units: Gallons. Multiplying pulses at any location by this value will result in the volume (in gallons) used at the site.
	13. **N_Bathrooms**: Number of bathrooms in the site. Type: Integer
	14. **Own/Rent**: Indicates if the habitants of a site own or rent the property. Type: Own / Rent. 
	15. **LegalAcreage_SqFt**: Legal area of the parcel. Units: Square Foot - Rounded to the nearest 10. 
	16. **YearBuilt**: Year the property was built. Unit: Year - rounded to the nearest 5. 
	17. **BuildingSqFt**: Built area of each site. Units: Square Foot - rounded to the nearest 10. 
	18. **ZipCode**: Zip code for each site. Type: 5-digit integer.
	19. **UserPercentile_City_LastYear**: Ranking of the participants sites compared with all residential properties, by city, for the last year of data available. Logan (2018), Providence (2019). Units: ranking.
	20. **MonthlyAverageWinter**: Average monthly water use during winter months (November - April) at each site. Units: 1,000 Gallons.
	21. **MonthlyAverageSummer**: Average water use during winter months (May - October) at each site. Units: 1,000 Gallons.
	22. **Irr_Area**: Irrigated Area, computed from satellite imagery. Units: Square Foot.


* **2_LogFiles** contains a log for each site (in CSV format) with the following columns: 
	1. **DataCollectionPeriod**: An identifier for each data collection period from 1 to n.
	2. **StartDate**: The start date of each data collection period. Format: 'YYYY-MM-DD HH:MM:SS'
	3. **EndDate**: The end date of each data collection period. Format: 'YYYY-MM-DD HH:MM:SS'
	4. **Meter_WaterUse**: Volume of water used in each data collection period, obtained from reading the actual meter installed at each site. Unit: Gallons.
	5. **CIWS-DL_WaterUse**: Volume of water used in each data collection period, measured by the CIWS-Datalogger installed at each site. Unit: Gallons.
	6. **PercentError_Vol**: Percent error, in volume, of the CIWS-Datalogger measurement. PercentError_Vol = (column 5 - column 4)*100 / column 4. Unit: %.
	7. **N_ExpectedValues**: Number of recorded values expected in each data collection period, calculated as (column 3 - column 2)/4 seconds Unit: count
	8. **N_ActualValues**: Number of actual values recorded by the CIWS-Datalogger in each data collection period. Unit: count. 
	9. **PercentError_Count**: Percent error, in count, of the CIWS-Datalogger measurement. PE_count = (column 8 - column 7)*100 / column 7. Unit: %.
	10. **OutdoorWaterUse_Expected**: An indication of whether outdoor water use is expected during the data collection period or not. Type: Binary (Yes / No).

The log files are named site**n**qc_log.csv, where **n** represents the siteID for each site where data was collected. 


* **3_QC_Data** contains all the data collected that passed the quality control procedure defined. There is 1 CSV file per site with the following columns:
	1. **Time**: Date and time stamp for each value collected. Format: 'YYYY-MM-DD HH:MM:SS'
	2. **Pulses**: The number of pulses recorded in every 4-second time period. Type: integer.
	
The QC_Data files  are named site**NNN**qc_data.csv, where **NNN** represents the siteID for each site where data was collected.

The volumetric meter resolution (volume per pulse) for each meter is included in the SitesInformation_HS.csv file (MeterResolution)

* **4_EventFilesOriginal** contains an events file for each site (in CSV format) with the following columns: 
	1. **StartTime**: Start date and time of each individual event. Format: 'YYYY-MM-DD HH:MM:SS'
	2. **EndTime**: End date and time of each individual event. Format: 'YYYY-MM-DD HH:MM:SS'
	3. **Duration**: Duration of each individual event (end time - start time). Units: Minutes
	4. **OriginalVolume**: Volume of water used in each individual event. Unit: Gallons.
	5. **OriginalFlowRate**: Average flow rate of each individual event. Unit: Gallons per minute.
	6. **Peak_Value**: Maximum volume value observed in each 4-seconds period within each event. Unit: Gallons
	7. **Mode_Value**: Most frequent volume value observed in an event. Unit: Gallons
	8. **Label**: Event classification. Values: faucet, toilet, shower, irrigation, clotheswasher, bathtub.
	9. **Site**: Identifier assigned to each participant in the study. Type: Integer. 

The event files are named LabelledEvents_site_**n**.csv, where **n** represents the siteID for each site where data was collected. 

The files in this folder were generated from processing the files in the 3_QC_Data folder using the tools contained in the HydroShare resource available at [https://www.hydroshare.org/resource/3143b3b1bdff48e0aaebcb4aedf02feb/](https://www.hydroshare.org/resource/3143b3b1bdff48e0aaebcb4aedf02feb/).


* **5_EventFiles_Processed** contains an events file for each site (in CSV format) with the following columns: 
	1. **StartTime**: Start date and time of each individual event. Format: 'YYYY-MM-DD HH:MM:SS'
	2. **EndTime**: End date and time of each individual event. Format: 'YYYY-MM-DD HH:MM:SS'
	3. **Duration**: Duration of each individual event (end time - start time). Units: Minutes
	4. **OriginalVolume**: Volume of water used in each individual event. Unit: Gallons.
	5. **OriginalFlowRate**: Average flow rate of each individual event. Unit: Gallons per minute.
	6. **Peak_Value**: Maximum volume value observed in each 4-seconds period within each event. Unit: Gallons
	7. **Mode_Value**: Most frequent volume value observed in an event. Unit: Gallons
	8. **Label**: Event classification. Values: faucet, toilet, shower, irrigation, clotheswasher, bathtub, unclassified, unknown.
	9. **Site**: Identifier assigned to each participant in the study. Type: Integer. 

The event files are named Events_site_**n**.csv, where **n** represents the siteID for each site where data was collected. 

These files were processed using the methodology described in the HydroShare resource available at [https://www.hydroshare.org/resource/379d9e7037f04478a99d5aec22e841e6/](https://www.hydroshare.org/resource/379d9e7037f04478a99d5aec22e841e6/).

