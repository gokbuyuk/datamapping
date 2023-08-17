sample_events = {
    'labsample': { # will be maped to one row in target; more precisely "labsample"; parse of one sample could be sent to different labs
        # information about firm where sample was collected:
        'firm':{
            "Firm Information Identifier":"EstablishmentID",
            "Legal Name": "EstablishmentName", #
            "Responsible Firm Type ": None,
            "State": "State",
            "Zip Code": None,
            "Address Line 1": None,
            "Address Line 2": None,
            "City": None,
            "Country Code": 'US',
        },
        'lab':{
            "Accomplishing Org Name":None,
        },
        'sample':{
            "Sample number": None,
            "Lab Sample number": "FormID",  #
            "Brand Name": None,
            "Product Code": None,
            "Expiry Date": None,
            "Lot Size": None,
            "Sample Description": "SampleSource",  #
            "Sample Collector ID on Package": None,
            "Sample Collector ID on Seal": None,
            "Sample Collection date": "CollectionDate",
            "Sample Collection Remarks": None,
            "SampleReceivedDate": None,
            "Sample Delivered Date": None,
        },
        'misc': {
            "Problem Area Flag": None,
            "Regulatory Program Name": None,
        }
    },
    "events": { # multiple events per sample such as lab tests; each evant mapped to seaparate row in target
        'Salmonella_Screen': {
            "Result": "SalmonellaSPAnalysis", # Salmonella Screen Result",
            "Method Genus":"PCR",  # placeholder, method was not specified, TODO
            "Method Remarks":"",
            "Result Genus": "Salmonella Screen Result",
            # "Lab Start Date": "NULL",
            # "Lab End Date": "NULL",
            "Species":"Salmonella enterica",
            "Species Code":"28901", # Taxon ID for Salmonella @VERIFY
            "Test Result Remarks":["SalmonellaAlleleCode", "SalmonellaSerotype",
                "SalmonellaPFGEPattern", "SalmonellaFSISNumber","SalmonellaAMRResistanceProfile"],
        },
        # 'Salmonella_Confirmatory': {
        #     "Result": "Salmonella Confirmatory Result",
        #     "Method Genus":"Salmonella Confirmatory Method",
        #     "Method Remarks":"Salmonella Confirmatory Method Remarks",
        #     "Result Genus": "Salmonella Confirmatory Result",
        #     "Lab Start Date": "Salmonella Analysis Start Date", # same as for screen? verify TODO
        #     "Lab End Date": "Salmonella Analysis Completion Date",
        #     "Species":"Salmonella enterica", # Taxon ID for Salmonella @VERIFY
        #     "Species Code":"28901", # Taxon ID for Salmonella @VERIFY
        #     "Test Result Remarks":["Salmonella Final Result","Salmonella Serotype"],
        # },
        "CampylobacterAnalysis1ml":{
            "Result": "CampylobacterAnalysis1ml",
            "Method Genus":"Campylobacter Analysis",
            "Method Remarks":"",
            "Result Genus": "Campylobacter Analysis Result",
            # "Lab Start Date": "Salmonella Analysis Start Date", # same as for screen? verify TODO
            # "Lab End Date": "Salmonella Analysis Completion Date",
            "Species":"CampylobacterSpecies", # Taxon ID for Salmonella @VERIFY
            "Species Code":"CampylobacterSpecies", # Taxon ID for Salmonella @VERIFY
            "Test Result Remarks":["CampylobacterPFGEPattern","CampylobacterAlleleCode", "CampyFSISNumber", "CampyAMRResistanceProfile"],            
        },
        "CampylobacterAnalysis30ml":{
            "Result": "CampylobacterAnalysis30ml",
            "Method Genus":"Campylobacter Analysis",
            "Method Remarks":"",
            "Result Genus": "Campylobacter Analysis Result",
            # "Lab Start Date": "NULL",
            # "Lab End Date": "NULL",
            "Species":"CampylobacterSpecies", # Taxon ID for Salmonella @VERIFY
            "Species Code":"CampylobacterSpecies", # Taxon ID for Salmonella @VERIFY
            "Test Result Remarks":["CampylobacterPFGEPattern","CampylobacterAlleleCode", "CampyFSISNumber", "CampyAMRResistanceProfile"],            
        },        

        # "Analyte Presence Indicator": "Salmonella Screen Result", ?? verify
        # "Result Genus": "Salmonella Screen Result",
        # "Firm Establishment Identifier": "NOT MAPPED",
        # "Method Modified Indicator": "NOT MAPPED"
    }
}