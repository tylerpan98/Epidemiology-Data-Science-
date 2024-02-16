import pandas as pd


data = pd.read_csv("merge.csv")

political_affiliation = pd.read_csv("Political Affiliation_cleaned.csv")

population = pd.read_csv("Population_Estimate_AllStates_cleaned.csv")

povertyRate = pd.read_csv("PovertyRate_cleaned.csv")

unemployment = pd.read_csv("UnemploymentRate_clean.csv")

HS_grad = pd.read_csv("HighSchoolDiploma_clean.csv")


merge_political = data.merge(political_affiliation,how="left")

merge_population = merge_political.merge(population,how="left")

merge_poverty = merge_population.merge(povertyRate,how="left")

merge_unemployment = merge_poverty.merge(unemployment,how="left")

merge_HS_grad = merge_unemployment.merge(HS_grad,how="left")

print(merge_HS_grad)

merge_HS_grad.to_csv('All_data_merged.csv')

