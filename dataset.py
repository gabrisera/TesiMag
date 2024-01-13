from library import *
def import_data():
    df={}
    df1={}
    #04-12-2017
    df["04-12-2017_t028"]=pd.read_csv("sample_data/data/GA_single_04-12-2017_t028_filtered.txt",sep="\t")
    df1["04-12-2017_t028"]=pd.read_csv("sample_data/ground_truth/GA_single_04-12-2017_t028_space_ground_truth.txt",sep="\t")
    #08-08-2019
    df["08-08-2019_t001"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t001_filtered.txt",sep="\t")
    df1["08-08-2019_t001"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t001_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t004"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t004_filtered.txt",sep="\t")
    df1["08-08-2019_t004"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t004_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t007"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t007_filtered.txt",sep="\t")
    df1["08-08-2019_t007"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t007_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t008"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t008_filtered.txt",sep="\t")
    df1["08-08-2019_t008"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t008_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t014"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t014_filtered.txt",sep="\t")
    df1["08-08-2019_t014"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t014_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t015"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t015_filtered.txt",sep="\t")
    df1["08-08-2019_t015"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t015_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t017"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t017_filtered.txt",sep="\t")
    df1["08-08-2019_t017"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t017_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t019"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t018_filtered.txt",sep="\t")
    df1["08-08-2019_t019"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t018_space_ground_truth.txt",sep="\t")
    df["08-08-2019_t020"]=pd.read_csv("sample_data/data/GA_single_08-08-2019_t020_filtered.txt",sep="\t")
    df1["08-08-2019_t020"]=pd.read_csv("sample_data/ground_truth/GA_single_08-08-2019_t020_space_ground_truth.txt",sep="\t")
    #21-03-2016
    df["21-03-2016_t014"]=pd.read_csv("sample_data/data/GA_single_21-03-2016_t014_filtered.txt",sep="\t")
    df1["21-03-2016_t014"]=pd.read_csv("sample_data/ground_truth/GA_single_21-03-2016_t014_space_ground_truth.txt",sep="\t")
    df["21-03-2016_t018"]=pd.read_csv("sample_data/data/GA_single_21-03-2016_t018_filtered.txt",sep="\t")
    df1["21-03-2016_t018"]=pd.read_csv("sample_data/ground_truth/GA_single_21-03-2016_t018_space_ground_truth.txt",sep="\t")
    df["21-03-2016_t026"]=pd.read_csv("sample_data/data/GA_single_21-03-2016_t026_filtered.txt",sep="\t")
    df1["21-03-2016_t026"]=pd.read_csv("sample_data/ground_truth/GA_single_21-03-2016_t026_space_ground_truth.txt",sep="\t")
    df["21-03-2016_t027"]=pd.read_csv("sample_data/data/GA_single_21-03-2016_t027_filtered.txt",sep="\t")
    df1["21-03-2016_t027"]=pd.read_csv("sample_data/ground_truth/GA_single_21-03-2016_t027_space_ground_truth.txt",sep="\t")
    #22-03-2016
    df["22-03-2016_t007"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t007_filtered.txt",sep="\t")
    df1["22-03-2016_t007"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t007_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t008"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t008_filtered.txt",sep="\t")
    df1["22-03-2016_t008"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t008_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t009"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t009_filtered.txt",sep="\t")
    df1["22-03-2016_t009"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t009_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t010"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t010_filtered.txt",sep="\t")
    df1["22-03-2016_t010"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t010_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t012"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t012_filtered.txt",sep="\t")
    df1["22-03-2016_t012"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t012_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t013"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t013_filtered.txt",sep="\t")
    df1["22-03-2016_t013"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t013_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t017"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t017_filtered.txt",sep="\t")
    df1["22-03-2016_t017"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t017_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t018"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t018_filtered.txt",sep="\t")
    df1["22-03-2016_t018"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t018_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t019"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t019_filtered.txt",sep="\t")
    df1["22-03-2016_t019"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t019_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t024"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t024_filtered.txt",sep="\t")
    df1["22-03-2016_t024"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t024_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t026"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t026_filtered.txt",sep="\t")
    df1["22-03-2016_t026"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t026_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t041"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t041_filtered.txt",sep="\t")
    df1["22-03-2016_t041"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t041_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t042"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t042_filtered.txt",sep="\t")
    df1["22-03-2016_t042"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t042_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t043"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t043_filtered.txt",sep="\t")
    df1["22-03-2016_t043"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t043_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t047"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t047_filtered.txt",sep="\t")
    df1["22-03-2016_t047"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t047_space_ground_truth.txt",sep="\t")
    df["22-03-2016_t048"]=pd.read_csv("sample_data/data/GA_single_22-03-2016_t048_filtered.txt",sep="\t")
    df1["22-03-2016_t048"]=pd.read_csv("sample_data/ground_truth/GA_single_22-03-2016_t048_space_ground_truth.txt",sep="\t")
    #23-03-2016
    df["23-03-2016_t028"]=pd.read_csv("sample_data/data/GA_single_23-03-2016_t028_filtered.txt",sep="\t")
    df1["23-03-2016_t028"]=pd.read_csv("sample_data/ground_truth/GA_single_23-03-2016_t028_space_ground_truth.txt",sep="\t")
    df["23-03-2016_t029"]=pd.read_csv("sample_data/data/GA_single_23-03-2016_t029_filtered.txt",sep="\t")
    df1["23-03-2016_t029"]=pd.read_csv("sample_data/ground_truth/GA_single_23-03-2016_t029_space_ground_truth.txt",sep="\t")
    df["23-03-2016_t030"]=pd.read_csv("sample_data/data/GA_single_23-03-2016_t030_filtered.txt",sep="\t")
    df1["23-03-2016_t030"]=pd.read_csv("sample_data/ground_truth/GA_single_23-03-2016_t030_space_ground_truth.txt",sep="\t")

    return df, df1
def import_data1():
    df={}
    df["prova_muriel_t0030"]=pd.read_csv("sample_data/data/provat0030.txt",sep=" ")
    return df
def import_data2():
    df={}
    df["prova_cora_t0005"]=pd.read_csv("sample_data/data/provat0005.txt",sep=" ")
    return df