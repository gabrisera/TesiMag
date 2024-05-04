from library import *
from functions import *
def import_samples(h):
    
    dataset={"Cora":[],"Marianne": [],"Muriel": []}
    path="Dataset_and_Ground_Truth/data/"
    path2="Dataset_and_Ground_Truth/Ground_Truth/"
    file_list= os.listdir(path)
    file_list2=os.listdir(path2)
    for file_name in file_list:
        for file_name2 in file_list2:
            if file_name==file_name2:
                temp1=path+file_name
                temp2=path2+file_name
                if "Cora" in file_name:
                    dataset["Cora"].append(samples_creation(temp1,temp2,h))
                else:
                    if "Marianne" in file_name:
                        dataset["Marianne"].append(samples_creation(temp1,temp2,h))
                    else:
                        dataset["Muriel"].append(samples_creation(temp1,temp2,h))


    return dataset

