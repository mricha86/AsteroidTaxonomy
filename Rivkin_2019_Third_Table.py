# Purpose: Convert third table in Rivkin et al. 2019 into csv file 

# Import Libraries
import tabula


def main():
    # Retrieve all tables from Rivkin et al. 2019
    file = "/Users/mrichardson/Desktop/Work_with_Andy/Papers/Rivkin_et_al_2019.pdf"
    tables = tabula.read_pdf(file, pages="all", multiple_tables=True)

    # List of asteroid target names for corrections to third table 
    Asteroid = ["1 Ceres"]
    Asteroid.extend(["10 Hygiea"]*7)
    Asteroid.extend(["24 Themis*"])
    Asteroid.extend(["31 Euphrosyne"]*4)
    Asteroid.extend(["52 Europa"]*4)
    Asteroid.extend(["88 Thisbe"]*2)
    Asteroid.extend(["324 Bamberga"]*8)
    Asteroid.extend(["451 Patientia"]*2)
    Asteroid.extend(["704 Interamnia"]*5)
    
    # Convert and save third table as a csv file
    filename = "/Users/mrichardson/Desktop/Work_with_Andy/Data/Rivkin_2019_Extracted_Spectral_Features.csv"
    data = tables[2]
    n = len(data)
    for i in range(n):
        if data.loc[i,"Asteroid"] != Asteroid[i]:
            data.iloc[i,2:9] = data.iloc[i,1:8]
            data.iloc[i,1:2] = Asteroid[i]
    data.to_csv(filename, index=False)


if __name__ == "__main__":
   main()
