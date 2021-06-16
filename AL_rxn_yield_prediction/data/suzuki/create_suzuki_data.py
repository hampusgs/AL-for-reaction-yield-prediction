import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


data = pd.read_excel("./aap9112_Data_File_S1.xlsx")


reactant1 = data["Reactant_1_Name"].copy()

reactant2 = data["Reactant_2_Name"].copy()

ligand = data["Ligand_Short_Hand"].copy()

reagent = data["Reagent_1_Short_Hand"].copy()

solvent = data["Solvent_1_Short_Hand"].copy()

productYieldAreaUV = data["Product_Yield_PCT_Area_UV"].copy()

productYieldMassIonCount = data["Product_Yield_Mass_Ion_Count"].copy()


uniqueReactant1 = reactant1.unique()
print(f"Reactant 1s : {uniqueReactant1}")


reactant1.loc[reactant1 == "6-chloroquinoline"] = 0
reactant1.loc[reactant1 == "6-Bromoquinoline"] = 1
reactant1.loc[reactant1 == "6-triflatequinoline"] = 2
reactant1.loc[reactant1 == "6-Iodoquinoline"] = 3
reactant1.loc[reactant1 == "6-quinoline-boronic acid hydrochloride"] = 4
reactant1.loc[reactant1 == "6-Quinolineboronic acid pinacol ester"] = 5
reactant1.loc[reactant1 == "Potassium quinoline-6-trifluoroborate"] = 6

print(reactant1)
uniqueReactant2 = reactant2.unique()

print(f"Reactant 2s : {uniqueReactant2}")

reactant2.loc[reactant2 == "2a, Boronic Acid"] = 0
reactant2.loc[reactant2 == "2b, Boronic Ester"] = 1
reactant2.loc[reactant2 == "2c, Trifluoroborate"] = 2
reactant2.loc[reactant2 == "2d, Bromide"] = 3

uniqueLigand = ligand.unique()

print(f"Ligands: {uniqueLigand}")

i = 0
for iLigand in uniqueLigand:
    ligand.loc[ligand == iLigand] = i
    i += 1

uniqueReagent = reagent.unique()

print(f"Bases: {uniqueReagent}")

i = 0

for iReagent in uniqueReagent:
    reagent.loc[reagent == iReagent] = i
    i += 1

solvent.loc[solvent == "MeOH/H2O_V2 9:1"] = "MeOH"
solvent.loc[solvent == "THF_V2"] = "THF"

uniqueSolvent = solvent.unique()

print(f"Solvent: {uniqueSolvent}")

i = 0

for iSolvent in uniqueSolvent:
    solvent.loc[solvent == iSolvent] = i
    i += 1


suzuki_data = pd.concat(
    [reactant1, reactant2, ligand, reagent, solvent, productYieldAreaUV], axis=1
)


suzuki_data = suzuki_data.drop(suzuki_data.index[4608:]).reset_index(drop=True)

suzuki_data.to_csv("./suzuki_data.csv", index=False)


enc = OneHotEncoder().fit(suzuki_data.to_numpy()[:, :-1])


test_size = int(len(suzuki_data) * 0.2)
n_start_list = [10, 100, 1000]
save_suffix_test = "0_2"
save_suffix_train = "0_8"
threshold = 20


states_start = [111, 222, 333, 444, 555]
states_test = [666, 777, 888, 999, 1111]

for n_start in n_start_list:

    ##############################################
    ########### VARYING START SETS ###############
    ##############################################

    test_state = states_test[0]
    for i_start, start in enumerate(states_start):

        train_and_start_suzuki, test_suzuki = train_test_split(
            suzuki_data, test_size=test_size, random_state=test_state
        )

        train_and_start_idx_suzuki = np.arange(len(train_and_start_suzuki))

        (
            train_suzuki,
            start_suzuki,
            train_idx_suzuki,
            start_idx_suzuki,
        ) = train_test_split(
            train_and_start_suzuki,
            train_and_start_idx_suzuki,
            test_size=n_start,
            random_state=start,
        )

        np.save(
            f"./start{n_start}_idx_suzuki_start{i_start}",
            start_idx_suzuki,
        )

        start_suzuki.to_csv(
            f"./start{n_start}_suzuki_start{i_start}.csv",
            index=False,
        )
        train_suzuki.to_csv(
            f"./train{n_start}_suzuki_start{i_start}.csv",
            index=False,
        )
        test_suzuki.to_csv(
            f"./test{n_start}_suzuki_start{i_start}.csv", index=False
        )


        train_suzuki = train_suzuki.to_numpy()
        test_suzuki = test_suzuki.to_numpy()
        train_and_start_suzuki = train_and_start_suzuki.to_numpy()

        # Training set

        train_target = []
        train_yields = []

        for i_outer in range(train_and_start_suzuki.shape[0]):

            if train_and_start_suzuki[i_outer, -1] > threshold:
                train_target.append(1)
            else:
                train_target.append(0)

            train_yields.append(train_and_start_suzuki[i_outer, -1])


        np.save(
            f"./suzuki_data_random_train{n_start}_{save_suffix_train}_one_hot_start{i_start}",
            enc.transform(train_and_start_suzuki[:, :-1]).toarray(),
        )
        np.save(
            f"./suzuki_target_binary_{threshold}_random_train{n_start}_{save_suffix_train}_start{i_start}",
            np.array(train_target),
        )
        np.save(
            f"./suzuki_yields_random_train{n_start}_{save_suffix_train}_start{i_start}",
            np.array(train_yields),
        )

        # Test set

        test_target = []
        test_yields = []

        for i_outer in range(test_suzuki.shape[0]):

            if test_suzuki[i_outer, -1] > threshold:
                test_target.append(1)
            else:
                test_target.append(0)

            test_yields.append(test_suzuki[i_outer, -1])


        np.save(
            f"./suzuki_data_random_test{n_start}_{save_suffix_test}_one_hot_start{i_start}",
            enc.transform(test_suzuki[:, :-1]).toarray(),
        )
        np.save(
            f"./suzuki_target_binary_{threshold}_random_test{n_start}_{save_suffix_test}_start{i_start}",
            np.array(test_target),
        )
        np.save(
            f"./suzuki_yields_random_test{n_start}_{save_suffix_test}_start{i_start}",
            np.array(test_yields),
        )

        n_ones = np.sum(test_target) + np.sum(train_target)
        n_zeros = len(test_target) + len(train_target) - n_ones

        print(f"Number of ones: {n_ones}, number of zeros: {n_zeros}")
