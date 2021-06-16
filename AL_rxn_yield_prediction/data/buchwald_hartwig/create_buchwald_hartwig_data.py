# Aryl Halides, 16 different variants
# 0: None

# Ligands, 4 different variations
# 0: XPhos
# 1: t-BuXPhos
# 2: t-BuBrettPhos
# 3: AdBrettPhosV

# Bases, 3 different variations
# 0: P2Et
# 1: BTMG
# 2: MTBD

# Additives, 24 different variations
# 0: None


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#####################################################
############### REACTION VARIABLES ##################
#####################################################

additivePlate1 = np.array(
    [
        0,
        2,
        4,
        6,
        0,
        2,
        4,
        6,
        0,
        2,
        4,
        6,
        0,
        2,
        4,
        6,
        1,
        3,
        5,
        7,
        1,
        3,
        5,
        7,
        1,
        3,
        5,
        7,
        1,
        3,
        5,
        7,
    ]
)

additivePlate2 = np.array(
    [
        8,
        10,
        12,
        14,
        8,
        10,
        12,
        14,
        8,
        10,
        12,
        14,
        8,
        10,
        12,
        14,
        9,
        11,
        13,
        15,
        9,
        11,
        13,
        15,
        9,
        11,
        13,
        15,
        9,
        11,
        13,
        15,
    ]
)

additivePlate3 = np.array(
    [
        23,
        17,
        19,
        21,
        23,
        17,
        19,
        21,
        23,
        17,
        19,
        21,
        23,
        17,
        19,
        21,
        16,
        18,
        20,
        22,
        16,
        18,
        20,
        22,
        16,
        18,
        20,
        22,
        16,
        18,
        20,
        22,
    ]
)

ligand = np.array(
    [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
    ]
)

base = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
)


arylHalide = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        0,
    ]
)


###############################################
############### CREATE DATA ###################
###############################################

df_plate = pd.read_csv("./data_table.csv")


yieldPlate1 = -1 * np.ones((32, 48))
yieldPlate2 = -1 * np.ones((32, 48))
yieldPlate3 = -1 * np.ones((32, 48))

for index, row in df_plate.iterrows():
    if row["plate"] == 1:
        yieldPlate1[row["row"] - 1, row["col"] - 1] = row["yield"]
    elif row["plate"] == 2:
        yieldPlate2[row["row"] - 1, row["col"] - 1] = row["yield"]
    elif row["plate"] == 3:
        yieldPlate3[row["row"] - 1, row["col"] - 1] = row["yield"]


column_names = ["Aryl Halide", "Ligand", "Base", "Additive", "Yield"]
data = pd.DataFrame(columns=column_names)

for iRow in range(yieldPlate1.shape[0]):
    iCol = 0
    for valueRow in yieldPlate1[iRow, :]:
        # Skip NaN values in the data
        if valueRow == -1:
            continue

        data = data.append(
            pd.DataFrame(
                [
                    [
                        arylHalide[iCol],
                        ligand[iRow],
                        base[iCol],
                        additivePlate1[iRow],
                        valueRow,
                    ]
                ],
                columns=column_names,
            ),
            ignore_index=True,
        )
        iCol += 1

for iRow in range(yieldPlate2.shape[0]):
    iCol = 0
    for valueRow in yieldPlate2[iRow, :]:
        # Skip NaN values in the data
        if valueRow == -1:
            continue

        data = data.append(
            pd.DataFrame(
                [
                    [
                        arylHalide[iCol],
                        ligand[iRow],
                        base[iCol],
                        additivePlate2[iRow],
                        valueRow,
                    ]
                ],
                columns=column_names,
            ),
            ignore_index=True,
        )
        iCol += 1

for iRow in range(yieldPlate3.shape[0]):
    iCol = 0
    for valueRow in yieldPlate3[iRow, :]:
        # Skip NaN values in the data
        if valueRow == -1:
            continue

        data = data.append(
            pd.DataFrame(
                [
                    [
                        arylHalide[iCol],
                        ligand[iRow],
                        base[iCol],
                        additivePlate3[iRow],
                        valueRow,
                    ]
                ],
                columns=column_names,
            ),
            ignore_index=True,
        )
        iCol += 1


data.to_csv("./buchwald_hartwig_data.csv", index=False)

enc = OneHotEncoder().fit(data.to_numpy()[:, :-1])



test_size = int(0.2 * len(data))
n_start_list = [10, 100, 1000]  #
save_suffix_test = "0_2"
save_suffix_train = "0_8"

states_start = [111, 222, 333, 444, 555]
states_test = [666, 777, 888, 999, 1111]

##############################################
########### VARYING START SETS ###############
##############################################
for n_start in n_start_list:
    test_state = states_test[0]
    for i_start, start in enumerate(states_start):

        train_and_start_buchwald, test_buchwald = train_test_split(
            data, test_size=test_size, random_state=test_state
        )

        print(f"Number of total training points: {len(train_and_start_buchwald)}")
        train_and_start_idx_buchwald = np.arange(len(train_and_start_buchwald))

        (
            train_buchwald,
            start_buchwald,
            train_idx_buchwald,
            start_idx_buchwald,
        ) = train_test_split(
            train_and_start_buchwald,
            train_and_start_idx_buchwald,
            test_size=n_start,
            random_state=start,
        )

        np.save(
            f"./start{n_start}_idx_buchwald_hartwig_start{i_start}",
            start_idx_buchwald,
        )

        start_buchwald.to_csv(
            f"./start{n_start}_buchwald_hartwig_start{i_start}.csv",
            index=False,
        )

        train_buchwald.to_csv(
            f"./train{n_start}_buchwald_hartwig_start{i_start}.csv",
            index=False,
        )

        test_buchwald.to_csv(
            f"./test{n_start}_buchwald_hartwig_start{i_start}.csv",
            index=False,
        )


        test_buchwald = test_buchwald.to_numpy()
        train_buchwald = train_buchwald.to_numpy()
        train_and_start_buchwald = train_and_start_buchwald.to_numpy()

        threshold = 20

        # Training set

        train_target = []
        train_yields = []

        for i_outer in range(train_and_start_buchwald.shape[0]):

            if train_and_start_buchwald[i_outer, -1] > threshold:
                train_target.append(1)
            else:
                train_target.append(0)

            train_yields.append(train_and_start_buchwald[i_outer, -1])


        np.save(
            f"./buchwald_hartwig_data_random_train{n_start}_{save_suffix_train}_one_hot_start{i_start}",
            enc.transform(train_and_start_buchwald[:, :-1]).toarray(),
        )

        np.save(
            f"./buchwald_hartwig_target_binary_{threshold}_random_train{n_start}_{save_suffix_train}_start{i_start}",
            np.array(train_target),
        )
        np.save(
            f"./buchwald_hartwig_yields_random_train{n_start}_{save_suffix_train}_start{i_start}",
            np.array(train_yields),
        )

        # Test set

        test_target = []
        test_yields = []

        for i_outer in range(test_buchwald.shape[0]):

            if test_buchwald[i_outer, -1] > threshold:
                test_target.append(1)
            else:
                test_target.append(0)

            test_yields.append(test_buchwald[i_outer, -1])

        np.save(
            f"./buchwald_hartwig_data_random_test{n_start}_{save_suffix_test}_one_hot_start{i_start}",
            enc.transform(test_buchwald[:, :-1]).toarray(),
        )

        np.save(
            f"./buchwald_hartwig_target_binary_{threshold}_random_test{n_start}_{save_suffix_test}_start{i_start}",
            np.array(test_target),
        )
        np.save(
            f"./buchwald_hartwig_yields_random_test{n_start}_{save_suffix_test}_start{i_start}",
            np.array(test_yields),
        )

        n_ones = np.sum(test_target) + np.sum(train_target)
        n_zeros = len(test_target) + len(train_target) - n_ones

        print(f"Number of ones: {n_ones}, number of zeros: {n_zeros}")

        test_state = states_test[0]
