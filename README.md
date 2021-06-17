# Using Active Learning to Develop Machine Learning Models for Reaction Yield Prediction

## Description
Implementations used in *Using Active Learning to Develop Machine Learning Models for Reaction Yield Prediction*. It provides implementations for using active learning with neural networks, matrix factorization and random forest for reaction yield prediction.

## Data

### Suzuki reaction data
The Suzuki reaction data is available as supplemntary material in [A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow](https://doi.org/10.1126/science.aap9112).


### Buchwald-Hartwig reaction data
The Buchwald-Hartwig reaction data is published in [Predicting reaction performance in C–N cross-coupling using machine learning](https://doi.org/10.1126/science.aar5169) and the data is made available [here](https://github.com/doylelab/rxnpredict) by the Doyle Laboratory.


## Preprocess data
Create and activate enviornment
```
conda env create -f env-data.yml
conda activate al-data-env
```
Use this enviornment to preprocess the data as follows.


### Suzuki reaction data
With the the data file `aap9112_Data_File_S1.xlsx` in the same folder as `create_suzuki_data.py`, run
```
python create_suzuki_data.py
```

### Buchwald-Hartwig reaction data
With the the data file `data_table.csv` in the same folder as `create_buchwald_hartwig_data.py`, run
```
python create_buchwald_hartwig_data.py
```
## Usage

Create and activate enviornment
```
conda env create -f al-env.yml
conda activate al-env
```

### Running matrix factorization / random forest



Using the environment `al-env`, run `rf_mf.py` with appropriate arguments. For instance

```
python rf_mf.py \
--data  ../data/buchwald_hartwig/buchwald_hartwig_data.csv \
--test_file ../data/buchwald_hartwig/test10_buchwald_hartwig_start0.csv \
--starting_file ../data/buchwald_hartwig/start10_buchwald_hartwig_start0.csv
```


### Running neural network


Using the environment `al-env`, run `nn.py` with appropriate arguments. For instance
```
python nn.py --layer_size 10 2 \
--results_dir log_dir \
--input_path_train ../data/suzuki/suzuki_data_random_train10_0_8_one_hot_start0.npy \
--target_path_train ../data/suzuki/suzuki_target_binary_20_random_train10_0_8_start0.npy \
--input_path_test ../data/suzuki/suzuki_data_random_test10_0_2_one_hot_start0.npy \
--target_path_test ../data/suzuki/suzuki_target_binary_20_random_test10_0_2_start0.npy \
--start_idx_path ../data/suzuki/start10_idx_suzuki_start0.npy \
--yields_path ../data/suzuki/suzuki_yields_random_train10_0_8_start0.npy \
--query_size 1 --query_strategy random --init_states_dir init_states/suzuki_simple
```

## Contributors
* [@hampusgs](https://www.github.com/hampusgs)
* [@SeemonJ](https://github.com/SeemonJ)

## License
The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.

## References

