
import argparse
import numpy as np
from src.data.reference_dataloader import ReferenceDataLoader
from src.data.io_handler import IOHandler


def parse_cmd():
    parser = argparse.ArgumentParser(description="Script for splitting bags into test and train.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--mol_dataset",
        type=str,
        default="qm7",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=20,
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_cmd()

    # Load the reference data
    loader = ReferenceDataLoader(data_dir=args.data_dir)
    ref_data = loader._load_data(mol_dataset=args.mol_dataset)
    energies = loader.get_benchmark_energies(ref_data)

    bags = list(energies.keys())
    num_bags = len(bags)
    print(f"Number of bags: {num_bags}")
    print(f"Bags: {bags}")

    # Split into train and test
    np.random.seed(42)
    test_indices = np.random.choice(num_bags, args.n_test, replace=False)
    train_indices = np.array([i for i in range(num_bags) if i not in test_indices])

    train_bags = [bags[i] for i in train_indices]
    test_bags = [bags[i] for i in test_indices]

    print(f"Train bags: {train_bags}")
    print(f"Test bags: {test_bags}")

    # Save the split
    split = dict(train=train_bags, test=test_bags)
    split_path = f'{args.data_dir}/{args.mol_dataset}/processed/split.json'

    IOHandler.write_json(split, split_path)


# {
#     "train": [
#         "H6C2",
#         "H2C2",
#         "H4C",
#         "H4C2",
#         "H4C2O",
#         "H6C3",
#         "H6C2O",
#         "H5C2N",
#         "H4C3",
#         "H7C2N",
#         "H3C2N",
#         "H7C3N",
#         "H6C4",
#         "H8C4",
#         "HC3N",
#         "H5C3N",
#         "H8C3O",
#         "H4C3O",
#         "H5C2NO",
#         "H2C3O",
#         "H7C2NO",
#         "H4C4",
#         "H2C4",
#         "H10C4O",
#         "H8C4O",
#         "H4C4O",
#         "H8C5",
#         "H5C3NO",
#         "H3C3NO",
#         "H7C3NO",
#         "H6C5",
#         "H9C3NO",
#         "H4C4S",
#         "H4C5",
#         "H3C3NS",
#         "H5C4N",
#         "H9C4N",
#         "H7C4N",
#         "H3C4N",
#         "H6C3N2",
#         "H12C5",
#         "H4C3N2",
#         "H12C6",
#         "H10C6",
#         "H6C4N2",
#         "H7C5N",
#         "H9C5N",
#         "H5C5N",
#         "H11C5N",
#         "H4C4N2",
#         "H10C4N2",
#         "H8C4N2",
#         "H14C6",
#         "H3C5N",
#         "H13C5N",
#         "HC5N",
#         "H8C6",
#         "H6C6",
#         "H2C4N2",
#         "C4N2",
#         "H6C5O",
#         "H8C5O",
#         "H10C5O",
#         "H8C4O2",
#         "H6C4O2",
#         "H4C6",
#         "H12C5O",
#         "H4C5O",
#         "H4C4O2",
#         "H2C6",
#         "H10C4O2",
#         "H2C5O",
#         "H2C4O2",
#         "H7C4NO",
#         "H5C4NO",
#         "H11C4NO",
#         "H3C4NO",
#         "HC4NO",
#         "H5C4NS",
#         "H6C5S",
#         "H6C3O2S",
#         "H8C3O2S",
#         "H4C3O2S",
#         "H4C4OS",
#         "H13C6N",
#         "H15C6N",
#         "H9C6N",
#         "H11C6N",
#         "H12C7",
#         "H7C6N",
#         "H10C7",
#         "H10C5N2",
#         "H12C5N2",
#         "H8C7",
#         "H8C5N2",
#         "H5C6N",
#         "H4C5N2",
#         "H2C5N2",
#         "H6C5N2",
#         "H14C7",
#         "H5C4N3",
#         "H8C6O",
#         "H6C5O2",
#         "H10C6O",
#         "H12C6O",
#         "H10C5O2",
#         "H4C5O2",
#         "H4C7",
#         "H4C6O",
#         "H14C6O",
#         "H12C5O2",
#         "H2C5O2",
#         "H7C5NO",
#         "H6C4N2O",
#         "H5C4NO2",
#         "H9C5NO",
#         "H4C4N2O",
#         "H5C5NO",
#         "H7C4NO2",
#         "H11C5NO",
#         "H8C4N2O",
#         "H3C4NO2",
#         "H10C4N2O",
#         "H2C4N2O",
#         "H13C5NO",
#         "H12C4N2O",
#         "H11C4NO2",
#         "H16C7",
#         "H4C4N2S",
#         "H5C5NS",
#         "H6C6S",
#         "H4C5OS",
#         "H3C4NOS",
#         "H8C6S",
#         "H7C5NS",
#         "H6C5OS",
#         "H5C4NOS",
#         "H6C4N2S",
#         "H8C4O2S",
#         "H8C4N2S",
#         "H7C3NO2S",
#         "H6C3O3S",
#         "H3C5NS",
#         "H2C4N2S",
#         "H10C4N2S",
#         "H10C4O2S",
#         "H6C4O2S",
#         "H9C3NO2S",
#         "H5C3NO2S",
#         "H8C3O3S",
#         "H4C3O3S",
#         "H4C4O2S",
#         "H3C3NO2S",
#         "H2C4O2S",
#         "HC3NO2S",
#         "H3C6N"
#     ],
#     "test": [
#         "H6C3O",
#         "H11C4N",
#         "H9C4NO2",
#         "H6C4O",
#         "H12C4N2",
#         "H3C3N",
#         "H6C6O",
#         "H2C3O3S",
#         "H14C5N2",
#         "H3C5NO",
#         "H10C4",
#         "H4C6S",
#         "H10C5",
#         "H8C5O2",
#         "H9C3N",
#         "H7C4N3",
#         "H6C7",
#         "H9C4NO",
#         "H8C3",
#         "H3C4N3"
#     ]
# }