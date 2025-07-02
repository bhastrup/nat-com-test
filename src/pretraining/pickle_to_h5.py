import pickle, os
import h5py
import numpy as np


def save_data_to_hdf5(file_name, data):
    with h5py.File(file_name, 'w') as f:
        for key, value in data.items():
            if key != 'obs':
                f.create_dataset(key, data=value)
            else:
                elements = []
                positions = []
                bags = []

                for obs in value:
                    canvas, bag = obs
                    z, pos = zip(*canvas)
                    
                    elements.append(np.array(z))
                    positions.append(np.array(pos))
                    bags.append(np.array(bag))

                elements_array = np.stack(elements)
                positions_array = np.stack(positions)
                bags_array = np.stack(bags)

                obs_group = f.create_group('obs')
                obs_group.create_dataset("elements", data=elements_array)
                obs_group.create_dataset("positions", data=positions_array)
                obs_group.create_dataset("bags", data=bags_array)


def load_data_from_hdf5(file_name):
    loaded_data = {}
    with h5py.File(file_name, 'r') as f:
        for key in f.keys():
            if key != 'obs':
                if isinstance(f[key][0], bytes):
                    loaded_data[key] = [s.decode('utf-8') for s in f[key]]
                else:
                    loaded_data[key] = np.array(f[key])
            else:
                obs_group = f[key]
                elements_array = np.array(obs_group["elements"])
                positions_array = np.array(obs_group["positions"])
                bags_array = np.array(obs_group["bags"])

                obs_list = []
                for i in range(len(elements_array)):
                    canvas = list(zip(elements_array[i], [tuple(pos) for pos in positions_array[i]]))
                    bag = tuple(bags_array[i])
                    obs_elem = (tuple(canvas), bag)
                    obs_list.append(obs_elem)

                loaded_data['obs'] = obs_list

    return loaded_data


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':


    #target_directory = '/scratch/bjaha/expert_data/'
    target_directory = os.path.dirname(os.path.realpath(__file__))
    ensure_directory_exists(target_directory)
    target_name = os.path.join(target_directory, 'TMQM_0.h5')


    with open('TMQM_0.pkl', 'rb') as f:
        data = pickle.load(f)


    save_data_to_hdf5(file_name=target_name, data=data)
