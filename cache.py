import os

import numpy as np
from tqdm import tqdm

from model import Model


def weight_generator(N, digits):
    for n1 in np.arange(N):
        for n2 in np.arange(N - n1):
            for n3 in np.arange(N - n1 - n2):
                n4 = N - n1 - n2 - n3 - 1
                w1 = np.round(1 / (N - 1) * n1, digits)
                w2 = np.round(1 / (N - 1) * n2, digits)
                w3 = np.round(1 / (N - 1) * n3, digits)
                w4 = np.round(1 / (N - 1) * n4, digits)
                assert w1 + w2 + w3 + w4 - 1 < 1e-9
                yield [w1, w2, w3, w4]


def build_cache():
    print("GENERATE CACHE")
    print("==============")

    # settings
    lambda_values = [1, 1, 1, 1, 1, 1]
    alpha_pq = [4, 5]
    enable_box_constraint_lambda_rescaling = True

    # spawn instances to generate cache
    model = Model()
    model.data_processor.verbose = False
    pbar = tqdm(list(weight_generator(11, 1)), desc="build cache")
    for weights in pbar:
        pbar.set_postfix_str(f'weights={weights}')
        test_file_1 = os.path.join(model.data_processor.cache_path,
                                   f'model_cache-{";".join([f"{weight:.9f}" for weight in weights])}.json')
        test_file_2 = os.path.join(model.data_processor.cache_path,
                                   f'costmatrix-{";".join([f"{weight:.9f}" for weight in weights[:3]])}.csv')
        test_file_3 = os.path.join(model.data_processor.cache_path,
                                   f'solutionspace-{";".join([f"{weight:.9f}" for weight in weights[:3]])}.csv')
        test_file_4 = os.path.join(model.data_processor.cache_path,
                                   f'paths-{";".join([f"{weight:.9f}" for weight in weights[:3]])}.csv')
        if os.path.exists(test_file_1) and os.path.exists(test_file_2) and os.path.exists(
                test_file_3) and os.path.exists(test_file_4):
            continue
        model.spawn_instance(weights, alpha_pq, lambda_values, enable_box_constraint_lambda_rescaling,
                             variable_assignments=None, verbose=False)


if __name__ == "__main__":
    build_cache()
