import torch
import random
import numpy as np
from torchvision import datasets, transforms

dataset_path = '/data/zgma/dataset'

def flatten_paras(paras):
    flatten_paras = []
    for p in paras:
        flatten_paras.append(torch.reshape(p.data.clone().detach(), [-1]))
    return torch.cat(flatten_paras)

def ada_pruning(numpy_value, numpy_idx, est_beta, max_ratio):
    temp_beta = est_beta.item()
    mask = np.zeros(len(numpy_value))
    sum_ele = 1e-7
    for i in range(len(numpy_value)):
        ele = numpy_value[i]
        if ele == 0.0:
            break
        if i > len(numpy_value) * max_ratio:
            break
        temp_sum = sum_ele + np.square(ele)
        if temp_sum > np.square(temp_beta):
            break
        else:
            mask[numpy_idx[i]] = 1
            sum_ele = temp_sum
    return mask

def find_index(targets, labels):
    return [i for (i,v) in enumerate(targets) if v in labels]

def find_non_index(targets, labels, N):
    count = 0
    indices = [i for i in range(len(targets))]
    random.shuffle(indices)
    if N == 0:
        return []
        
    return_indices = []
    for i in range(len(indices)):
        if targets[indices[i]] not in labels:
            return_indices.append(indices[i])
            count += 1
        if count >= N:
            return return_indices

def load_imagenet(ratio, select_labels, num_devices):
    random.seed(0)
    transform = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
    train_dataset = datasets.ImageFolder(dataset_path+'/IMAGE100/train', transform=transform)
    test_dataset = datasets.ImageFolder(dataset_path+'/IMAGE100/test', transform=transform)

    train_labels = train_dataset.targets
    test_labels = test_dataset.targets
    main_train_indices = find_index(train_labels, select_labels)
    main_test_indices = find_index(test_labels, select_labels)
    random.shuffle(main_train_indices)
    random.shuffle(main_test_indices)

    main_train_samples = len(main_train_indices)
    main_test_samples = len(main_test_indices)

    main_device_train_samples = int(main_train_samples / num_devices)
    main_device_test_samples = int(main_test_samples / num_devices)
    extra_device_train_samples = int((main_device_train_samples / ratio) * (1-ratio))
    extra_device_test_samples = int((main_device_test_samples / ratio) * (1-ratio))

    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(num_devices):
        start_train_idx = main_device_train_samples * i
        start_test_idx = main_device_test_samples * i

        device_train_indices = main_train_indices[start_train_idx:(start_train_idx+main_device_train_samples)]
        extra_device_train_indices = find_non_index(train_labels, select_labels, extra_device_train_samples)
        device_train_indices.extend(extra_device_train_indices)

        device_test_indices = main_test_indices[start_test_idx:(start_test_idx+main_device_test_samples)]
        extra_device_test_indices = find_non_index(test_labels, select_labels, extra_device_test_samples)
        device_test_indices.extend(extra_device_test_indices)

        i_train_data, i_train_label = zip(*([train_dataset[j] for j in device_train_indices]))
        i_test_data, i_test_label = zip(*([test_dataset[j] for j in device_test_indices]))

        train_data.append(torch.stack(i_train_data))
        train_label.append(torch.tensor(i_train_label))
        test_data.append(torch.stack(i_test_data))
        test_label.append(torch.tensor(i_test_label))
    return train_data, train_label, test_data, test_label
