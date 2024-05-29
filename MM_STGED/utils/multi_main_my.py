import argparse
import torch

from utils.datasets_my import Dataset, collate_fn

from models.model_utils import AttrDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MM_STGED')
    parser.add_argument('--dataset', type=str, default='Porto', help='city name')
    
    # dataset parameters
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio for training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size')
    
    # model parameters
    parser.add_argument('--module_type', type=str, default='simple', help='module type')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # choose dataset and set parameters
    if args.dataset == 'Porto':
        city = 'Porto'
        min_lat, min_lng, max_lat, max_lng = (41.142, -8.652, 41.174, -8.578)
        user_num = 439
        road_num = 1136 + 1
        time_span = 15
        win_size = 25
    elif args.dataset == 'Chengdu':
        city = 'Chengdu'
        min_lat, min_lng, max_lat, max_lng = (30.655, 104.07, 30.727, 104.129)
        user_num = 77499
        road_num = 2902 + 1
        time_span = 15
        win_size = 25
        
    else:
        print('Invalid dataset name')
        
    dataset_args = AttrDict()
    dataset_args = {
        'device': device,
        'city': city,
        'user_num': user_num,
        'road_num': road_num,
        
        'min_lat': min_lat,
        'min_lng': min_lng,
        'max_lat': max_lat,
        'max_lng': max_lng,
        
        'keep_ratio': args.keep_ratio,
        'grid_size': args.grid_size,
        'time_span': time_span,
        'win_size': win_size,
        'ds_type': 'uniform',
        'shuffle': True,
        
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        
    }
    model_args = AttrDict()
    
    # prepare data
    # trajecotry data
    train_trajs_dir = './data/{}/train_data/'.format(city)
    valid_trajs_dir = './data/{}/valid_data/'.format(city)
    test_trajs_dir = './data/{}/test_data/'.format(city)
    # map data
    extra_info_dir = './data/map/{}/extra_data/'.format(city)
    raw2new_rid_dict = None
    
    # load dataset
    train_dataset = Dataset(train_trajs_dir, raw2new_rid_dict, parameters=dataset_args)
    
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_args.batch_size, 
                                                 shuffle=args.shuffle, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)
    
    
    