import argparse
import torch
import torch.optim as optim

from tqdm import tqdm

from common.mbr import MBR

from utils.datasets_my import Dataset, collate_fn

from common.road_network import load_rn_shp

from models.model_utils import AttrDict,load_rn_dict,load_rid_freqs, get_rid_grid

from models.model_utils import init_weights

from models.model_my import Encoder, DecoderMulti, Seq2SeqMulti

from models.model_train import train

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
        road_num = 1366 + 1
        time_span = 15
        win_size = 48
        road_grid_size = 64 # use for road feature extraction
    elif args.dataset == 'Chengdu':
        city = 'Chengdu'
        min_lat, min_lng, max_lat, max_lng = (30.655, 104.07, 30.727, 104.129)
        user_num = 77499
        road_num = 2902 + 1
        time_span = 15
        win_size = 48
        road_grid_size = 64 
        
    else:
        print('Invalid dataset name')
        
    dataset_args = AttrDict()
    dataset_args_dict = {
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
        'road_grid_size': road_grid_size,
        'time_span': time_span,
        'win_size': win_size,
        'ds_type': 'uniform',
        'shuffle': True,
        
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        
    }
    dataset_args.update(dataset_args_dict)
    
    model_args = AttrDict()
    model_args_dict = {
        'device': device,
        'output_size': 1,
        'num_layers': 1,
        
        
        # constraints
        'dis_prob_mask_flag': True,
        'search_dist': 50,
        'beta': 15,
        'id_size' : road_num,
        
        # Encoder
        'input_size': 3,
        'hid_dim': args.hidden_size,
        'id_emb_dim': 128,
        'dropout': 0.5,
        
        # Attntion
        'attn_flag': True,
        
        # training
        'train_flag': True,
        'n_epochs': args.epochs,
        'learning_rate': args.lr,
        
    }
    model_args.update(model_args_dict)
    
    
    # prepare data
    # trajecotry data
    train_trajs_dir = './data/{}/train_data/'.format(city)
    valid_trajs_dir = './data/{}/valid_data/'.format(city)
    test_trajs_dir = './data/{}/test_data/'.format(city)
    # map data
    extra_info_dir = './data/map/{}/extra_data/'.format(city)
    rn_dir = './data/map/{}/'.format(city)
    rn = load_rn_shp(rn_dir, is_directed=True)
    raw_rn_dict = load_rn_dict(extra_info_dir, file_name='raw_rn_dict.json')
    new2raw_rid_dict = load_rid_freqs(extra_info_dir, file_name='new2raw_rid.json')
    raw2new_rid_dict = load_rid_freqs(extra_info_dir, file_name='raw2new_rid.json')
    rn_dict = load_rn_dict(extra_info_dir, file_name='rn_dict.json')
    
    mbr = MBR(dataset_args.min_lat, dataset_args.min_lng, dataset_args.max_lat, dataset_args.max_lng)
    dataset_args.mbr = mbr
    
    grid_rn_dict, max_xid, max_yid = get_rid_grid(mbr, args.grid_size, rn_dict)
    model_args.max_xid = max_xid
    model_args.max_yid = max_yid
    model_args.grid_rn_dict = grid_rn_dict
    model_args.raw_rn_dict = raw_rn_dict
    model_args.new2raw_rid_dict = new2raw_rid_dict
    model_args.raw2new_rid_dict = raw2new_rid_dict
    model_args.rn = rn
    model_args.rn_dict = rn_dict
    
    # load dataset
    train_dataset = Dataset(train_trajs_dir, parameters=dataset_args)
    valid_dataset= Dataset(valid_trajs_dir, parameters=dataset_args)
    test_dataset = Dataset(test_trajs_dir, parameters=dataset_args)
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('test dataset shape: ' + str(len(test_dataset)))
    
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_args.batch_size, 
                                                 shuffle=dataset_args.shuffle, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=dataset_args.batch_size, 
                                                 shuffle=dataset_args.shuffle, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_args.batch_size, 
                                                 shuffle=dataset_args.shuffle, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=False)
    
    # load model
    encoder = Encoder(model_args)
    decoder = DecoderMulti(model_args)
    model = Seq2SeqMulti(encoder, decoder, model_args)
    model.apply(init_weights)
    
    print('model', str(model))
    
    
    
    # training
    optimizer = optim.AdamW(model.parameters(), lr=model_args.learning_rate)
    if model_args.train_flag:
        for epoch in tqdm(range(model_args.n_epochs)):
            ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
            ls_train_rate_loss, ls_train_id_loss = [], [], [], [], [], []
            ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
            ls_valid_dis_mae_loss, ls_valid_dis_rmse_loss = [], [], [], [], [], []
            ls_valid_dis_rn_mae_loss, ls_valid_dis_rn_rmse_loss, ls_valid_rate_loss, ls_valid_id_loss = [], [], [], []
        
            new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision, \
                train_rate_loss, train_id_loss = train(model, train_iterator, optimizer, args)
                
            print('Epoch: {}, Train Loss: {:.4f}, Train ID Acc1: {:.4f}, Train ID Recall: {:.4f}, Train ID Precision: {:.4f}, Train Rate Loss: {:.4f}, Train ID Loss: {:.4f}'.format(epoch, train_loss, train_id_acc1, train_id_recall, train_id_precision, train_rate_loss, train_id_loss))

        
    