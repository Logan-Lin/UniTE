import json
import os
from argparse import ArgumentParser
import copy

import torch
from torch.cuda import is_available as cuda_available

from data import Data
from pretrain import trainer as PreTrainer, generative_losses, contrastive_losses
import modules
from downstream import predictor as DownPredictor, task
import utils


def main():
    def parse_args():
        """
        Parse command line arguments and set up CUDA device.
        
        Returns:
            tuple: (parsed_args, device_string)
            - parsed_args contains config file name and cuda device index
            - device_string is 'cuda:0' or 'cpu' depending on availability
        """
        parser = ArgumentParser()
        parser.add_argument('-c', '--config', help='name of the config file to use', type=str, required=True)
        parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default=0)
        args = parser.parse_args()
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        device = f'cuda:0' if cuda_available() else 'cpu'
        return args, device

    def load_data(data_entry):
        """
        Load and initialize dataset based on config entry.
        
        Args:
            data_entry (dict): Dataset configuration containing name and road_type
            
        Returns:
            Data: Initialized data object with loaded statistics
        """
        data = Data(data_entry['name'], data_entry.get('road_type', 'road_network'))
        data.load_stat()
        return data

    def create_model(model_entry, data, num_roads, num_classes, pretrain):
        """
        Create model instance based on configuration.
        
        Args:
            model_entry (dict): Model configuration containing name and parameters
            data (Data): Dataset object for loading metadata
            num_roads (int): Number of unique road segments
            num_classes (int): Number of classes for classification tasks
            pretrain (bool): Whether the model is pretrained
            
        Returns:
            nn.Module: Initialized model instance
            
        Raises:
            NotImplementedError: If model name is not recognized
        """
        # Prepare sampler
        sampler = create_augmentation(model_entry.get('augmentation', {'name': 'pass'}))
        
        # Prepare model config
        model_config = model_entry.get('config', {})
        if "pre_embed" in model_config:
            model_config["pre_embed"] = data.load_meta(model_config.get("pre_embed"), 0)[0]
            model_config["pre_embed_update"] = model_config.get("pre_embed_update", True)

        # Create model based on name
        model_name = model_entry['name']
        if model_name == 'ia':
            return modules.model.induced_att.InducedAttEncoder(sampler=sampler, **model_config)
        elif model_name == 'transformer_encoder':
            return modules.model.transformer.TransformerEncoder(sampler=sampler, **model_config)
        elif model_name == 'transformer_decoder':
            return modules.model.transformer.TransformerDecoder(**model_config)
        elif model_name == 'transformer_denoiser':
            return modules.model.transformer.TransformerDenoiser(**model_config)
        elif model_name == 'dualpos_transformer':
            return modules.model.transformer.DualPosTransformer(sampler=sampler, **model_config)
        elif model_name == 'mlm_transformer':
            return modules.model.transformer.MLMTransformer(sampler=sampler, **model_config)
        elif model_name == 'cde':
            return modules.model.ode.CDEEncoder(sampler=sampler, **model_config)
        elif model_name == 'coa':
            return modules.model.ode.CoeffAttEncoder(sampler=sampler, **model_config)
        elif model_name == 'stode':
            return modules.model.ode.STODEEncoder(sampler=sampler, **model_config)
        elif model_name == 'trajode_decoder':
            return modules.model.ode.TrajODEDecoder(**model_config)
        elif model_name == 'rnn_encoder':
            return modules.model.rnn.RnnEncoder(sampler=sampler, num_embed=num_roads, **model_config)
        elif model_name == 'rnn_decoder':
            return modules.model.rnn.RnnDecoder(num_roads=num_roads, **model_config)
        elif model_name == 'gmvsae_encoder':
            return modules.model.gmvsae.GMVSAEEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'gmvsae_decoder':
            return modules.model.gmvsae.GMVSAEDecoder(num_embed=num_roads, **model_config)
        elif model_name == 'bert':
            return modules.model.start.BERTEncoder(sampler=sampler, vocab_size=num_roads, **model_config)
        elif model_name == 'trajectory2vec_encoder':
            return modules.model.trajectory2vec.Trajectory2VecEncoder(sampler=sampler, **model_config)
        elif model_name == 'trajectory2vec_decoder':
            return modules.model.trajectory2vec.Trajectory2vecDecoder(sampler=sampler, **model_config)
        elif model_name == 'trajsim_embedding':
            model = modules.model.trajectorysim.TrajSimEmbed(meta_dir=data.meta_dir, **model_config, pretrain=pretrain)
            global vocab_size, dist_path
            vocab_size = model.vocab_size
            dist_path = model.dist_path
            return model
        elif model_name == 'trajsim_encoder':
            return modules.model.trajectorysim.TrajSimEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'trajsim_decoder':
            model = modules.model.trajectorysim.TrajSimDecoder(**model_config)
            global hidden_size
            hidden_size = model.hidden_size
            return model
        elif model_name == 't2vecEmbedding':
            model = modules.model.t2vec.t2vecEmbedding(meta_dir=data.meta_dir, **model_config, pretrain=pretrain)
            global vocab_size, dist_path
            vocab_size = model.vocab_size
            dist_path = model.dist_path
            return model
        elif model_name == 't2vecEncoder':
            return modules.model.t2vec.t2vecEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 't2vecDecoder':
            model = modules.model.t2vec.t2vecDecoder(**model_config)
            global hidden_size
            hidden_size = model.hidden_size
            return model
        elif model_name == 'traj2vec_encoder':
            return modules.model.trembr.Traj2VecEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'traj2vec_decoder':
            return modules.model.trembr.Traj2VecDecoder(**model_config)
        elif model_name == 'cae_encoder':
            return modules.model.cnn.CNNEncoder(sampler=sampler, **model_config)
        elif model_name == 'cae_decoder':
            return modules.model.cnn.CNNDecoder(**model_config)
        elif model_name == 'geoconstrains_skipgram':
            return modules.model.word2vec.GeoConstrainSkipGramEncoder(sampler=sampler, **model_config)
        elif model_name == 'dual_view_encoder':
            return modules.model.dual_view.DualViewEncoder(sampler=sampler, num_users=num_classes, **model_config)
        elif model_name == 'robustDAAEncoder':
            return modules.model.robustDAA.RobustDAA_Encoder(sampler=sampler, **model_config)
        elif model_name == 'robustDAADecoder':
            return modules.model.robustDAA.RobustDAA_Decoder(**model_config)
        elif model_name == 'robustDAA_attention':
            return modules.model.robustDAA.RobustDAA_Attention(**model_config)
        elif model_name == 'maerrcdvit':
            return modules.model.light_path.MAERRCD(sampler=sampler, num_roads=num_roads, **model_config)
        else:
            raise NotImplementedError(f'No model called "{model_name}".')

    def create_augmentation(aug_entry):
        """
        Create data augmentation sampler based on configuration.
        
        Args:
            aug_entry (dict): Augmentation configuration containing name and parameters
            
        Returns:
            Sampler: Initialized augmentation sampler
            
        Raises:
            NotImplementedError: If augmentation name is not recognized
        """
        aug_name = aug_entry['name']
        aug_config = aug_entry.get('config', {})
        
        if aug_name == 'pass':
            return modules.preprocessor.PassSampler()
        elif aug_name == 'khop':
            return modules.preprocessor.KHopSampler(**aug_config)
        elif aug_name == 'index':
            return modules.preprocessor.IndexSampler(**aug_config)
        elif aug_name == 'pool':
            return modules.preprocessor.PoolSampler(**aug_config)
        elif aug_name == 'Trajectory2VecSampler':
            return modules.preprocessor.Trajectory2VecSampler(**aug_config)
        elif aug_name == 'random':
            return modules.preprocessor.RandomViewSampler(**aug_config)
        else:
            raise NotImplementedError(f'No augmentation called "{aug_name}".')

    def setup_pretraining(entry, models, data, device, datetime_key, num_entry, repeat_i):
        """
        Set up and execute model pretraining based on configuration.
        
        Args:
            entry (dict): Full experiment configuration
            models (list): List of model instances
            data (Data): Dataset object
            device (str): Device to run training on
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
            
        Returns:
            tuple: (trainer, models)
            - trainer is the pretraining trainer instance
            - models are the pretrained model instances
        """
        if 'pretrain' not in entry:
            pre_trainer = PreTrainer.NoneTrainer(models=models, data=data, device=device)
            pre_trainer.save_models()
            print('Skip pretraining.')
            return pre_trainer, models

        pretrain_entry = entry['pretrain']
        loss_func = create_loss_functions(pretrain_entry['loss'], models)
        pre_trainer = create_pretrainer(pretrain_entry['trainer'], data, models, loss_func, device, 
                                      datetime_key, num_entry, repeat_i)

        # Handle training or loading
        if pretrain_entry.get('load', False):
            if pretrain_entry.get('load_epoch', None):
                pre_trainer.load_models(epoch=int(pretrain_entry['load_epoch']))
            else:
                pre_trainer.load_models()
        else:
            pre_trainer.train(pretrain_entry.get('resume', -1))

        return pre_trainer, pre_trainer.get_models()

    def run_downstream_tasks(entry, pre_trainer, models, data, device, datetime_key, num_entry, repeat_i):
        """
        Execute downstream tasks after pretraining.
        
        Args:
            entry (dict): Full experiment configuration
            pre_trainer: Pretraining trainer instance
            models (list): List of pretrained models
            data (Data): Dataset object
            device (str): Device to run training on
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
        """
        if 'downstream' not in entry:
            print('Finishing program without performing downstream tasks.')
            return

        for down_i, down_entry in enumerate(entry['downstream']):
            print(f'\n....{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat, '
                  f'{down_i+1}/{len(entry["downstream"])} downstream task ....\n')

            if down_i > 0:
                pre_trainer.load_models()
                models = pre_trainer.get_models()

            down_trainer = setup_downstream_task(down_entry, models, data, device, pre_trainer.BASE_KEY,
                                              datetime_key, num_entry, repeat_i)

            if down_entry.get('load', False):
                down_trainer.load_models()
            else:
                down_trainer.train()
            down_trainer.eval(down_entry['eval_set'])

    # Main execution flow
    args, device = parse_args()
    datetime_key = utils.get_datetime_key()
    print('Datetime key', datetime_key)
    torch.autograd.set_detect_anomaly(True)

    # Load config file
    with open(f'config/{args.config}.json', 'r') as fp:
        config = json.load(fp)

    for num_entry, entry in enumerate(config):
        print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')
        
        # Load dataset
        data = load_data(entry['data'])
        
        # Save config
        conf_save_dir = os.path.join(data.base_path, 'config')
        utils.create_if_noexists(conf_save_dir)
        with open(os.path.join(conf_save_dir, f'{datetime_key}_e{num_entry}.json'), 'w') as fp:
            json.dump(entry, fp)

        # Run experiments
        num_repeat = entry.get('repeat', 1)
        for repeat_i in range(num_repeat):
            print(f'\n----{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat----\n')
            
            # Create models
            models = [create_model(model_entry, data, data.data_info['num_road'], 
                                   data.data_info['num_class'], 'pretrain' in entry) 
                      for model_entry in entry['models']]
            
            # Handle pretraining
            pre_trainer, models = setup_pretraining(entry, models, data, device, datetime_key, 
                                                  num_entry, repeat_i)
            
            # Run downstream tasks
            run_downstream_tasks(entry, pre_trainer, models, data, device, datetime_key, 
                               num_entry, repeat_i)

if __name__ == "__main__":
    main()