import json
import os
from argparse import ArgumentParser
import copy

import torch
from torch.cuda import is_available as cuda_available
import yaml

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
        parser.add_argument('-c', '--config', help='path of the config file to use', type=str, required=True)
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
        # Add global declarations at the start of the function
        global vocab_size, dist_path, hidden_size
        
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
            vocab_size = model.vocab_size
            dist_path = model.dist_path
            return model
        elif model_name == 'trajsim_encoder':
            return modules.model.trajectorysim.TrajSimEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 'trajsim_decoder':
            model = modules.model.trajectorysim.TrajSimDecoder(**model_config)
            hidden_size = model.hidden_size
            return model
        elif model_name == 't2vecEmbedding':
            model = modules.model.t2vec.t2vecEmbedding(meta_dir=data.meta_dir, **model_config, pretrain=pretrain)
            vocab_size = model.vocab_size
            dist_path = model.dist_path
            return model
        elif model_name == 't2vecEncoder':
            return modules.model.t2vec.t2vecEncoder(num_embed=num_roads, sampler=sampler, **model_config)
        elif model_name == 't2vecDecoder':
            model = modules.model.t2vec.t2vecDecoder(**model_config)
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

    def create_loss_functions(loss_entries, models):
        """
        Create loss functions based on configuration entries.
        
        Args:
            loss_entries (dict or list): Loss function configurations
            models (list): List of model instances
            
        Returns:
            list or object: Single loss function or list of loss functions
            
        Raises:
            NotImplementedError: If loss function name is not recognized
        """
        # Handle single loss entry case
        if isinstance(loss_entries, dict):
            loss_entries = [loss_entries]
            single_loss = True
        else:
            single_loss = False

        loss_funcs = []
        for loss_entry in loss_entries:
            loss_name = loss_entry['name']
            loss_param = loss_entry.get('config', {})
            
            if loss_name == 'infonce':
                loss_funcs.append(contrastive_losses.InfoNCE(**loss_param))
            elif loss_name == 'mec':
                loss_funcs.append(contrastive_losses.MEC(**loss_param,
                                                       teachers=(copy.deepcopy(model) for model in models)))
            elif loss_name == 'ddpm':
                loss_funcs.append(generative_losses.DDPM(**loss_param))
            elif loss_name == 'autoreg':
                loss_funcs.append(generative_losses.AutoRegressive(**loss_param))
            elif loss_name == 'mlm':
                loss_funcs.append(generative_losses.MLM(**loss_param))
            elif loss_name == 'gmvsae':
                loss_funcs.append(generative_losses.GMVSAE(**loss_param))
            elif loss_name == 'simclr':
                loss_funcs.append(contrastive_losses.SimCLR(**loss_param))
            elif loss_name == 'trajectory2vec':
                loss_funcs.append(generative_losses.Trajectory2Vec(**loss_param))
            elif loss_name == 'trajsim':
                loss_funcs.append(generative_losses.TrajectorySim(device=device, 
                                                                hidden_size=hidden_size,
                                                                vocab_size=vocab_size,
                                                                knn_vocabs_path=dist_path, 
                                                                **loss_param))
            elif loss_name == 't2vec':
                loss_funcs.append(generative_losses.t2vec(device=device,
                                                        hidden_size=hidden_size,
                                                        vocab_size=vocab_size,
                                                        knn_vocabs_path=dist_path,
                                                        **loss_param))
            elif loss_name == 'trembr':
                loss_funcs.append(generative_losses.Trembr(num_roads=num_roads, **loss_param))
            elif loss_name == 'cae':
                loss_funcs.append(generative_losses.ConvolutionalAutoRegressive(**loss_param))
            elif loss_name == 'geoconstrains_word2vec':
                loss_funcs.append(contrastive_losses.GeoConstrainWord2Vec(**loss_param))
            elif loss_name == 'robustDAA':
                loss_funcs.append(generative_losses.RobustDAA(**loss_param))
            elif loss_name == 'trajode':
                loss_funcs.append(generative_losses.TrajODE(**loss_param))
            elif loss_name == 'maerr':
                loss_funcs.append(generative_losses.MAERR(**loss_param))
            else:
                raise NotImplementedError(f'No loss function called "{loss_name}".')

        return loss_funcs[0] if single_loss else loss_funcs

    def create_pretrainer(pretrainer_entry, data, models, loss_func, device, datetime_key, num_entry, repeat_i):
        """
        Create pretraining trainer based on configuration.
        
        Args:
            pretrainer_entry (dict): Trainer configuration
            data (Data): Dataset object
            models (list): List of model instances
            loss_func: Loss function(s)
            device (str): Device to run training on
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
            
        Returns:
            Trainer: Initialized trainer instance
            
        Raises:
            NotImplementedError: If trainer name is not recognized
        """
        pretrainer_name = pretrainer_entry['name']
        pretrainer_config = pretrainer_entry.get('config', {})
        
        # Common parameters for all trainers
        common_params = {
            "data": data,
            "models": models,
            "loss_func": loss_func,
            "device": device,
            "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'
        }
        
        if pretrainer_name == 'contrastive':
            return PreTrainer.ContrastiveTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'generative':
            return PreTrainer.GenerativeTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'generativeiteration':
            return PreTrainer.GenerativeIterationTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'momentum':
            return PreTrainer.MomentumTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'multiple':
            return PreTrainer.MultiTrainer(**common_params, **pretrainer_config)
        elif pretrainer_name == 'ADMM':
            return PreTrainer.ADMMTrainer(**common_params, **pretrainer_config)
        else:
            raise NotImplementedError(f'No trainer called "{pretrainer_name}".')

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
                                              datetime_key, num_entry, repeat_i, data.data_info['num_road'])

            if down_entry.get('load', False):
                down_trainer.load_models()
            else:
                down_trainer.train()
            down_trainer.eval(down_entry['eval_set'])

    def setup_downstream_task(down_entry, models, data, device, base_key, datetime_key, num_entry, repeat_i, num_roads):
        """
        Set up downstream task trainer based on configuration.
        
        Args:
            down_entry (dict): Downstream task configuration
            models (list): List of pretrained models
            data (Data): Dataset object
            device (str): Device to run training on
            base_key (str): Base key for model loading
            datetime_key (str): Unique datetime identifier
            num_entry (int): Current experiment index
            repeat_i (int): Current repetition index
            
        Returns:
            Trainer: Initialized downstream task trainer
            
        Raises:
            NotImplementedError: If task name is not recognized
        """
        # Select models and calculate embedding size
        down_models = [models[i] for i in down_entry['select_models']]
        down_embed_size = sum([model.output_size for model in down_models])
        
        # Get task configuration
        down_task = down_entry['task']
        down_config = down_entry.get('config', {})
        predictor_entry = down_entry.get('predictor', {})
        predictor_config = predictor_entry.get('config', {})
        
        # Common parameters for all tasks
        common_params = {
            "data": data,
            "models": down_models,
            "device": device,
            "base_name": base_key,
            "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'
        }
        
        # Create appropriate predictor and trainer based on task
        if down_task == 'classification':
            predictor = DownPredictor.FCPredictor(
                input_size=down_embed_size,
                output_size=data.data_info['num_class'],
                **predictor_config
            )
            return task.Classification(predictor=predictor, **common_params, **down_config)
        
        elif down_task == 'destination':
            predictor = DownPredictor.FCPredictor(
                input_size=down_embed_size,
                output_size=num_roads,
                **predictor_config
            )
            return task.Destination(predictor=predictor, **common_params, **down_config)
        
        elif down_task == 'search':
            predictor = DownPredictor.NonePredictor()
            return task.Search(predictor=predictor, **common_params, **down_config)
        
        elif down_task == 'tte':
            predictor = DownPredictor.FCPredictor(
                input_size=down_embed_size,
                output_size=1,
                **predictor_config
            )
            return task.TTE(predictor=predictor, **common_params, **down_config)
        
        else:
            raise NotImplementedError(f'No downstream task called "{down_task}".')

    # Main execution flow
    args, device = parse_args()
    datetime_key = utils.get_datetime_key()
    print('Datetime key', datetime_key)
    torch.autograd.set_detect_anomaly(True)

    # Load config file
    if args.config.endswith('.json'):
        with open(args.config, 'r') as fp:
            config = json.load(fp)
    elif args.config.endswith('.yaml') or args.config.endswith('.yml'):
        import yaml
        with open(args.config, 'r') as fp:
            config = yaml.safe_load(fp)
    else:
        raise ValueError(f"Config file must be .json, .yaml or .yml, got {args.config}")

    for num_entry, entry in enumerate(config):
        print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')
        
        # Load dataset
        data = load_data(entry['data'])
        
        # Save config
        conf_save_dir = os.path.join(data.base_path, 'config')
        utils.create_if_noexists(conf_save_dir)
        with open(os.path.join(conf_save_dir, f'{datetime_key}_e{num_entry}.yaml'), 'w') as fp:
            yaml.dump(entry, fp)

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