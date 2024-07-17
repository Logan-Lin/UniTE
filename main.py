import json
import os
from argparse import ArgumentParser
import copy

import torch
from torch.cuda import is_available as cuda_available
# import dgl

from data import Data
from pretrain import trainer as PreTrainer, generative_losses, contrastive_losses
from model import sample as EncSampler, ode, transformer, induced_att, rnn, prior, gmvsae, start, trajectory2vec, trajectorysim, trembr, cnn, word2vec, t2vec, dual_view, robustDAA, light_path
from downstream import predictor as DownPredictor, task
import utils

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('-c', '--config', help='name of the config file to use', type=str, default="robustDAA/small_test") # TrajectorySim / t2vec / gmvsae / robustDAA
parser.add_argument('--cuda', help='index of the cuda device to use', type=int, default=0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
device = f'cuda:0' if cuda_available() else 'cpu'
datetime_key = utils.get_datetime_key()
print('Datetime key', datetime_key)
torch.autograd.set_detect_anomaly(True)

# Load config file
with open(f'config/{args.config}.json', 'r') as fp:
    config = json.load(fp)

# Each config file can contain multiple entries. Each entry is a different set of configuration.
for num_entry, entry in enumerate(config):
    print(f'\n{"=" * 30}\n===={num_entry+1}/{len(config)} experiment entry====')

    # Load dataset.
    data_entry = entry['data']
    data = Data(data_entry['name'], data_entry.get('road_type', 'road_network'))
    data.load_stat()
    num_roads = data.data_info['num_road']
    num_classes = data.data_info['num_class']
    num_w = data.data_info['num_w'] if 'num_w' in data.data_info.index else None
    num_h = data.data_info['num_h'] if 'num_h' in data.data_info.index else None

    conf_save_dir = os.path.join(data.base_path, 'config')
    utils.create_if_noexists(conf_save_dir)
    with open(os.path.join(conf_save_dir, f'{datetime_key}_e{num_entry}.json'), 'w') as fp:
        json.dump(entry, fp)

    # Each entry can be repeated for several times.
    num_repeat = entry.get('repeat', 1)
    for repeat_i in range(num_repeat):
        print(f'\n----{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat----\n')

        models = []
        for model_entry in entry['models']:
            # Prepare samplers for models.
            if 'sampler' in model_entry:
                sampler_entry = model_entry['sampler']
                sampler_name = sampler_entry['name']
                sampler_config = sampler_entry.get('config', {})

                if sampler_name == 'khop':
                    sampler = EncSampler.KHopSampler(**sampler_config)
                elif sampler_name == 'pass':
                    sampler = EncSampler.PassSampler()
                elif sampler_name == 'index':
                    sampler = EncSampler.IndexSampler(**sampler_config)
                elif sampler_name == 'pool':
                    sampler = EncSampler.PoolSampler(**sampler_config)
                elif sampler_name == 'Trajectory2VecSampler':
                    sampler = EncSampler.Trajectory2VecSampler(**sampler_config)
                elif sampler_name == 'random':
                    sampler = EncSampler.RandomViewSampler(**sampler_config)
                else:
                    raise NotImplementedError(f'No sampler called "{sampler_name}".')
            else:
                sampler = EncSampler.PassSampler()

            # Create models.
            model_name = model_entry['name']
            model_config = model_entry.get('config', {})
            if "pre_embed" in model_config:
                model_config["pre_embed"] = data.load_meta(model_config.get("pre_embed"), 0)[0]
                model_config["pre_embed_update"] = model_config.get("pre_embed_update", True)

            if model_name == 'ia':
                models.append(induced_att.InducedAttEncoder(
                    sampler=sampler,
                    **model_config))
            elif model_name == 'transformer_encoder':
                models.append(transformer.TransformerEncoder(
                    sampler=sampler,
                    **model_config))
            elif model_name == 'transformer_decoder':
                models.append(transformer.TransformerDecoder(**model_config))
            elif model_name == 'transformer_denoiser':
                models.append(transformer.TransformerDenoiser(**model_config))
            elif model_name == 'dualpos_transformer':
                models.append(transformer.DualPosTransformer(sampler=sampler, **model_config))
            elif model_name == 'mlm_transformer':
                models.append(transformer.MLMTransformer(sampler=sampler, **model_config))
            elif model_name == 'cde':
                models.append(ode.CDEEncoder(sampler=sampler, **model_config))
            elif model_name == 'coa':
                models.append(ode.CoeffAttEncoder(sampler=sampler, **model_config))
            elif model_name == 'stode':
                models.append(ode.STODEEncoder(sampler=sampler, **model_config))
            elif model_name == 'trajode_decoder':
                models.append(ode.TrajODEDecoder(**model_config))
            # elif model_name == 'gat_denoiser':
            #     models.append(gnn.GATDenoiser(graph=dgl_graph, input_size=3, **model_config))
            elif model_name == 'rnn_encoder':
                models.append(rnn.RnnEncoder(sampler=sampler, num_embed=num_roads,
                                             **model_config))
            elif model_name == 'rnn_decoder':
                models.append(rnn.RnnDecoder(num_roads=num_roads,
                                             **model_config))
            elif model_name == 'prior_encoder':
                models.append(prior.PriorEncoder(sampler=sampler, **model_config))
            elif model_name == 'gmvsae_encoder':
                models.append(gmvsae.GMVSAEEncoder(num_embed=num_roads, sampler=sampler, **model_config))
            elif model_name == 'gmvsae_decoder':
                models.append(gmvsae.GMVSAEDecoder(num_embed=num_roads, **model_config))
            elif model_name == 'bert':
                models.append(start.BERTEncoder(sampler=sampler, vocab_size=num_roads, **model_config))
            elif model_name == 'trajectory2vec_encoder':
                models.append(trajectory2vec.Trajectory2VecEncoder(sampler=sampler, **model_config))
            elif model_name == 'trajectory2vec_decoder':
                models.append(trajectory2vec.Trajectory2vecDecoder(sampler=sampler, **model_config))
            elif model_name == 'trajsim_embedding':
                models.append(trajectorysim.TrajSimEmbed(meta_dir=data.meta_dir, **model_config, pretrain=('pretrain' in entry)))
                vocab_size = models[-1].vocab_size
                dist_path = models[-1].dist_path
            elif model_name == 'trajsim_encoder':
                models.append(trajectorysim.TrajSimEncoder(num_embed=num_roads, sampler=sampler, **model_config))
            elif model_name == 'trajsim_decoder':
                models.append(trajectorysim.TrajSimDecoder(**model_config))
                hidden_size = models[-1].hidden_size
            elif model_name == 't2vecEmbedding':
                models.append(t2vec.t2vecEmbedding(meta_dir=data.meta_dir, **model_config, pretrain=('pretrain' in entry)))
                vocab_size = models[-1].vocab_size
                dist_path = models[-1].dist_path
            elif model_name == 't2vecEncoder':
                models.append(t2vec.t2vecEncoder(num_embed=num_roads, sampler=sampler, **model_config))
            elif model_name == 't2vecDecoder':
                models.append(t2vec.t2vecDecoder(**model_config))
                hidden_size = models[-1].hidden_size
            elif model_name == 'traj2vec_encoder':
                models.append(trembr.Traj2VecEncoder(num_embed=num_roads, sampler=sampler,
                                                     **model_config))
            elif model_name == 'traj2vec_decoder':
                models.append(trembr.Traj2VecDecoder(**model_config))
            elif model_name == 'cae_encoder':
                models.append(cnn.CNNEncoder(sampler=sampler, **model_config))
            elif model_name == 'cae_decoder':
                models.append(cnn.CNNDecoder(**model_config))
            elif model_name == 'geoconstrains_skipgram':
                models.append(word2vec.GeoConstrainSkipGramEncoder(sampler=sampler, **model_config))
            elif model_name == 'dual_view_encoder':
                models.append(dual_view.DualViewEncoder(sampler=sampler, num_users=num_classes, **model_config))
            elif model_name == 'robustDAAEncoder':
                models.append(robustDAA.RobustDAA_Encoder(sampler=sampler, **model_config))
            elif model_name == 'robustDAADecoder':
                models.append(robustDAA.RobustDAA_Decoder(**model_config))
            elif model_name == 'robustDAA_attention':
                models.append(robustDAA.RobustDAA_Attention(**model_config))
            elif model_name == 'maerrcdvit':
                models.append(light_path.MAERRCD(sampler=sampler, num_roads=num_roads, **model_config))
            else:
                raise NotImplementedError(f'No encoder called "{model_name}".')

        if 'pretrain' in entry:
            # Create pre-training loss function.
            pretrain_entry = entry['pretrain']
            loss_entries = pretrain_entry['loss']
            loss_func = []
            if isinstance(pretrain_entry['loss'], dict):
                loss_entries = [loss_entries]

            for loss_entry in loss_entries:
                loss_name = loss_entry['name']

                loss_param = loss_entry.get('config', {})
                if loss_name == 'infonce':
                    loss_func.append(contrastive_losses.InfoNCE(**loss_param))
                elif loss_name == 'mec':
                    loss_func.append(contrastive_losses.MEC(**loss_param,
                                                       teachers=(copy.deepcopy(model) for model in models)))
                elif loss_name == 'ddpm':
                    loss_func.append(generative_losses.DDPM(**loss_param))
                elif loss_name == 'autoreg':
                    loss_func.append(generative_losses.AutoRegressive(**loss_param))
                elif loss_name == 'mlm':
                    loss_func.append(generative_losses.MLM(**loss_param))
                elif loss_name == 'gmvsae':
                    loss_func.append(generative_losses.GMVSAE(**loss_param))
                elif loss_name == 'simclr':
                    loss_func.append(contrastive_losses.SimCLR(**loss_param))
                elif loss_name == 'trajectory2vec':
                    loss_func.append(generative_losses.Trajectory2Vec(**loss_param))
                elif loss_name == 'trajsim':
                    loss_func.append(generative_losses.TrajectorySim(device=device, hidden_size=hidden_size, vocab_size=vocab_size,
                                                                knn_vocabs_path=dist_path, **loss_param))
                elif loss_name == 't2vec':
                    loss_func.append(generative_losses.t2vec(device=device, hidden_size=hidden_size, vocab_size=vocab_size,
                                                                knn_vocabs_path=dist_path, **loss_param))
                elif loss_name == 'trembr':
                    loss_func.append(generative_losses.Trembr(num_roads=num_roads, **loss_param))
                elif loss_name == 'cae':
                    loss_func.append(generative_losses.ConvolutionalAutoRegressive(**loss_param))
                elif loss_name == 'geoconstrains_word2vec':
                    loss_func.append(contrastive_losses.GeoConstrainWord2Vec(**loss_param))
                elif loss_name == 'robustDAA':
                    loss_func.append(generative_losses.RobustDAA(**loss_param))
                elif loss_name == 'trajode':
                    loss_func.append(generative_losses.TrajODE(**loss_param))
                elif loss_name == 'maerr':
                    loss_func.append(generative_losses.MAERR(**loss_param))
                else:
                    raise NotImplementedError(f'No loss function called "{loss_name}".')

            if isinstance(pretrain_entry['loss'], dict):
                loss_func = loss_func[0]

            # Create pre-trainer.
            pretrainer_entry = pretrain_entry['trainer']
            pretrainer_name = pretrainer_entry['name']
            pretrainer_comm_param = {"data": data, "models": models, "loss_func": loss_func,
                                     "device": device, "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'}
            pretrainer_config = pretrainer_entry.get('config', {})
            if pretrainer_name == 'contrastive':
                pre_trainer = PreTrainer.ContrastiveTrainer(
                    **pretrainer_comm_param,
                    **pretrainer_config)
            elif pretrainer_name == 'generative':
                pre_trainer = PreTrainer.GenerativeTrainer(
                    **pretrainer_comm_param,
                    **pretrainer_config)
            elif pretrainer_name == 'generativeiteration':
                pre_trainer = PreTrainer.GenerativeIterationTrainer(
                    **pretrainer_comm_param,
                    **pretrainer_config)
            elif pretrainer_name == 'momentum':
                pre_trainer = PreTrainer.MomentumTrainer(
                    **pretrainer_comm_param,
                    **pretrainer_config)
            elif pretrainer_name == 'multiple':
                pre_trainer = PreTrainer.MultiTrainer(
                    **pretrainer_comm_param,
                    **pretrainer_config
                )
            elif pretrainer_name == 'ADMM':
                pre_trainer = PreTrainer.ADMMTrainer(
                    **pretrainer_comm_param,
                    **pretrainer_config)
            else:
                raise NotImplementedError(f'No loss function called "{pretrainer_name}".')

            # Pre-training on the training set, or load from trained cache.
            if pretrain_entry.get('load', False):
                if pretrain_entry.get('load_epoch', None):
                    pre_trainer.load_models(epoch=int(pretrain_entry['load_epoch']))
                else:
                    pre_trainer.load_models()
            else:
                pre_trainer.train(pretrain_entry.get('resume', -1))

            if "generation" in pretrain_entry:
                generation_entry = pretrain_entry['generation']
                pre_trainer.generate(generation_entry['eval_set'],
                                     **generation_entry.get('config', {}))

            models = pre_trainer.get_models()

            for i_model, model in enumerate(models):
                model_entry = entry['models'][i_model]
                if 'down_sampler' in model_entry:
                    down_sampler_entry = model_entry['down_sampler']
                    model.sampler = utils.load_sampler(
                        down_sampler_entry['name'], down_sampler_entry.get('config', {}))
        else:
            pre_trainer = PreTrainer.NoneTrainer(models=models, data=data, device=device)
            pre_trainer.save_models()
            print('Skip pretraining.')

        # Downstream evaluation
        if 'downstream' in entry:
            num_down = len(entry['downstream'])
            for down_i, down_entry in enumerate(entry['downstream']):
                print(f'\n....{num_entry+1}/{len(config)} experiment entry, {repeat_i+1}/{num_repeat} repeat, '
                      f'{down_i+1}/{num_down} downstream task ....\n')

                if down_i > 0:
                    pre_trainer.load_models()
                    models = pre_trainer.get_models()

                down_models = [models[i] for i in down_entry['select_models']]
                down_embed_size = sum([model.output_size for model in down_models])
                down_task = down_entry['task']
                down_config = down_entry.get('config', {})

                down_comm_params = {
                    "data": data, "models": down_models, "device": device,
                    "base_name": pre_trainer.BASE_KEY, "log_name_key": datetime_key + f'_e{num_entry}_r{repeat_i}'}
                predictor_entry = down_entry.get('predictor', {})
                predictor_config = predictor_entry.get('config', {})
                if down_task == 'classification':
                    predictor = DownPredictor.FCPredictor(
                        input_size=down_embed_size, output_size=data.data_info['num_class'],
                        **predictor_config)
                    down_trainer = task.Classification(
                        predictor=predictor,
                        **down_comm_params, **down_config)
                elif down_task == 'destination':
                    predictor = DownPredictor.FCPredictor(
                        input_size=down_embed_size, output_size=num_roads,
                        **predictor_config)
                    down_trainer = task.Destination(
                        predictor=predictor,
                        **down_comm_params, **down_config)
                elif down_task == 'search':
                    predictor = DownPredictor.NonePredictor()
                    down_trainer = task.Search(
                        predictor=predictor,
                        **down_comm_params, **down_config)
                elif down_task == 'tte':
                    predictor = DownPredictor.FCPredictor(
                        input_size=down_embed_size, output_size=1,
                        **predictor_config)
                    down_trainer = task.TTE(
                        predictor=predictor,
                        **down_comm_params, **down_config)
                else:
                    raise NotImplementedError(f'No downstream task called "{down_task}".')

                if down_entry.get('load', False):
                    down_trainer.load_models()
                else:
                    down_trainer.train()
                down_trainer.eval(down_entry['eval_set'])
        else:
            print('Finishing program without performing downstream tasks.')
