from evodiff.pretrained import OA_DM_38M
from evodiff.pretrained import OA_DM_640M
from evodiff.generate import generate_oaardm
from evodiff.generate import generate_GA_reward_metrics, generate_oaardm_reward_metrics, generate_oaardm_reward_metrics_edit, likelihood, generate_oaardm_reward_metrics_edit_initial
import os.path
import datetime
from datetime import date
import logging
import warnings
import pandas as pd
import numpy as np
import torch
import os, sys
import warnings
import os.path
from types import SimpleNamespace
import utils
import esm

from utils import set_seed
from args_file import get_args
from reward import *
#import esm2

import datetime
current_datetime = datetime.datetime.now()

MASKED_TOKEN = 'Z'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_WITH_MASK = ALPHABET + MASKED_TOKEN
MASK_TOKEN_INDEX = ALPHABET_WITH_MASK.index(MASKED_TOKEN)

warnings.filterwarnings("ignore", category=UserWarning)

# Get Arugments
args = get_args()

if args.diffusion_model == "38M":
    checkpoint = OA_DM_38M()
else:
    checkpoint= OA_DM_640M()


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


model, collater, tokenizer, scheme = checkpoint


iteration_number = args.num_symmetry # Symmetry
ori_pdb_file_path = "/home/ueharam1/projects3/Pdesign/datasets/AlphaFold_model_PDBs/" + args.proteinname  + ".pdb" 
seq_len = args.seq_length  
metrics_name = args.metrics_name.split(",")

if ('match_ss' in metrics_name) or ('crmsd' in metrics_name) or ('tm'in metrics_name):
    # Initialize PyRosetta
    pyrosetta.init()
    # Load the PDB file into a pose
    pose = pyrosetta.pose_from_pdb(ori_pdb_file_path)
    # Get the protein length (number of residues)
    seq_len = pose.size() # Reset sequence length
    print(seq_len)

if args.seed_design == "True": 
    repeat_num = args.repeatnum
    sequence = args.initial_seq 
    S_initial = torch.from_numpy(tokenizer.tokenize([sequence])).to(device)
    S_initial = S_initial.repeat(repeat_num, 1)


class RewardCal:
    def __init__(
            self,
            metrics_name: str,
            metrics_list: str,
            esm_model: str,
            device,
            run_name="",
            pdb_save_path="sc_tmp",
            ss_match='a',
    ):
        self.metrics_name = metrics_name.split(",")
        metrics_list = metrics_list.split(",")
        self.metrics_list = [float(x) for x in metrics_list]  # weight for different metrics
        assert len(self.metrics_name) == len(self.metrics_list)

        self.ss_match = ss_match

        self.pdb_save_path = pdb_save_path
        self.run_name = run_name

        # initialize the esmfold model
        self.folding_model = esm.pretrained.esmfold_v1().eval()  # esmfold 3b v1

        '''
        if esm_model == '3b':
            self.folding_model = esm2.esm.pretrained.esmfold_structure_module_only_3B().eval()
        elif esm_model == '15b':
            self.folding_model = esm2.esm.pretrained.esmfold_structure_module_only_15B().eval()
        elif esm_model == '650m':
            self.folding_model = esm2.esm.pretrained.esmfold_structure_module_only_650M().eval()
        elif esm_model == '150m':
            self.folding_model = esm2.esm.pretrained.esmfold_structure_module_only_150M().eval()
        elif esm_model == '35m':
            self.folding_model = esm2.esm.pretrained.esmfold_structure_module_only_35M().eval()
        elif esm_model == '8m':
            self.folding_model = esm2.esm.pretrained.esmfold_structure_module_only_8M().eval()
        else:
            raise NotImplementedError()
        '''
        self.folding_model = self.folding_model.to(device)

    def metrics_cal(self, metrics_name, ori_pdb_file=None, gen_pdb_file=None, folding_results=None, protein_idx=0, save_pdb=False, pdb_raw =None):
        all_results = []
        for metric in metrics_name:
            if metric == 'ptm':
                r = esm_to_ptm(folding_results, idx=protein_idx)
            elif metric == 'plddt':
                r = esm_to_plddt(folding_results, idx=protein_idx)
            elif metric == 'tm':
                r = pdb_to_tm(ori_pdb_file, pdb_raw)
            elif metric == 'crmsd':
                #r = pdb_to_crmsd(ori_pdb_file, gen_pdb_file)
                r = pdb_to_crmsd(ori_pdb_file, pdb_raw)
            elif metric == 'drmsd':
                r = pdb_to_drmsd(ori_pdb_file, gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'lddt':
                r = pdb_to_lddt(ori_pdb_file, gen_pdb_file)
            elif metric == 'hydrophobic':
                r = pdb_to_hydrophobic_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'match_ss':
                #r, _ = pdb_to_match_ss_score(ori_pdb_file if save_pdb else StringIO(gen_pdb_file))
                r, _ = pdb_to_match_ss_score(ori_pdb_file, gen_pdb_file if save_pdb else StringIO(gen_pdb_file), 1, seq_len+1)
                #r, _ = pdb_to_match_ss_score_original(ori_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'surface_expose':
                r = pdb_to_surface_expose_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'symmetry':
                r = symmetry_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file),  [ (i * seq_len) + 1  for i in range(iteration_number)], [(i+1) * seq_len+1 for i in range(iteration_number)] )
            elif metric == 'globularity':
                r = pdb_to_globularity_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            else:
                raise NotImplementedError()
            all_results.append(r)
        return all_results

    def reward_metrics(self, protein_name , mask_for_loss, S_sp, ori_pdb_file, save_pdb=False, add_info = ""):
        sc_output_dir = os.path.join(self.pdb_save_path, self.run_name)
        os.makedirs(sc_output_dir, exist_ok=True)
        esm_input_data = []
        
        if 'symmetry' in self.metrics_name:
            S_sp = S_sp.repeat(1,iteration_number)
            mask_for_loss = mask_for_loss.repeat(1,iteration_number) 
        for _it, ssp in enumerate(S_sp):
            seq_string = "".join([ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1])
            # protein sequence string
            esm_input_data.append(seq_string)

        # esmfold forward
        output = self.folding_model.infer(esm_input_data)
        pdbs = self.folding_model.output_to_pdb(output)

        # reward calculation
        record_reward, record_reward_agg = [], []
        for _it, pdb in enumerate(pdbs):
            if save_pdb:
                # save pdb for potential use
                pdb_path = os.path.join(sc_output_dir, f"{protein_name}" + str(_it) + "_" + str(add_info) +  ".pdb")
                with open(pdb_path, "w") as ff:
                    ff.write(pdb)
                # save fasta for potential use
                fasta_path = os.path.join(sc_output_dir, f"{protein_name}"+ str(_it) +"_" + str(add_info)+ ".fasta")
                header = f">{protein_name}"
                with open(fasta_path, 'w') as f:
                    f.write(f"{header}\n")
                    f.write(f"{esm_input_data[_it]}\n")
            else:
                pdb_path = pdb

            # calculate metrics
            all_reward = self.metrics_cal(
                metrics_name=self.metrics_name,
                gen_pdb_file=pdb_path,
                ori_pdb_file=ori_pdb_file,
                folding_results=output,
                protein_idx=_it,
                save_pdb=save_pdb,
                pdb_raw = pdb
            )
            aggregate_reward = sum(v * w for v, w in zip(all_reward, self.metrics_list))
            record_reward_agg.append(aggregate_reward)
            record_reward.append(all_reward)

        return record_reward, record_reward_agg, 0.0

if __name__ == "__main__":
    folder_path = f"log/" + str(current_datetime) + args.decoding + "_"+ str(args.metrics_name) + "_" + str(args.metrics_list) + "_" + str(args.proteinname)
    if os.path.exists(folder_path):
        warnings.warn(f"Result folder already exists: {folder_path}", RuntimeWarning)
    os.makedirs(folder_path, exist_ok=True)


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(f"{folder_path}", "run.log"),
    )
    logging.info(args)

    set_seed(args.seed, torch.cuda.is_available())

    folder_name = 'log'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    folder_name = '../.cache'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    repeat_num = args.repeatnum
    duplicate = args.duplicate
    valid_sp_acc, valid_sp_weights = 0., 0.
    gen_foldtrue_mpnn_results_merge = []
    gen_true_mpnn_results_merge = []
    foldtrue_true_mpnn_results_merge = []
    all_model_logl = []
    rewards_eval = []
    rewards = []

    model = model.to(device)

    # result save path
    # save_path = os.path.join(args.path_for_outputs, f'eval_{args.run_name}')

 
    new_reward_model = RewardCal(
        metrics_name=args.metrics_name,
        metrics_list=args.metrics_list,
        esm_model=args.esm_model,
        run_name=args.run_name,
        pdb_save_path = folder_path,  
        device=device,
    )

    reward_name_list = new_reward_model.metrics_name

    # model running
    all_result_dict = {}
    all_diversity, all_reward_agg, all_reward = [], [], []

    mask_for_loss = torch.ones((repeat_num, seq_len))
            # path for original pdb file
    protein_name = 'generated_proteins'
    edit_seqlength = int(args.edit_seqlength * seq_len)

    # Start Decoding

    if args.decoding == 'random':
        tokenized_sample, generated_sequence = generate_oaardm(model, tokenizer, seq_len, batch_size=repeat_num,
                                                                device=device)
    elif args.decoding == 'SVDD': # Baseline 
        tokenized_sample, generated_sequence = generate_oaardm_reward_metrics(
            model, tokenizer, seq_len, new_reward_model,
            ori_pdb_file_path=ori_pdb_file_path,
            batch = protein_name, 
            mask_for_loss= mask_for_loss,
            repeat_num=repeat_num,
            candidate=duplicate,
            edit_seqlength = edit_seqlength, 
            device=device,
        )
    elif args.decoding == 'SVDD_edit' and args.seed_design == 'True': 

        tokenized_sample, generated_sequence = generate_oaardm_reward_metrics_edit_initial(
        model, tokenizer, seq_len, new_reward_model,
        ori_pdb_file_path=ori_pdb_file_path,
        batch = protein_name, 
        mask_for_loss= mask_for_loss,
        repeat_num=repeat_num,
        candidate=duplicate,
        folder_path = folder_path,
        edit_seqlength = edit_seqlength,
        device=device,
        iteration = args.iteration, 
        initial_sample  = S_initial
    )

    elif args.decoding == 'SVDD_edit' and args.seed_design == 'False':
        tokenized_sample, generated_sequence = generate_oaardm_reward_metrics_edit(
            model, tokenizer, seq_len, new_reward_model,
            ori_pdb_file_path=ori_pdb_file_path,
            batch = protein_name, 
            mask_for_loss= mask_for_loss,
            repeat_num=repeat_num,
            candidate=duplicate,
            edit_seqlength = edit_seqlength,
            folder_path = folder_path,
            iteration = args.iteration, 
            device=device
        )
    elif args.decoding == 'SVDD_edit' and args.seed_design == "True":
        print('aaa')
        tokenized_sample, generated_sequence = generate_oaardm_reward_metrics_edit(
            model, tokenizer, seq_len, new_reward_model,
            ori_pdb_file_path=ori_pdb_file_path,
            batch = protein_name, 
            mask_for_loss= mask_for_loss,
            repeat_num=repeat_num,
            candidate=duplicate,
            edit_seqlength = edit_seqlength,
            folder_path = folder_path,
            iteration = args.iteration, 
            device=device
        ) 
    elif args.decoding == "GA" and args.seed_design == "True": # Baseline 
        tokenized_sample, generated_sequence = generate_GA_reward_metrics(
            model, tokenizer, seq_len, new_reward_model,
            ori_pdb_file_path=ori_pdb_file_path,
            batch = protein_name, 
            mask_for_loss= mask_for_loss,
            repeat_num=repeat_num,
            candidate=duplicate,
            edit_seqlength = edit_seqlength,
            folder_path = folder_path,
            device=device,
            iteration = args.iteration, 
            initial_sample  = S_initial
        )
    else:
        print(args.decoding)
        raise NotImplementedError()


    '''
    Start Evaluation
    '''

    args.metrics_name = args.metrics_name + ",plddt,ptm"
    args.metrics_list = args.metrics_list + ",1,1" 

    test_reward_model = RewardCal(
        metrics_name=args.metrics_name ,
        metrics_list=args.metrics_list ,
        esm_model=args.esm_model,
        run_name=args.run_name,
        pdb_save_path = folder_path,  
        device=device,
    )

    reward_name_list = test_reward_model.metrics_name

    likelihooed_reward = likelihood(model, tokenizer,seq_len, tokenized_sample, repeat_num, device)

    cur_reward_before, cur_reward_agg, _ = test_reward_model.reward_metrics(
        protein_name=protein_name,
        mask_for_loss= mask_for_loss ,
        S_sp=tokenized_sample,
        ori_pdb_file=ori_pdb_file_path,
        save_pdb=True,
    )


    cur_reward_before = list(map(list, zip(*cur_reward_before)))
    print(cur_reward_before)

    df = pd.DataFrame(np.array(cur_reward_before).transpose(), columns= reward_name_list)
    df['likelihood'] = likelihooed_reward 

    cur_reward = [sum(sublist) / len(sublist) for sublist in cur_reward_before]
    cur_reward_agg = sum(cur_reward_agg) / len(cur_reward_agg)

    cur_diversity = set_diversity(tokenized_sample.detach().cpu().numpy(), mask_for_loss.detach().cpu().numpy())
    df['cur_diversity'] = np.array([cur_diversity for i in range(repeat_num)])

    df.to_csv(folder_path + "/output.csv", index=False)

    # Save data
    ''''
    cur_protein_prefix = f"{args.decoding}_{args.reward_name}"
    cur_result_dict = {f"{cur_protein_prefix}_{reward_name_list[i]}": cur_reward[i] for i in range(len(reward_name_list))}
    cur_result_dict[f'{cur_protein_prefix}_reward_agg'] = cur_reward_agg
    cur_result_dict[f'{cur_protein_prefix}_diversity'] = cur_diversity
    logging.info(f"current result: {cur_result_dict}")
    assert not set(all_result_dict.keys()) & set(cur_result_dict.keys())
    all_result_dict.update(cur_result_dict)

    all_diversity.append(cur_diversity)
    all_reward_agg.append(cur_reward_agg)
    all_reward.append(cur_reward)

    all_reward = list(map(list, zip(*all_reward)))
    all_reward_mean = [sum(sublist) / len(sublist) for sublist in all_reward]

    final_result_dict = {f"final_{reward_name_list[i]}": all_reward_mean[i] for i in range(len(reward_name_list))}
    final_result_dict['final_reward_agg'] = sum(all_reward_agg) / len(all_reward_agg)
    final_result_dict['final_diversity'] = sum(all_diversity) / len(all_diversity)
    logging.info(f"final result: {final_result_dict}")
    all_result_dict.update(final_result_dict)

    with open(folder_path + "/result_dict.json", "w") as f:
        json.dump(all_result_dict, f)
    '''
