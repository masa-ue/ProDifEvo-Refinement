import argparse


def get_args():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    argparser.add_argument("--path_for_pdbs", type=str, default="../datasets/AlphaFold_model_PDBs",
                           help="path for loading pdb files")
    argparser.add_argument("--path_for_outputs", type=str, default="../datasets", help="path for loading pdb files")
 


    argparser.add_argument("--run_name", type=str, default="debug", help="run name for save the results")
    
    argparser.add_argument("--num_timesteps", type=int, default=50)  # 500
    argparser.add_argument("--seed", type=int, default=0)

    argparser.add_argument("--alpha", type=float, default=0.001)
    argparser.add_argument("--dps_scale", type=float, default=0.0)
    argparser.add_argument("--tds_alpha", type=float, default=0.0)

    # Important parameters
    argparser.add_argument("--seq_length", type=int, default=120, help = "We don't need to specify when optimizing crmsd,match_ss,tm")
    argparser.add_argument("--repeatnum", type=int, default=10, help = "Batch size")
    argparser.add_argument("--duplicate", type=int, default=20) 
    argparser.add_argument("--decoding", type=str, default='original')

    # Pre-trained diffusion model
    argparser.add_argument("--diffusion_model", type=str, default='38M')
    # ESM model (refold model)
    argparser.add_argument("--esm_model", type=str, default='650m')
    argparser.add_argument("--proteinname", type= str, default = "None", help = "when optimizing crmsd,tm,match_ss, we should specify")
    # Reward setting
    argparser.add_argument("--metrics_name", type=str, required=True,
                           help="ptm,plddt,tm,crmsd,drmsd,lddt,hydrophobic,match_ss,surface_expose,symmetry,globularity")
    argparser.add_argument("--metrics_list", type=str, required=True, help='weight for different metrics')
    argparser.add_argument("--edit_seqlength", type = float, default = 0.1, help = "how many edits we do in each iteration") 
    argparser.add_argument("--iteration", type = int, default = 50, help = "Iteration number of our proposal")
    argparser.add_argument("--num_symmetry", type = int, default = 8, help ='Number of symmetries')
    # If we have seed designs, change the folloiwing
    argparser.add_argument("--seed_design", type = str, default = "False", help ="Do we have seed sequences?")
    argparser.add_argument("--initial_seq", type = str, default = "AA", help = "Seed seqyebces")
    args = argparser.parse_args()
    return args
