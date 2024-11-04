from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "bc_xfmr"
    
    if args.addmask:
        config_file = 'robomimic/exps/templates/bc_transformer_addmask.json'
    else:
        config_file = 'robomimic/exps/templates/bc_transformer.json'

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, config_file),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    
    
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"


    # EVAL_TASKS = ["OpenDrawer", "CloseDrawer"] # or evaluate all tasks by setting EVAL_TASKS = None
    # EVAL_TASKS = None
    EVAL_TASKS = ["PnPCounterToCab"]
    if not args.ckpt:
        ### Multi-task training on atomic tasks ###
        generator.add_param(
            key="train.data",
            name="ds",
            group=123456,
            values_and_names=[
                # (get_ds_cfg("single_stage", src="human", eval=EVAL_TASKS, filter_key="50_demos"), "human-50"), # training on human datasets
                # (get_ds_cfg("single_stage", src="mg", eval=EVAL_TASKS, filter_key="3000_demos"), "mg-3000"), # training on MimicGen datasets
                (get_ds_cfg("PnPCounterToCab", src="mg", eval=EVAL_TASKS, filter_key="3000_demos"), "mg-3000"), # training on one dataset
            ]
        )
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[args.enable_rollout],
            value_names=[""],
        )

    """
    ### Uncomment this code to train composite task policies ###
    generator.add_param(
        key="train.data",
        name="ds",
        group=123456,
        values_and_names=[
            (get_ds_cfg("ArrangeVegetables", gen_tex=False, rand_cams=False, filter_key="50_demos"), "ArrangeVegetables"),
            (get_ds_cfg("MicrowaveThawing", gen_tex=False, rand_cams=False, filter_key="50_demos"), "MicrowaveThawing"),
            (get_ds_cfg("RestockPantry", gen_tex=False, rand_cams=False, filter_key="50_demos"), "RestockPantry"),
            (get_ds_cfg("PreSoakPan", gen_tex=False, rand_cams=False, filter_key="50_demos"), "PreSoakPan"),
            (get_ds_cfg("PrepareCoffee", gen_tex=False, rand_cams=False, filter_key="50_demos"), "PrepareCoffee"),
        ]
    )
    generator.add_param(
        key="experiment.ckpt_path",
        name="ckpt",
        group=1389,
        values_and_names=[
            (None, "none"),
            # ("set checkpoint pth path here", "trained-ckpt"),
        ],
    )
    """
    
    if args.ckpt:
        ## Uncomment this code to evaluate checkpoints ###
        generator.add_param(
            key="train.data",
            name="ds",
            group=1389,
            values_and_names=[
                # (get_ds_cfg("single_stage", src="human", eval=EVAL_TASKS, filter_key="50_demos"), "human-50"),
                # (get_ds_cfg("single_stage", src="mg", eval=EVAL_TASKS, filter_key="3000_demos"), "mg-3000"),
                (get_ds_cfg("PnPCounterToCab", src="mg", eval=EVAL_TASKS, filter_key="3000_demos"), "mg-3000"), # training on one dataset
            ],
        )
        generator.add_param(
            key="experiment.ckpt_path",
            name="ckpt",
            group=1389,
            values_and_names=[
                (args.ckpt, "trained-ckpt"),
            ],
        )
        generator.add_param(
            key="experiment.rollout.warmstart",
            name="",
            group=-1,
            values=[-1],
        )
        generator.add_param(
            key="train.num_epochs",
            name="",
            group=-1,
            values=[0],
        )
        generator.add_param(
            key="train.num_data_workers",
            name="",
            group=-1,
            values=[0],
        )
    

    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
