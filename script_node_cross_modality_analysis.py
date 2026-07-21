# Hack to avoid some import problem due to the library being a subfolder
import random
import sys
import torch
import argparse
import logging
import os

try:
    sys.path.append("third_party/TransformerLens")
    import transformer_lens as lens  # Some python problem causes this to throw on the first import
except:
    import transformer_lens as lens


from component import Component
from general_utils import (
    set_deterministic,
    get_top_scoring_components,
)
from analysis_utils import (
    SUPPORTED_TASKS,
    get_limited_labels_for_task,
    load_l_vl_scores,
    load_model,
    load_dataset,
)
from modality_alignment_utils import (
    POS_MAPPINGS,
    convert_components_modality,
    get_image_positions,
    get_text_sequence_positions,
)
from evaluation_utils import circuit_faithfulness

DISCOVERY_EVAL_SPLIT_PERCENT = 0.75
METRICS = ["LD"]

torch.set_grad_enabled(False)
device = "cuda"


def load_l_vl_datasets(model, processor, args):
    """
    Loads both L (textual) and VL (visual) evaluation datasets for a given task and model.
    """
    _, __, eval_l_prompts = load_dataset(
        model=model,
        processor=processor,
        task_name=args.task_name,
        model_name=args.model_name,
        language_only=True,
        seed=args.seed,
        train_test_split_ratio=DISCOVERY_EVAL_SPLIT_PERCENT,
    )
    _, __, eval_vl_prompts = load_dataset(
        model=model,
        processor=processor,
        task_name=args.task_name,
        model_name=args.model_name,
        language_only=False,
        seed=args.seed,
        train_test_split_ratio=DISCOVERY_EVAL_SPLIT_PERCENT,
    )
    return eval_l_prompts, eval_vl_prompts


def get_random_sub_circuit(model, sub_circuit, possible_positions):
    """
    Given a sub-circuit, return a random sub-circuit of the same size.
    """
    random_sub_circuit = []
    for c in sub_circuit:
        random_sub_circuit.append(
            Component(
                c.hook_name,
                layer=random.randint(0, model.cfg.n_layers - 1),
                head=(
                    random.randint(0, model.cfg.n_heads - 1)
                    if c.head_idx is not None
                    else None
                ),
                position=random.choice(possible_positions),
                neurons=(
                    random.sample(
                        list(range(0, model.cfg.d_mlp)), k=len(c.neuron_indices)
                    )
                    if c.neuron_indices is not None
                    else None
                ),
            )
        )
    return random_sub_circuit


def analyze_cross_modality_DQL_interchange_faithfulness(
    model, args, eval_l_prompts, eval_vl_prompts, pos_mapping
):
    """
    Given that the L/V circuits can be split to D_L -> Q_L -> L_L and D_V -> Q_V -> L_V,
    we anaylze the accuracy of the model when replacing the Q_L with Q_V (and vice versa) or L_L with L_V (and vice versa),
    to give a sense of the

    Calculating faithfulness of metric=acc does this because the clean acc is 1.0 and the bad acc is 0.0 (so normalization doesn't affect).

    NOTE: In the paper, the last section is named "generation". Here we name it last and mark it with "L".
          This was left for compatability with existing result files.
    """
    discovery_metric = "LD"

    # Load L and VL node scores
    logging.info(f"Loading L and VL scores ({discovery_metric=})")
    l_scores, vl_scores = load_l_vl_scores(
        args.task_name, args.model_name, discovery_metric
    )
    l_seq_len = l_scores[f"blocks.0.attn.hook_z"].shape[0]
    vl_seq_len = vl_scores[f"blocks.0.attn.hook_z"].shape[0]
    limited_labels = get_limited_labels_for_task(args.task_name, model)

    MANDATORY_KEYS = [
        # High Baseline, all same modality
        "DV_QV_LV",
        "DL_QL_LL",
        # Replace Q
        "DV_QL_LV",
        "DV_QR_LV",
        "DV_QV_LL",
        "DV_QV_LR",
        # Replace L
        "DL_QV_LL",
        "DL_QR_LL",
        "DL_QL_LV",
        "DL_QL_LR",
        # Replace D
        "DL_QV_LV",
        "DR_QV_LV",
        "DV_QL_LL",
        "DR_QL_LL",
    ]

    for eval_metric in METRICS:
        logging.info(f"Analyzing {eval_metric=}")
        interchange_results_path = f"./data/{args.task_name}/results/{args.model_name}/faithfulness_nodes_cross_interchanges_{eval_metric}.pt"
        if os.path.exists(interchange_results_path):
            results_dict = torch.load(interchange_results_path)
            if all([k in results_dict for k in MANDATORY_KEYS]):
                return
        else:
            results_dict = {}

        # Get circuits and split them to D, Q, Last
        n_l_heads = int(
            args.l_circuit_percentage
            * model.cfg.n_heads
            * model.cfg.n_layers
            * l_seq_len
        )
        n_l_mlp_neurons = int(
            args.l_circuit_percentage * model.cfg.d_mlp * model.cfg.n_layers * l_seq_len
        )
        n_vl_heads = int(
            args.vl_circuit_percentage
            * model.cfg.n_heads
            * model.cfg.n_layers
            * vl_seq_len
        )
        n_vl_mlp_neurons = int(
            args.vl_circuit_percentage
            * model.cfg.d_mlp
            * model.cfg.n_layers
            * vl_seq_len
        )

        # Sort the found heads and neurons for both modalities
        top_l_heads, top_l_mlps_with_neurons, l_sorted_heads, l_sorted_mlp_neurons = (
            get_top_scoring_components(model, l_scores, n_l_heads, n_l_mlp_neurons)
        )
        l_circuit_comps = top_l_heads + top_l_mlps_with_neurons
        (
            top_vl_heads,
            top_vl_mlps_with_neurons,
            _,
            __,
        ) = get_top_scoring_components(model, vl_scores, n_vl_heads, n_vl_mlp_neurons)
        vl_circuit_comps = top_vl_heads + top_vl_mlps_with_neurons

        # Split components to D (data=image/text sequence), Q(query=question) and L(last) positions
        l_D_limits, vl_D_limits = get_text_sequence_positions(
            args.model_name, args.task_name
        ), get_image_positions(args.model_name, args.task_name)
        l_Q_limits, vl_Q_limits = [l_D_limits[-1], l_seq_len - 1], [
            vl_D_limits[-1],
            vl_seq_len - 1,
        ]
        l_D = [c for c in l_circuit_comps if l_D_limits[0] <= c.pos < l_D_limits[1]]
        l_Q = [c for c in l_circuit_comps if l_Q_limits[0] <= c.pos < l_Q_limits[1]]
        l_last = [c for c in l_circuit_comps if c.pos == l_seq_len - 1]
        vl_D = [c for c in vl_circuit_comps if vl_D_limits[0] <= c.pos < vl_D_limits[1]]
        vl_Q = [c for c in vl_circuit_comps if vl_Q_limits[0] <= c.pos < vl_Q_limits[1]]
        vl_last = [c for c in vl_circuit_comps if c.pos == vl_seq_len - 1]

        assert len(l_D + l_Q + l_last) == len(l_circuit_comps)
        assert len(vl_D + vl_Q + vl_last) == len(vl_circuit_comps)

        vl_random_Q = get_random_sub_circuit(
            model, vl_Q, range(vl_Q_limits[0], vl_Q_limits[1])
        )
        l_random_Q = get_random_sub_circuit(
            model, l_Q, range(l_Q_limits[0], l_Q_limits[1])
        )
        vl_random_last = get_random_sub_circuit(model, vl_last, [vl_seq_len - 1])
        l_random_last = get_random_sub_circuit(model, l_last, [l_seq_len - 1])

        l_random_D = get_random_sub_circuit(
            model, l_D, range(l_D_limits[0], l_D_limits[1])
        )
        vl_random_D = get_random_sub_circuit(
            model, vl_D, range(l_D_limits[0], l_D_limits[1])
        )

        if "DV_QV_LV" not in results_dict:
            logging.info("Calculating DV_QV_LV")
            results_dict["DV_QV_LV"] = circuit_faithfulness(
                model,
                vl_D + vl_Q + vl_last,
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DL_QL_LL" not in results_dict:
            logging.info("Calculating DL_QL_LL")
            results_dict["DL_QL_LL"] = circuit_faithfulness(
                model,
                l_D + l_Q + l_last,
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DV_QL_LV" not in results_dict:
            logging.info("Calculating DV_QL_LV")
            results_dict["DV_QL_LV"] = circuit_faithfulness(
                model,
                vl_D
                + list(convert_components_modality(l_Q, pos_mapping, l_to_vl=True))
                + vl_last,
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DV_QR_LV" not in results_dict:
            logging.info("Calculating DV_QR_LV")
            results_dict["DV_QR_LV"] = circuit_faithfulness(
                model,
                vl_D
                + list(
                    convert_components_modality(l_random_Q, pos_mapping, l_to_vl=True)
                )
                + vl_last,
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )
        if "DV_QV_LL" not in results_dict:
            logging.info("Calculating DV_QV_LL")
            results_dict["DV_QV_LL"] = circuit_faithfulness(
                model,
                vl_D
                + vl_Q
                + list(convert_components_modality(l_last, pos_mapping, l_to_vl=True)),
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )
        if "DV_QV_LR" not in results_dict:
            logging.info("Calculating DV_QV_LR")
            results_dict["DV_QV_LR"] = circuit_faithfulness(
                model,
                vl_D
                + vl_Q
                + list(
                    convert_components_modality(
                        l_random_last, pos_mapping, l_to_vl=True
                    )
                ),
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DL_QV_LL" not in results_dict:
            logging.info("Calculating DL_QV_LL")
            results_dict["DL_QV_LL"] = circuit_faithfulness(
                model,
                l_D
                + list(convert_components_modality(vl_Q, pos_mapping, l_to_vl=False))
                + l_last,
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DL_QR_LL" not in results_dict:
            logging.info("Calculating DL_QR_LL")
            results_dict["DL_QR_LL"] = circuit_faithfulness(
                model,
                l_D
                + list(
                    convert_components_modality(vl_random_Q, pos_mapping, l_to_vl=False)
                )
                + l_last,
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DL_QL_LV" not in results_dict:
            logging.info("Calculating DL_QL_LV")
            results_dict["DL_QL_LV"] = circuit_faithfulness(
                model,
                l_D
                + l_Q
                + list(
                    convert_components_modality(vl_last, pos_mapping, l_to_vl=False)
                ),
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DL_QL_LR" not in results_dict:
            logging.info("Calculating DL_QL_LR")
            results_dict["DL_QL_LR"] = circuit_faithfulness(
                model,
                l_D
                + l_Q
                + list(
                    convert_components_modality(
                        vl_random_last, pos_mapping, l_to_vl=False
                    )
                ),
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DL_QV_LV" not in results_dict:
            logging.info("Calculating DL_QV_LV")
            results_dict["DL_QV_LV"] = circuit_faithfulness(
                model,
                list(convert_components_modality(l_D, pos_mapping, l_to_vl=True))
                + vl_Q
                + vl_last,
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DR_QV_LV" not in results_dict:
            logging.info("Calculating DR_QV_LV")
            results_dict["DR_QV_LV"] = circuit_faithfulness(
                model,
                list(convert_components_modality(l_random_D, pos_mapping, l_to_vl=True))
                + vl_Q
                + vl_last,
                eval_vl_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DV_QL_LL" not in results_dict:
            logging.info("Calculating DV_QL_LL")
            results_dict["DV_QL_LL"] = circuit_faithfulness(
                model,
                list(convert_components_modality(vl_D, pos_mapping, l_to_vl=False))
                + l_Q
                + l_last,
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        if "DR_QL_LL" not in results_dict:
            logging.info("Calculating DR_QL_LL")
            results_dict["DR_QL_LL"] = circuit_faithfulness(
                model,
                list(
                    convert_components_modality(vl_random_D, pos_mapping, l_to_vl=False)
                )
                + l_Q
                + l_last,
                eval_l_prompts,
                metric=eval_metric,
                limited_labels=limited_labels,
                batch_size=1,
                verbose=False,
            )

        torch.save(results_dict, interchange_results_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Name of the model to be loaded")
    parser.add_argument("--model_path", type=str, help="Path to the model to be loaded")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument(
        "--task_name",
        type=str,
        choices=SUPPORTED_TASKS,
        help="Name of the task to be localized",
    )
    parser.add_argument(
        "--l_circuit_percentage",
        type=float,
        help="For which percentage of components in the L circuit to analyze cross-modality functionallity",
    )
    parser.add_argument(
        "--vl_circuit_percentage",
        type=float,
        help="For which percentage of components in the VL circuit to analyze cross-modality functionallity",
    )
    args = parser.parse_args()
    return args


def main():
    logging.info("Running script_nodes_intersection_cross_modality_analysis.py")
    args = parse_args()

    set_deterministic(args.seed)

    # Load model
    logging.info(f"Loading model {args.model_name} from {args.model_path}")
    model, processor = load_model(
        args.model_name,
        args.model_path,
        device,
        use_tlens_wrapper=True,
        extra_hooks=True,
        torch_dtype=torch.float32,
    )
    logging.info("Model loaded")

    # Load L and VL datasets
    logging.info(f"Loading dataset for task {args.task_name}")
    eval_l_prompts, eval_vl_prompts = load_l_vl_datasets(model, processor, args)

    # Analyze cross-modality faithfulness
    logging.info("Loading L-VL position mappings and verifying")
    pos_mapping = POS_MAPPINGS[f"{args.model_name[:4]}_{args.task_name}"]
    pos_mapping.assert_full_mapping(eval_l_prompts[0], eval_vl_prompts[0], model)

    logging.info("Analyzing cross-modality DQL interchange faithfulness")
    analyze_cross_modality_DQL_interchange_faithfulness(
        model, args, eval_l_prompts, eval_vl_prompts, pos_mapping
    )

    logging.info("Analysis complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
