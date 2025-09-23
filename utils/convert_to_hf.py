#! /usr/bin/env python3
########################################################################################
#
# Script to convert my custom checkpoint to huggingface format.

# Author(s): Nik Vaessen
########################################################################################

import pathlib

import click
import torch

from safetensors.torch import save_file

########################################################################################
# main script

def find_layer_num(key: str):
    for sub in key.split("."):
        try:
            return int(sub)
        except ValueError:
            continue

    return None

def convert_to_huggingface(network):
    prefix = "wav2vec2."
    out_ckpt = {}

    for k, v in network.items():
        layer = find_layer_num(k)

        is_weight = k.split(".")[-1] == "weight"
        is_bias = k.split(".")[-1] == "bias"
        is_numbered = layer is not None
        has_prefix = True

        if is_weight and not is_bias:
            kind = "weight"
        elif is_bias and not is_weight:
            kind = "bias"
        else:
            kind = None

        print(f"currently looking at {k} with {v.shape=} {is_numbered=} {is_weight=} {is_bias=} {layer=} {kind=}")

        new_k = None

        if "masking_vector" in k:
            new_k = "masked_spec_embed"

        elif "conv_network" in k:
            assert is_numbered

            if "norm" in k:
                new_k = f"feature_extractor.conv_layers.{layer}.layer_norm.{kind}"
            else:
                new_k = f"feature_extractor.conv_layers.{layer}.conv.{kind}"

        elif f"project_speech_feature_norm.{kind}" in k:
            new_k = f"feature_projection.layer_norm.{kind}"

        elif "project_speech_feature." in k:
            new_k = f"feature_projection.projection.{kind}"

        elif "rel_pos_layer" in k:
            if "conv" in k and is_bias:
                new_k = f"encoder.pos_conv_embed.conv.bias"
            elif "original0" in k:
                new_k = "encoder.pos_conv_embed.conv.weight_g"
            elif "original1" in k:
                new_k = "encoder.pos_conv_embed.conv.weight_v"
            elif "norm" in k:
                new_k = f"encoder.layer_norm.{kind}"

        elif "transformer_network" in k:
            if "attention" in k:
                # special case because we model it as 1 tensor instead of 3
                if "attention_projection" in k:
                    tensor = network[k]
                    num_dim = tensor.shape[0] // 3
                    assert tensor.shape[0] % 3 == 0

                    if kind == "weight":
                        k_tensor = tensor[num_dim * 0 : num_dim * 1, :].cpu().clone()
                        q_tensor = tensor[num_dim * 1 : num_dim * 2, :].cpu().clone()
                        v_tensor = tensor[num_dim * 2 : num_dim * 3, :].cpu().clone()

                        out_ckpt[
                            f"{prefix}encoder.layers.{layer}.attention.k_proj.weight"
                        ] = k_tensor
                        out_ckpt[
                            f"{prefix}encoder.layers.{layer}.attention.v_proj.weight"
                        ] = v_tensor
                        out_ckpt[
                            f"{prefix}encoder.layers.{layer}.attention.q_proj.weight"
                        ] = q_tensor
                    else:
                        k_tensor = tensor[num_dim * 0 : num_dim * 1].cpu().clone()
                        q_tensor = tensor[num_dim * 1 : num_dim * 2].cpu().clone()
                        v_tensor = tensor[num_dim * 2 : num_dim * 3].cpu().clone()
                        out_ckpt[
                            f"{prefix}encoder.layers.{layer}.attention.k_proj.bias"
                        ] = k_tensor
                        out_ckpt[
                            f"{prefix}encoder.layers.{layer}.attention.v_proj.bias"
                        ] = v_tensor
                        out_ckpt[
                            f"{prefix}encoder.layers.{layer}.attention.q_proj.bias"
                        ] = q_tensor

                    continue
                else:
                    # out_proj
                    new_k = f"encoder.layers.{layer}.attention.out_proj.{kind}"

            elif "norm_att" in k:
                new_k = f"encoder.layers.{layer}.layer_norm.{kind}"
            elif "fc1" in k:
                new_k = f"encoder.layers.{layer}.feed_forward.intermediate_dense.{kind}"
            elif "fc2" in k:
                new_k = f"encoder.layers.{layer}.feed_forward.output_dense.{kind}"
            elif "norm_ffn" in k:
                new_k = f"encoder.layers.{layer}.final_layer_norm.{kind}"

        # self-supervised params
        if "quantization_layer" in k:
            has_prefix = False
            if "classification_layer" in k:
                new_k = f"quantizer.weight_proj.{kind}"
            elif "quantization_choices" in k:
                new_k = "quantizer.codevectors"
                v = v[None, :, :]
            elif "temp" in k:
                continue

        if "project_quantized_feature" in k:
            has_prefix = False
            new_k = f"project_q.{kind}"

        if "project_context_feature" in k:
            has_prefix = False
            new_k = f"project_hid.{kind}"

        if "classifier" in k:
            has_prefix = False
            new_k = f"lm_head.{kind}"

        if new_k is None:
            raise ValueError(f"unhandled key {k}")

        new_k = f"{prefix if has_prefix else ''}{new_k}"
        print(f"\trenaming to {new_k}")
        out_ckpt[new_k] = v.cpu()

    return out_ckpt


@click.command()
@click.argument("ckpt", type=pathlib.Path)
@click.option("--out", type=pathlib.Path, required=False, default=None)
def main(ckpt: pathlib.Path, out: pathlib.Path):
    fn = "model.safetensors"
    if out is None:
        if ckpt.name == fn:
            raise ValueError(f"will not overwrite {ckpt=}")
        out = ckpt.parent / fn

    print("converting")
    print(f"ckpt={str(ckpt)}")
    print(f"out={str(out)}")

    ckpt = torch.load(ckpt, map_location='cpu')

    network = ckpt["state_dict"]
    state_dict = convert_to_huggingface(network)

    out.parent.mkdir(exist_ok=True, parents=True)
    save_file(state_dict, str(out), {"format": "pt"})

if __name__ == "__main__":
    main()

