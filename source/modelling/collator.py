from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
import torch
from .utils import load_dict_from_file


@dataclass
class DataCollatorForTokenRegression():
    """From https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/data_collator.py#L266
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: tuple[np.float32, np.float32] = None
    return_tensors: str = "pt"
    sign: bool = False

    def __call__(self, features):
        
        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)
        
        if self.label_pad_token_id is None:
            label_pad_token_id = tuple([np.nan, np.nan])
        else:
            label_pad_token_id = self.label_pad_token_id
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        
        

        no_labels_features = [{k: v for k, v in feature.items() if k not in (
            label_name,
            'file_path_bpp',
            'indices_bpp',
            'values_bpp',
            'file_path_mfe_distance',
            'structure_eternafold',
            'sn_2a3_map',
            'sn_dms_map',
            'stn_nts_2A3_MaP',
            'stn_nts_DMS_MaP',
            'capr_structure_probs',
            'len',
        )} for feature in features]
        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side
        
        bpp_matrices = [load_dict_from_file(feature['file_path_bpp']) for feature in features]
        batch['attention_injection'] = [
            ((8, 16), 0.5, torch.stack([
                torch.sparse_coo_tensor(
                    torch.tensor(d['indices_bpp']),
                    torch.tensor(d['values_bpp']),
                    size=(sequence_length, sequence_length)
                ).coalesce().to_dense() for d in bpp_matrices
            ])),
        ]
        batch['capr_structure_probs'] = torch.tensor([
            feature['capr_structure_probs'] + [[0] * 6] * (sequence_length - len(feature['capr_structure_probs'])) for feature in features
        ], dtype=torch.float32)
        if self.sign:
            base_distance_matrix = torch.LongTensor(np.fromfunction(lambda i, j: (i - j), (sequence_length, sequence_length)))
        else:
            base_distance_matrix = torch.LongTensor(np.fromfunction(lambda i, j: abs(i - j), (sequence_length, sequence_length)))
        
        distance_matrices = [load_dict_from_file(feature['file_path_mfe_distance']) for feature in features]
        batch['input_ids_mfe_distance'] = torch.LongTensor([
            to_list(d['input_ids_mfe_distance']) + [self.tokenizer.pad_token_id] * (sequence_length - len(d['input_ids_mfe_distance'])) for d in distance_matrices
        ])
        
        batch['structure_eternafold'] = torch.LongTensor([
            to_list(feature['structure_eternafold']) + [self.tokenizer.pad_token_id] * (sequence_length - len(feature['structure_eternafold'])) for feature in features
        ])
        distance_matrices = [
            torch.sparse_coo_tensor(
                torch.torch.LongTensor(d['indices_mfe_distance']),
                torch.torch.LongTensor(d['values_mfe_distance']),
                size=(sequence_length, sequence_length),
            ) for d in distance_matrices
        ]
        for idx, dm in enumerate(distance_matrices):
            m = base_distance_matrix.clone()
            if self.sign:
                dm = -dm + dm.t()
            else:
                dm = dm + dm.t()
            dm = dm.coalesce()
            dm_dense = dm.to_dense()
            m[dm.indices()[0], dm.indices()[1]] = dm_dense[dm.indices()[0], dm.indices()[1]]
            distance_matrices[idx] = m
        
        batch['distance_batch'] = torch.stack(distance_matrices)
        
        if labels is None:
            return batch

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]
            raise NotImplementedError

        try:
            batch[label_name] = torch.tensor(batch[label_name], dtype=torch.float32)
        except Exception as e:
            print(label_name)
            print(batch[label_name])
            raise e
            
        batch['sn_nts'] = torch.tensor([[feature['sn_2a3_map'], feature['sn_dms_map']] for feature in features], dtype=torch.float32).unsqueeze(1).repeat_interleave(repeats=batch['labels'].shape[1], dim=1)
        batch['sn_nts'][torch.isnan(batch[label_name])] = 0.0 # set nt weight to 0 for nan labels
        batch['sn_nts'] = torch.clip(batch['sn_nts'], min=0.0)
        batch['sn_nts'][batch['sn_nts'] != 0] = torch.log(batch['sn_nts'][batch['sn_nts'] != 0] + 1.105) / 2.0
        return batch