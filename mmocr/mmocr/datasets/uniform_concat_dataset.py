# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import pandas as pd
from mmdet.datasets import DATASETS, ConcatDataset, build_dataset


@DATASETS.register_module()
class UniformConcatDataset(ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the results
            separately if it is used as validation dataset.
            Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True, pipeline=None, **kwargs):
        from_cfg = all(isinstance(x, dict) for x in datasets)
        if pipeline is not None:
            assert from_cfg, 'datasets should be config dicts'
            for dataset in datasets:
                if dataset['pipeline'] is None:
                    dataset['pipeline'] = copy.deepcopy(pipeline)
        datasets = [build_dataset(c, kwargs) for c in datasets]
        super().__init__(datasets, separate_eval)

    def format_results(self, results, filenames, output_filename='sub_mmocr.csv', **kwargs):
        """Placeholder to format result to dataset-specific output."""
#         print(len(results), len(filenames))
        assert len(results) == len(filenames)
        print(results)
        res_full = []
        res = [['id', 'text']]
        for f, r in zip(filenames, results):
            print(r['text'])
            res.append([os.path.splitext(f)[0], r['text']])
            res_full.append([os.path.splitext(f)[0], r['text'], min(r['score']), r['score']])

        res = [','.join(r) for r in res]
#         print(res)
        sub_text = '\r\n'.join(res)
        sub_text.rstrip()
        with open(output_filename, 'w') as f:
            f.write(sub_text)
            
        res_full = pd.DataFrame(res_full, columns=['id', 'text', 'min_score', 'score'])
        print(res_full)
        output_prefix = os.path.splitext(output_filename)[0]
        res_full.to_csv(f"{output_prefix}_score.csv", index=False)
