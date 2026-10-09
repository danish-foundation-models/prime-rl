"""Warm the shared search datasets/index and record source provenance without exposing answers."""
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer

from browsecomp_plus.corpus import ensure_cache, index_dir
from browsecomp_plus.taskset import BrowseCompPlusConfig, BrowseCompPlusTaskset


def main():
    output = Path(os.environ['SEARCH_PREP_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'datasets': {}}
    api = HfApi()
    for name, split in [('PolarSeeker/OpenSeeker-v1-Data', 'train'),
                        ('Tevatron/browsecomp-plus', 'test'),
                        ('Tevatron/browsecomp-plus-corpus', 'train')]:
        dataset = load_dataset(name, split=split)
        report['datasets'][name] = {'split': split, 'rows': len(dataset),
                                    'revision': api.dataset_info(name).sha,
                                    'fingerprint': dataset._fingerprint}
        print(name, len(dataset), flush=True)
    Tokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    ensure_cache()
    # Exercise the real loader after the index build, without printing benchmark text.
    tasks = list(BrowseCompPlusTaskset(BrowseCompPlusConfig(id='browsecomp-plus')).load())
    assert len(tasks) == 830
    report['browsecomp_tasks'] = len(tasks)
    report['index'] = str(index_dir())
    (output / 'data-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('SEARCH_DATA_READY', flush=True)


if __name__ == '__main__':
    main()
