"""Train AdaDepth with a dataset configuration."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--initialize-from')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    for arg, key in [(args.data_path,'data_path'), (args.output_dir,'output_dir'),
                     (args.initialize_from,'initialize_from'), (args.device,'device')]:
        if arg is not None: config[key] = arg
    if args.dry_run:
        print(json.dumps(config, indent=2)); return
    if not config.get('data_path'):
        parser.error('--data-path is required')
    from adadepth.trainer import Trainer
    Trainer(config).train(args.max_steps)


if __name__ == '__main__':
    main()
