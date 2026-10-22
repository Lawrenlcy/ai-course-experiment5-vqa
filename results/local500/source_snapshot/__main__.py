import argparse
import json

def main():
    parser=argparse.ArgumentParser(description='Experiment 5: local visual question answering')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--n',type=int,default=500);p.add_argument('--seed',type=int,default=42);p.add_argument('--data',default='data')
    p=sub.add_parser('run');p.add_argument('--manifest',default='data/manifest_n500_seed42.json');p.add_argument('--out',default='results/local500');p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--settings',nargs='+',choices=['greedy','beam3','blank_image'])
    p=sub.add_parser('demo');p.add_argument('--port',type=int,default=7865)
    p=sub.add_parser('report');p.add_argument('--out',default='results/local500');p.add_argument('--data',default='data')
    sub.add_parser('doctor')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare
        print(prepare(args.data,args.n,args.seed))
    elif args.command=='run':
        from .experiment import run
        run(args.manifest,args.out,args.batch_size,args.device,args.settings)
    elif args.command=='demo':
        from .ui import launch
        launch(args.port)
    elif args.command=='report':
        from .report import build_report
        build_report(args.out,args.data)
    else:
        from .common import environment
        print(json.dumps(environment(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
