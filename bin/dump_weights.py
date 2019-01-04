

import argparse

from bilm_model.training import dump_weights as dw


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files', default='../try4/')
    parser.add_argument('--outfile', help='Output hdf5 file with weights', default='../try4/weights.hdf5')

    args = parser.parse_args()
    dw(args.save_dir, args.outfile)

"""
    nohup python -u  /home/zhlin/bilm-tf/bin/dump_weights.py  \
    --save_dir /home/zhlin/bilm-tf/try  \
    --outfile /home/zhlin/bilm-tf/try/weights.hdf5 >outfile.txt 2>&1 &
"""