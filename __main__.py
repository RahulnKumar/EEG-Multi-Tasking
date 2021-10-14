import argparse
from src.evaluate import evaluate
from src.train_st import single_task_train
from src.train_cmt import conventional_multi_task_train
from src.train_psmt import private_shared_multi_task_train
from src.train_amt import adversarial_multi_task_train
from src.train_amt_v2 import adversarial_multi_task_train_v2

# ===== ARGUMENTS =====
parser = argparse.ArgumentParser()
parser.add_argument('--st', action="store_true", default=False, help='Single task training')
parser.add_argument('--cmt', action="store_true",  default=False, help='Conventional Multi-task training')
parser.add_argument('--psmt', action="store_true",  default=False, help='Private-Shared Multitask training')
parser.add_argument('--amt', action="store_true",  default=False, help='Adversarial Multitask training version 1')
parser.add_argument('--amt_v2', action="store_true",  default=False, help='Adversarial Multitask training version 2')


args = parser.parse_args()

st_flag = args.st
cmt_flag = args.cmt
psmt_flag = args.psmt
amt_flag = args.amt
amt_v2_flag = args.amt_v2

if st_flag:
    print('------------------------ SINGLE-TASK TRAINING STARTED------------------------')
    single_task_train()
    print('------------------------ SINGLE-TASK TRAINING ENDED ------------------------')


if cmt_flag:
    print('------------------------CONVENTIONAL MULTI-TASK TRAINING STARTED ------------------------')
    conventional_multi_task_train()
    print('------------------------CONVENTIONAL MULTI-TASK TRAINING ENDED --------------------------')

if psmt_flag:
    print('------------------------ PRIVATE-SHARED MULTI-TASK TRAINING STARTED------------------------')
    private_shared_multi_task_train()
    print('------------------------ PRIVATE-SHARED MULTI-TASK TRAINING ENDED --------------------------')

if amt_flag:
    print('------------------------ ADVERSARIAL MULTI-TASK TRAINING STARTED ------------------------')
    adversarial_multi_task_train()
    print('------------------------ ADVERSARIAL MULTI-TASK TRAINING ENDED  --------------------------')

if amt_v2_flag:
    print('------------------------ ADVERSARIAL MULTI-TASK version 2 TRAINING STARTED ------------------------')
    adversarial_multi_task_train_v2()
    print('------------------------ ADVERSARIAL MULTI-TASK version 2 TRAINING ENDED  --------------------------')