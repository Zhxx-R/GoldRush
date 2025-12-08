import logging
import os
from datetime import datetime

LOG_LEVEL = logging.DEBUG
LOG_BASE_PATH = 'log'  

def set_logger(name):
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    os.makedirs(LOG_BASE_PATH, exist_ok=True)
    # 创建一个FileHandler，用于写入日志文件, 文件名包含日期和小时
    log_time = datetime.now().strftime('%Y%m%d_%H')
    file_handler = logging.FileHandler(f'{LOG_BASE_PATH}/{name}_{log_time}.log', mode='w')
    file_handler.setLevel(LOG_LEVEL)

    # 创建一个Formatter，用于设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将FileHandler添加到logger中
    logger.addHandler(file_handler)
    return logger

logger = set_logger('config')

# 0: state, 1: action, 2: reward, 3: next_state, 4: done, 5: value, 6: logit, 7: adv, 8: return
USE_GPU = False
USE_GPU_GLOBAL = True
INPUT_DIM = 868
EMBEDDING_DIM = 128

NUM_GPU = 1
NUM_META_AGENT = 32

GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2

MAX_ROUNDS = 300

MINIMUM_BUFFER_SIZE = 5000

REPLAY_SIZE = 10000
BATCH_SIZE = 2048
LR = 2.5e-4

LOAD_MODEL = True #

extra_info = 'debug'
FOLDER_NAME = 'result_{}'.format(
    extra_info
)
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'

SUMMARY_WINDOW = 5


logger.info(f"Config loaded - INPUT_DIM: {INPUT_DIM}, EMBEDDING_DIM: {EMBEDDING_DIM}")
logger.info(f"Training settings - BATCH_SIZE: {BATCH_SIZE}, LR: {LR}, MAX_ROUNDS: {MAX_ROUNDS}")
logger.info(f"Paths - model_path: {model_path}, train_path: {train_path}")
