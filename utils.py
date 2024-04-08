import os
import logging
import datetime

def setup_logger(prefix='model_training', log_dir='logs', console_level=logging.ERROR):
    """
    初始化并配置日志器，返回一个已经配置好的logger实例。

    参数:
        prefix (str): 日志文件名的前缀。
        log_dir (str): 存放日志文件的目录，默认为'logs'。
        console_level (logging.LEVEL): 控制台日志级别，默认只显示错误及以上级别的信息。

    返回:
        logger: 配置好的logging.Logger实例。
    """

    def create_log_filename():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'{prefix}_{timestamp}.log'

    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建自定义日志文件名
    log_file_name = create_log_filename()
    log_path = os.path.join(log_dir, log_file_name)

    # 创建一个logger
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入自定义日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建一个formatter，用于设置日志格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    # 添加FileHandler到logger
    logger.addHandler(file_handler)

    # 创建一个StreamHandler，用于将错误级别及以上的日志输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)

    # 添加StreamHandler到logger
    logger.addHandler(stream_handler)

    return logger