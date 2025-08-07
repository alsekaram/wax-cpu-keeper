import logging
import colorlog


def configure_color_logging(level=logging.WARNING):
    """
    Настраивает цветное логирование с добавлением кастомного уровня MINE.
    """
    # Отключаем нежелательные системные логи
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Создаем цветной форматер с использованием формата
    formatter = colorlog.ColoredFormatter(
        "[%(asctime)s.%(msecs)03d]%(module)5s:%(lineno)-3d%(log_color)s%(levelname)7s%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    # Создаем обработчик для вывода логов в консоль
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Получаем корневой логгер и назначаем ему обработчик
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)
