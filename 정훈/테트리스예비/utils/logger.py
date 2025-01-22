import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("TetrisLogger")

logger = setup_logger()

if __name__ == "__main__":
    logger.info("로깅 테스트 중")
