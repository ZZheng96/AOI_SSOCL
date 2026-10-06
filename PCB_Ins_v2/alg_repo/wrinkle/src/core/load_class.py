import logging

logger = logging.getLogger("system")
def import_class(class_path: str):
    """
    根据类路径动态导入类

    Args:
        class_path: 类的完整路径，如 'algorithms.ocr.ocr_detect_alg.OCRDetectionAlgorithm'

    Returns:
        导入的类，失败返回 None
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    except Exception as e:
        logger.error(f"导入类失败: {class_path}, 错误: {e}")
        return None