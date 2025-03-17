def check_types(
    sigma_l: float, sigma_h: float, sigma_iou: float, t_min: int, t_miss_max: int
) -> None:
    """Raise ValueErrors if wrong types"""

    if not isinstance(sigma_l, (int, float)):
        raise ValueError("sigma_l has to be int or float")
    if not isinstance(sigma_h, (int, float)):
        raise ValueError("sigma_h has to be int or float")
    if not isinstance(sigma_iou, (int, float)):
        raise ValueError("sigma_iou has to be int or float")
    if not isinstance(t_min, int):
        raise ValueError("t_min has to be int")
    if not isinstance(t_miss_max, int):
        raise ValueError("t_miss_max has to be int")
