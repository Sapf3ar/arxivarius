def new_function(param1: int, param2: str) -> str:
    """
    A new function that combines an integer and a string.

    Parameters:
    param1 (int): The integer to combine.
    param2 (str): The string to combine.

    Returns:
    str: A combined result of the integer and string.
    """
    try:
        result = f'{param1} - {param2}'
        return result
    except Exception as e:
        print(f'An error occurred: {e}')
        return ''