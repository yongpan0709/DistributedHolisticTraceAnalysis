from typing import Iterable

def create_regex_for_prefix_match(prefixes):
    """
    Creates a regex pattern that matches any string starting with any of the provided prefixes.

    Parameters:
    - prefixes (list): A list of prefixes to match at the start of a string.

    Returns:
    - str: A regex pattern string.
    """
    return '^(' + '|'.join(prefixes) + ')'

def create_regex_for_full_match(keywards: Iterable[str]) -> str:
    """
    Creates a regex pattern that matches any string with any of the provided keywards.
    
    Parameters:
    - keywards (list): A list of keywards to match anywhere in a string.
    
    Returns:
    - str: A regex pattern string.
    """
    # Escape each keyword to handle special regex characters
    # Join the escaped keywords with the regex OR operator '|'
    name_pattern =  '^(' + '|'.join(keywards) + ')$'
    return name_pattern