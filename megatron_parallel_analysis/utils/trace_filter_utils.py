from typing import Iterable
import re


def create_regex_for_prefix_match(prefixes):
    """
    Creates a regex pattern that matches any string starting with any of the provided prefixes.

    Parameters:
    - prefixes (list): A list of prefixes to match at the start of a string.

    Returns:
    - str: A regex pattern string.
    """
    return '^(' + '|'.join(re.escape(prefix) for prefix in prefixes) + ')'


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
    name_pattern =  '^(' + '|'.join(re.escape(keyword) for keyword in keywards) + ')$'
    return name_pattern