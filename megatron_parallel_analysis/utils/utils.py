import os
import shutil
import sys
import multiprocessing as mp
from datetime import datetime

from hta.configs.config import logger


def prepare_directory(directory_path, force_clear=False):
    """
    Prepares a directory for storing files. If the directory exists, it will be emptied; if it does not exist, it will be created.
    
    Parameters:
    - directory_path: The path to the target directory.
    """
    logger.info(f'prepare directory for {directory_path}, force_clear={force_clear}')
    if os.path.exists(directory_path):
        if force_clear:
            # Empty the target directory
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create the target directory
        os.makedirs(directory_path)
    
    #time.sleep(6)

def find_and_copy_files(source_directory, target_directory, suffix_list):
    """
    Finds files in the specified directory whose names end with numbers from the given list, and copies them to a new directory.
    """
    # Prepare the target directory
    prepare_directory(target_directory)

    # Iterate through the source directory and copy matching files
    for filename in os.listdir(source_directory):
        name, ext = os.path.splitext(filename)
        for suffix in suffix_list:
            if name.endswith(f'_rank{suffix}') or name.startswith(f'worker{suffix}.'):
                # Filename matches, copy the file
                source_path = os.path.join(source_directory, filename)
                target_path = os.path.join(target_directory, filename)
                shutil.copy2(source_path, target_path)

def find_and_create_symlinks(source_directory, target_directory, suffix_list):
    """
    Finds files in the specified directory whose names end with numbers from the given list, and creates symbolic links for them in a new directory.
    """
    # Prepare the target directory
    prepare_directory(target_directory, force_clear=False)

    # Iterate through the source directory and create symbolic links for matching files
    for filename in os.listdir(source_directory):
        name, ext = os.path.splitext(filename)
        for suffix in suffix_list:
            # Todo: check the format of trace filename
            if name.endswith(f'_rank{suffix}') or name.startswith(f'rank{suffix}.'):
                source_path = os.path.join(source_directory, filename)
                target_path = os.path.join(target_directory, filename)
                
                if not os.path.exists(target_path):
                    os.symlink(source_path, target_path)
                else:
                    logger.info(f'{target_path} exists')

def partition_files_across_directories(source_directory, target_directory, groups_list):
    """
    Partitions files across directories based on a list of groups, creating symbolic links for each group in separate subdirectories.
    """
    for i, group in enumerate(groups_list):
        sub_dir = target_directory + f'_{i}'
        find_and_create_symlinks(source_directory, sub_dir, group)

def add_rank_to_filename(filepath, rank):
    # Split the file path into root and extension
    root, ext = os.path.splitext(filepath)
    
    # Find the first extension to correctly insert the rank
    base_name = os.path.basename(root)
    dir_name = os.path.dirname(root)
    
    # Split the base name by the first dot, if it exists
    if '.' in base_name:
        parts = base_name.split('.', 1)
        new_base_name = f"{parts[0]}_rank{rank}.{parts[1]}"
    else:
        new_base_name = f"{base_name}_rank{rank}"
    
    # Combine directory, new base name, and extension
    if dir_name:
        new_filename = os.path.join(dir_name, new_base_name + ext)
    else:
        new_filename = new_base_name + ext
    
    return new_filename

def apply_function_for_parallel(function, inputs=None, use_multiprocessing: bool = True):
    if inputs is None:
        inputs = [None] * mp.cpu_count()  # Default to the number of CPU cores if no inputs are provided

    if not use_multiprocessing:
        total_results = []
        for input in inputs:
            if input is not None:
                if isinstance(input, (list, tuple)):
                    result = function(*input)
                else:
                    result = function(input)
            else:
                result = function()
            total_results.append(result)
        logger.debug(f"Finished applying func {function.__name__} using 1 process.")
        return total_results
    
    num_procs = min(mp.cpu_count(), len(inputs))
    with mp.get_context("fork").Pool(num_procs) as pool:
        if inputs[0] is not None:
            if isinstance(inputs[0], (list, tuple)):
                results = pool.starmap(function, inputs)
            else:
                results = pool.map(function, inputs)
        else:
            results = pool.map(lambda _: function(), inputs)
        pool.close()
        pool.join()
    logger.debug(f"Finished parallel applying func {function.__name__} using {num_procs} processes.")
    
    return results

def worker(instance, func_name, args, kwargs):
    func = getattr(instance, func_name)
    if args and kwargs:
        result = func(*args, **kwargs)
    elif args:
        result = func(*args)
    elif kwargs:
        result = func(**kwargs)
    else:
        result = func()
    return instance, result

def apply_class_function_for_parallel(instances, func_name, inputs=None, use_multiprocessing=True):
    if inputs is None:
        inputs = [((), {})] * len(instances)  # Default to an empty args and kwargs for each instance if no inputs are provided

    if not use_multiprocessing:
        total_results = []
        for instance, (args, kwargs) in zip(instances, inputs):
            instance, result = worker(instance, func_name, args, kwargs)
            total_results.append(result)
        logger.debug(f"Finished applying func {func_name} using 1 process.")
        return total_results

    num_procs = min(mp.cpu_count(), len(instances))
    with mp.get_context("fork").Pool(num_procs) as pool:
        tasks = [(instance, func_name, args, kwargs) for instance, (args, kwargs) in zip(instances, inputs)]
        results = pool.starmap(worker, tasks)
        pool.close()
        pool.join()
    logger.debug(f"Finished parallel applying func {func_name} using {num_procs} processes.")

    # Update instances with modified values and collect results
    final_results = []
    for i in range(len(instances)):
        instances[i], result = results[i]
        final_results.append(result)

    return final_results

class LogToFile:
    """
    A context manager for redirecting stdout to a file.

    This class provides a mechanism for capturing the standard output
    to a specified file. It's useful for logging purposes, where the output
    of a block of code needs to be saved. If the file has not been written to
    before, it opens in write mode and logs the current timestamp. Otherwise,
    it appends to the existing file.

    Attributes:
        _has_cleared_files (dict):
            Tracks whether files have been cleared to avoid duplicate headers.
        filepath (str or None): Path to the log file. If None, stdout is not redirected.
        original_stdout (io.TextIOWrapper): Reference to the original stdout.

    """
    _has_cleared_files = {}  # Used to track whether each file has been cleared

    def __init__(self, filepath=None):
        """
        Initializes the context manager with the path to the log file.

        Args:
            filepath (str, optional): The path to the file where stdout will be redirected.
        """
        self.filepath = filepath
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        The stdout is redirected to the specified file. If the file has not been
        written to before, it is cleared and initialized with a timestamp.

        Returns:
            LogToFile: The runtime context object.
        """
        if self.filepath:
            # Check if this file has already been cleared
            if self.filepath not in self._has_cleared_files:
                # If not, open in "w" mode to clear it and mark as cleared
                self.file = open(self.filepath, "w", encoding='utf-8')
                self.file.write(datetime.now().strftime("%Y-%m-%d, %H:%M:%S") + "\n")
                self._has_cleared_files[self.filepath] = True
            else:
                # If already cleared, open in "a" mode to append content
                self.file = open(self.filepath, "a", encoding='utf-8')
            sys.stdout = self.file
            sys.stderr = sys.stdout 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context and restores the original stdout.

        The log file is closed if it was opened, and stdout is restored
        to its original state.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Traceback object.
        """
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.filepath:
            self.file.close()
