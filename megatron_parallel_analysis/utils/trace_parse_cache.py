# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import tempfile
from typing import Dict, Optional

import pandas as pd

from hta.configs.config import logger


TRACE_PARSE_CACHE_SCHEMA_VERSION = 1
TRACE_PARSE_BEHAVIOR_VERSION = 1


def get_rank_parse_cache_path(cache_dir: str, rank_id: int) -> str:
    return os.path.join(cache_dir, f'rank{rank_id}.parse_cache.pkl')


def build_rank_parse_cache_metadata(
    rank_id: int,
    trace_file: str,
    bwd_annotation_str: str,
) -> Dict[str, object]:
    source_realpath = os.path.realpath(trace_file)
    stat_result = os.stat(source_realpath)
    return {
        'schema_version': TRACE_PARSE_CACHE_SCHEMA_VERSION,
        'parser_behavior_version': TRACE_PARSE_BEHAVIOR_VERSION,
        'rank_id': rank_id,
        'bwd_annotation_str': bwd_annotation_str,
        'source_realpath': source_realpath,
        'source_size': stat_result.st_size,
        'source_mtime_ns': stat_result.st_mtime_ns,
        'source_device': stat_result.st_dev,
        'source_inode': stat_result.st_ino,
    }


def load_rank_parse_cache(
    cache_path: str,
    expected_metadata: Optional[Dict[str, object]],
) -> Optional[pd.DataFrame]:
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'rb') as cache_file:
            payload = pickle.load(cache_file)
        if not isinstance(payload, dict):
            raise ValueError('cache payload is not a dictionary')
        if expected_metadata is not None and payload.get('metadata') != expected_metadata:
            logger.info(f'Ignoring stale trace parse cache {cache_path}')
            return None
        full_df = payload.get('full_df')
        if not isinstance(full_df, pd.DataFrame):
            raise ValueError('cache payload does not contain a DataFrame')
        return full_df
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, TypeError, ValueError) as error:
        logger.warning(f'Unable to load trace parse cache {cache_path}: {error}')
        return None


def write_rank_parse_cache(
    cache_path: str,
    metadata: Dict[str, object],
    full_df: pd.DataFrame,
) -> bool:
    cache_dir = os.path.dirname(cache_path)
    temp_path = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='wb',
            prefix=f'.{os.path.basename(cache_path)}.',
            suffix='.tmp',
            dir=cache_dir,
            delete=False,
        ) as cache_file:
            temp_path = cache_file.name
            pickle.dump(
                {'metadata': metadata, 'full_df': full_df},
                cache_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            cache_file.flush()
            os.fsync(cache_file.fileno())
        os.replace(temp_path, cache_path)
        return True
    except (OSError, pickle.PickleError, TypeError) as error:
        logger.warning(f'Unable to write trace parse cache {cache_path}: {error}')
        return False
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
