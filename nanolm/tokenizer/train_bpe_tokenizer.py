import os
import regex
from typing import BinaryIO
from collections import Counter, defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(special_tokens, list), "special_tokens must be a list"
    assert all(isinstance(t, bytes) for t in special_tokens), "All special tokens must be bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            escape_special_tokens = [regex.escape(t) for t in special_tokens]
            pattern = b"|".join(escape_special_tokens)
            found = regex.search(pattern, mini_chunk)
            if found:
                chunk_boundaries[bi] = initial_position + found.start()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _worker_pre_tokenize(file_path: str, start: int, end: int, special_tokens: list[bytes]) -> dict[str, int]:
    token_count = Counter()
    buffer_size = 10 * 1024 * 1024  # 内存安全阀值：每次只处理 10MB
    carry_over = b""  # 用来存放上次切剩的“尾巴”

    escape_special_tokens = [regex.escape(t.decode("utf-8", errors="ignore")) for t in special_tokens]
    special_pattern = "|".join(escape_special_tokens) if escape_special_tokens else None

    with open(file_path, "rb") as file:
        file.seek(start)
        bytes_read = 0
        total_to_read = end - start

        while bytes_read < total_to_read:
            # IMPORTANT: We cannot use a fixed mini chunk size to read, because we might accidentally split a UTF-8 character in the middle
            # which would cause decoding errors.
            # So instead, we read a buffer of bytes, then find the last safe cut point (like a space or newline) within that buffer to ensure we only decode complete characters.
            # 确保最后一次读取不会超过 end 边界
            current_read_size = min(buffer_size, total_to_read - bytes_read)
            chunk = file.read(current_read_size)
            bytes_read += current_read_size

            # 把上次剩下的尾巴和这次的新数据拼起来
            raw_data = carry_over + chunk

            # 如果这是这个 worker 的最后一块数据，直接全部处理
            if bytes_read >= total_to_read:
                safe_data = raw_data
                carry_over = b""
            else:
                # 寻找安全的切断点（寻找最后一个空格或换行符）
                # 这样既不会切断正常的英文单词，也绝不会把 UTF-8 字符从中间劈开
                last_safe_idx = max(raw_data.rfind(b' '), raw_data.rfind(b'\n'))
                
                if last_safe_idx != -1:
                    safe_data = raw_data[:last_safe_idx]
                    carry_over = raw_data[last_safe_idx:] # 留给下一轮
                else:
                    # 极端防御：如果 10MB 里连一个空格都没有，只能硬切退化
                    safe_data = raw_data
                    carry_over = b""

            # 现在可以绝对安全地解码了
            chunk_str = safe_data.decode("utf-8", errors="ignore")

            # 正常的正则分割和统计
            if special_pattern:
                split_chunk = regex.split(special_pattern, chunk_str)
            else:
                split_chunk = [chunk_str]

            for cur_chunk in split_chunk:
                for pre_token in regex.finditer(PAT, cur_chunk):
                    token_count[pre_token.group(0)] += 1

    return token_count

def parallel_pre_tokenize(file_path: str, worker_num: int, special_tokens: list[bytes]) -> dict[str, int]:
    if worker_num is None:
        worker_num = max(1, multiprocessing.cpu_count() - 1)
    
    chunk_boundaries = find_chunk_boundaries(
        open(file_path, "rb"),
        desired_num_chunks=worker_num,
        special_tokens=special_tokens
    )

    master_counter = Counter()

    with ProcessPoolExecutor(max_workers=worker_num) as executor:
        futures = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            futures.append(executor.submit(_worker_pre_tokenize, file_path, start, end, special_tokens))

        for future in as_completed(futures):
            try:
                worker_result = future.result()
                master_counter.update(worker_result)
            except Exception as e:
                print(f"Worker raised an exception: {e}")

    return dict(master_counter)

def get_pairs(splits: list[bytes]) -> list[tuple[bytes, bytes]]:
    """辅助函数：根据当前的拆分状态，获取所有相邻的 pair"""
    return [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]

def merge_tokens(pre_token_count: dict[str, int], vocab_size: int, special_tokens: list[bytes]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    # IMPORTANT: Remember to add special tokens to the vocab and make sure they are not merged during the merging process. 
    # We can achieve this by treating them as indivisible units in the splitting step of encoding, and also by skipping them in the merging step.
    for st in special_tokens:
        vocab[len(vocab)] = st
    
    merges = []

    # 1. 核心状态：记录每个词当前的切分块
    word_splits = {
        word: [bytes([b]) for b in word.encode('utf-8')]
        for word in pre_token_count.keys()
    }

    pair_counts = Counter()
    pair_to_words = defaultdict(set)

    # 初始化账本
    for word, count in pre_token_count.items():
        splits = word_splits[word]
        for pair in get_pairs(splits):
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    # 2. 开始 Merge 循环
    for _ in range(vocab_size - len(vocab)):
        if not pair_counts:
            break
            
        # 获取最高频的 pair
        # most_common_pair, _ = pair_counts.most_common(1)[0]
        # new_token = most_common_pair[0] + most_common_pair[1]
        best_pair_item = max(pair_counts.items(), key=lambda x: (x[1], x[0]))  # 先按频次排序，频次相同按 pair 本身排序保证稳定性
        most_common_pair, _ = best_pair_item
        new_token = most_common_pair[0] + most_common_pair[1]
        
        # 记录到词表和 merges 历史中
        vocab[len(vocab)] = new_token
        merges.append(most_common_pair)

        # 拿到所有包含这个 pair 的词（转为 list 防止迭代时 set 改变）
        words_to_process = list(pair_to_words[most_common_pair])

        for word in words_to_process:
            splits = word_splits[word]
            count = pre_token_count[word]

            # === 步骤 A：账本“退款” ===
            # 把这个词产生的所有旧 pair 从大账本里扣除
            for pair in get_pairs(splits):
                pair_counts[pair] -= count
                pair_to_words[pair].discard(word)
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            # === 步骤 B：物理合并（状态重建） ===
            # 完全不修改原 splits，而是新建一个数组
            new_splits = []
            i = 0
            while i < len(splits):
                # 匹配到了！拼合后放进新数组，指针跳 2 步
                if i < len(splits) - 1 and splits[i] == most_common_pair[0] and splits[i + 1] == most_common_pair[1]:
                    new_splits.append(new_token)
                    i += 2
                # 没匹配到，原样放进新数组，指针跳 1 步
                else:
                    new_splits.append(splits[i])
                    i += 1
            
            # 更新该词的最新切分状态
            word_splits[word] = new_splits

            # === 步骤 C：账本“重新入账” ===
            # 把这个词产生的新 pair 加回大账本
            for pair in get_pairs(new_splits):
                pair_counts[pair] += count
                pair_to_words[pair].add(word)

        # 彻底清理掉已经被合并的这个 pair
        if most_common_pair in pair_counts:
            del pair_counts[most_common_pair]

    return vocab, merges

def train_bpe_tokenizer(file_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bytes_special_tokens = [t.encode("utf-8") for t in special_tokens]
    pre_token_count = parallel_pre_tokenize(file_path, 10, bytes_special_tokens)
    return merge_tokens(pre_token_count, vocab_size, bytes_special_tokens)