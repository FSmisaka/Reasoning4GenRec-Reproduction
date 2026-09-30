import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CATEGORY_PHRASES = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
    "Video_Games": "video games",
}


class TokenExtender:
    """
    迁移自 SIDReasoner sft_Qwen3.py: 从 index.json 收集全部 SID token
    (<a_N>/<b_N>/<c_N>), 去重排序后作为新词表条目。
    """

    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        if self.indices is None:
            self._load_data()
        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens


def category_phrase(category):
    return CATEGORY_PHRASES.get(category, "items")


def load_sidreasoner_tokenizer(base_model, sid_index_path):
    """
    加载 tokenizer 并追加 SID token(<a_N>/<b_N>/<c_N>),
    与原始实现的 TokenExtender 流程一致。若 checkpoint 的词表
    已含全部 SID token(Stage 2/3 的输入), 则为幂等操作。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    num_new_tokens = 0
    if sid_index_path and os.path.exists(sid_index_path):
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split(".")[0],
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            existing_vocab = set(tokenizer.get_vocab().keys())
            tokens_to_add = [tok for tok in new_tokens if tok not in existing_vocab]
            if tokens_to_add:
                tokenizer.add_tokens(tokens_to_add)
                num_new_tokens = len(tokens_to_add)
    return tokenizer, num_new_tokens


def load_sidreasoner_model(base_model, tokenizer, num_new_tokens=0, bf16=True):
    """
    加载 Qwen3 因果语言模型, 并按扩展后的词表 resize embedding,
    与原始实现 sft_Qwen3.py 的流程一致。返回 (model, num_new_tokens)。
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if bf16 else None,
    )
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model, num_new_tokens


def maybe_restrict_to_new_tokens(model, num_new_tokens):
    """
    对应原始实现的 train_new_token_embeddings_only=True 分支:
    用 peft.TrainableTokensConfig 只训练新增 SID token 的 embedding。
    """
    if num_new_tokens <= 0:
        return model
    from peft import TrainableTokensConfig, get_peft_model

    vocab_size = model.get_input_embeddings().weight.shape[0]
    new_token_indices = list(range(vocab_size - num_new_tokens, vocab_size))
    peft_config = TrainableTokensConfig(
        token_indices=new_token_indices,
        target_modules=["embed_tokens"],
        init_weights=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
