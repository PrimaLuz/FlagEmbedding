import collections
import faiss
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from FlagEmbedding import FlagModel, FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker
import os
from pprint import pprint
import json
import time


logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="/data/disk2/zhangzh/models/BAAI/bge-m3",
        metadata={"help": "The encoder name or path."},
    )
    fp16: bool = field(default=True, metadata={"help": "Use fp16 in inference?"})
    add_instruction: bool = field(
        default=False, metadata={"help": "Add query-side instruction?"}
    )

    max_query_length: int = field(default=32, metadata={"help": "Max query length."})
    max_passage_length: int = field(
        default=128, metadata={"help": "Max passage length."}
    )
    batch_size: int = field(default=2048, metadata={"help": "Inference batch size."})
    index_factory: str = field(
        default="Flat", metadata={"help": "Faiss index factory."}
    )
    k: int = field(default=20, metadata={"help": "How many neighbors to retrieve?"})

    save_embedding: bool = field(
        default=False, metadata={"help": "Save embeddings in memmap at save_dir?"}
    )
    load_embedding: bool = field(
        default=False, metadata={"help": "Load embeddings from save_dir?"}
    )
    save_path: str = field(
        default="embeddings.memmap", metadata={"help": "Path to save embeddings."}
    )
    eval_data: str = field(
        default="/data/disk2/zhangzh/project/V1129/embed_eval/data3_qc/eval_data.json",
        metadata={"help": "eval_data path"},
    )
    corpus: str = field(
        default="/data/disk2/zhangzh/project/V1129/embed_eval/data3_qc/corpus_wo_title.json",
        metadata={"help": "corpus path."},
    )

    rerank: bool = field(default=False, metadata={"help": "use reranker"})


def index(
    model: FlagModel,
    corpus: datasets.Dataset,
    batch_size: int = 256,
    max_length: int = 512,
    index_factory: str = "Flat",
    save_path: str = None,
    save_embedding: bool = False,
    load_embedding: bool = False,
):
    """
    1. Encode the entire corpus into dense embeddings;
    2. Create faiss index;
    3. Optionally save embeddings.
    """
    print("index开始！！！")
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(save_path, mode="r", dtype=dtype).reshape(-1, dim)

    else:
        ## 改编#######################
        # corpus_list = []
        # for item in corpus:
        #     corpus_list.append(item["content"])
        # corpus_embeddings = model.encode_corpus(corpus_list, batch_size=batch_size, max_length=max_length)
        ########################

        corpus_embeddings = model.encode_corpus(
            corpus["content"], batch_size=batch_size, max_length=max_length
        )
        # for v1129, corpus_embeddings.shape = (12380, 1024)
        dim = corpus_embeddings.shape[-1]

        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype,
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(
                    range(0, length, save_batch_size),
                    leave=False,
                    desc="Saving Embeddings",
                ):
                    j = min(i + save_batch_size, length)
                    memmap[i:j] = corpus_embeddings[i:j]
            else:
                memmap[:] = corpus_embeddings

    corpus_sizee = len(corpus_embeddings)
    print(f"corpus Length: {corpus_sizee}")

    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
    logger.info("faiss.index_cpu_to_all_gpu跳过！！！")
    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    logger.info("index.train开始！！！")
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    faiss_index.train(corpus_embeddings)
    logger.info("index.train完成开始add！！！")
    faiss_index.add(corpus_embeddings)
    logger.info("index.add完成！！！")
    return faiss_index


def search(
    model: FlagModel,
    queries: datasets,
    faiss_index: faiss.Index,
    k: int = 100,
    batch_size: int = 256,
    max_length: int = 512,
):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    print("search开始！！！！！")
    ##############改动###############
    # query_list = []
    # for item in queries:
    #     query_list.append(item["query"])

    # query_embeddings = model.encode_queries(query_list, batch_size=batch_size, max_length=max_length)
    ########################
    query_embeddings = model.encode_queries(
        queries["query"], batch_size=batch_size, max_length=max_length
    )
    query_size = len(query_embeddings)
    # print(f"query_size Length: {query_size}")
    print(f"query_size Length: {len(query_embeddings)}")
    print(f"corpus_size Length: {faiss_index.ntotal}")
    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        query_embedding = query_embeddings[i:j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    # print(f"测试结果:{all_scores[0][0:10]}")
    # for v1129 all_scores.shape = (2470,20) (query_size,topk)
    # for v1129 all_indices.shape = (2470,20) (query_size,topk)
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    print("search完成！！！！！")
    return all_scores, all_indices


def rerank(reranker, query_list, retrieval_results, flag):
    # reranker = FlagReranker(
    #     model_name_or_path="/data/disk2/zhangzh/project/FlagEmbedding-master/BAAI/bge-reranker-large",
    #     use_fp16=True,
    # )

    all_scores = []
    sorted_results_list = []
    # sorted_score_list = []
    for index, query in tqdm(
        enumerate(query_list), desc="Rerank Processing", total=len(query_list)
    ):
        # one_scores一个query与topk使用rerank的得分
        one_scores = []
        for i, result in enumerate(retrieval_results[index]):
            if flag == 1:
                score = reranker.compute_score([query, result], cutoff_layers=[28], normalize=True)
            else:
                score = reranker.compute_score([query, result], normalize=True)
            one_scores.append(score)
        data_list = [
            (ii, rresult, one_scores[ii])
            for ii, rresult in enumerate(retrieval_results[index])
        ]
        sorted_data_list = sorted(data_list, key=lambda x: x[2], reverse=True)
        sorted_result = [data[1] for data in sorted_data_list]
        # sorted_score = [data[2] for data in sorted_data_list]
        all_scores.append(one_scores)
        sorted_results_list.append(sorted_result)
        # sorted_score_list.append(sorted_score)
    # print(sorted_results_list[1])
    return sorted_results_list

def rerankv2(reranker, query_list, retrieval_results, flag):
    sorted_results_list = []

    for index, query in tqdm(
        enumerate(query_list), desc=f"{flag} Rerank Processing", total=len(query_list)
    ):
        # 批量计算得分
        if flag == "minicpm":
            scores = reranker.compute_score([(query, result) for result in retrieval_results[index]], cutoff_layers=[28], normalize=True)
        else:
            scores = reranker.compute_score([(query, result) for result in retrieval_results[index]], normalize=True)
        
        # 将得分和结果结合，并按得分排序
        sorted_data = sorted(zip(range(len(retrieval_results[index])), retrieval_results[index], scores), key=lambda x: x[2], reverse=True)
        
        # 提取排序后的结果
        sorted_result = [data[1] for data in sorted_data]
        sorted_results_list.append(sorted_result)

    return sorted_results_list


def evaluate(preds, labels, cutoffs=[1, 5, 10]):
    """
    Evaluate MRR, Recall, and Hit Rate at cutoffs.
    """
    metrics = {}

    # MRR calculation
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall calculation
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall



    # NDCG calculation
    ndcgs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            dcg = 0
            idcg = 0
            label_set = set(label)
            rel = [1 if pred[i] in label_set else 0 for i in range(cutoff)]
            for i, r in enumerate(rel):
                dcg += (2**r - 1) / np.log2(i + 2)  # i+2 because index is zero-based and log base is 2
            sorted_rel = sorted(rel, reverse=True)
            for i, r in enumerate(sorted_rel):
                idcg += (2**r - 1) / np.log2(i + 2)
            ndcgs[k] += dcg / idcg if idcg > 0 else 0  # avoid division by zero

    ndcgs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        ndcg = ndcgs[i]
        metrics[f"NDCG@{cutoff}"] = ndcg

    
    ndcgs_v2 = np.zeros(len(cutoffs))

    # 遍历每个预测和标签对
    for pred, label in zip(preds, labels):
        label_set = set(label)
        # 计算rel和sorted_rel一次，然后用于所有cutoffs
        rel = [1 if pred[i] in label_set else 0 for i in range(max(cutoffs))]
        sorted_rel = sorted(rel, reverse=True)
        
        # 遍历每个截止点
        for k, cutoff in enumerate(cutoffs):
            dcg_v2 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel[:cutoff]))
            idcg_v2 = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted_rel[:cutoff]))
            # 累加NDCG
            ndcgs_v2[k] += dcg_v2 / idcg_v2 if idcg_v2 > 0 else 0

    # 平均NDCG
    ndcgs_v2 /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        ndcg_v2 = ndcgs_v2[i]
        metrics[f"NDCG_v2@{cutoff}"] = ndcg_v2
    # metrics = {f"NDCG_v2@{cutoff}": ndcg_v2 for cutoff, ndcg_v2 in zip(cutoffs, ndcgs_v2)}


    return metrics


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    eval_data = datasets.load_dataset("json", data_files=args.eval_data, split="train")
    eval_data = eval_data.select(range(1000))
    corpus = datasets.load_dataset("json", data_files=args.corpus, split="train")
    query_list = eval_data["query"]
    print(f"query_list Length: {len(query_list)}")

    model = FlagModel(
        args.encoder,
        query_instruction_for_retrieval=(
            "为这个句子生成表示以用于检索相关文章：" if args.add_instruction else None
        ),
        use_fp16=args.fp16,
    )
    faiss_index = index(
        model=model,
        corpus=corpus,
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding,
    )

    scores, indices = search(
        model=model,
        queries=eval_data,
        faiss_index=faiss_index,
        k=args.k,
        batch_size=args.batch_size,
        max_length=args.max_query_length,
    )

    retrieval_results = []
    print(f"indices Length: {len(indices)}")
    # indices.shape = (2470, 20)
    # len(indice) = 20
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        # corpus[indice]: {'content':str}
        retrieval_results.append(corpus[indice]["content"])
    print(f"retrieval_results Length: {len(retrieval_results)}")

    # sorted_retrieval_results = rerank(
    #     query_list=query_list[0:10], retrieval_results=retrieval_results
    # )

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["positive"])
    # print(ground_truths[0][0:2])
    from FlagEmbedding.llm_embedder.src.utils import save_json

    origin_metrics = evaluate(retrieval_results, ground_truths)

    if args.rerank:

        # bge-reranker-large:
        bge_reranker = FlagReranker(
            model_name_or_path="/data/disk2/zhangzh/models/BAAI/bge-reranker-large",
            use_fp16=True,
        )
        logger.info("bge_reranker_large processing >>>>>>>>>>>>>>>>")
        reranker_rerank_results = rerankv2(
            reranker=bge_reranker, query_list=query_list, retrieval_results=retrieval_results,flag="bge-large"
        )
        del bge_reranker

        # bge-reranker-v2-m3:
        m3_reranker = FlagReranker('/data/disk2/zhangzh/models/BAAI/bge-reranker-v2-m3', use_fp16=True)
        logger.info("m3_reranker processing >>>>>>>>>>>>>>>>")
        m3_rerank_results = rerankv2(
            reranker=m3_reranker, query_list=query_list, retrieval_results=retrieval_results,flag="m3"
        )
        del m3_reranker


        # bge-reranker-v2-gemma:
        gemma_reranker = FlagLLMReranker('/data/disk2/zhangzh/models/BAAI/bge-reranker-v2-gemma', use_fp16=True)
        logger.info("gemma_reranker processing >>>>>>>>>>>>>>>>")
        gemma_rerank_results = rerankv2(
            reranker=gemma_reranker, query_list=query_list, retrieval_results=retrieval_results,flag="gemma"
        )
        del gemma_reranker

        # bge-reranker-v2-minicpm:
        minicpm_reranker = LayerWiseFlagLLMReranker('/data/disk2/zhangzh/models/BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)
        logger.info("minicpm_reranker processing >>>>>>>>>>>>>>>>")
        minicpm_rerank_results = rerankv2(
            reranker=minicpm_reranker, query_list=query_list, retrieval_results=retrieval_results, flag="minicpm"
        )
        del minicpm_reranker

        bge_reranker_metrics = evaluate(reranker_rerank_results, ground_truths)
        m3_reranker_metrics = evaluate(m3_rerank_results, ground_truths)
        gemma_reranker_metrics = evaluate(gemma_rerank_results, ground_truths)
        minicpm_reranker_metrics = evaluate(minicpm_rerank_results, ground_truths)




    print(f"----------Origin Result:----------------------")
    pprint(origin_metrics, sort_dicts=False)
    print()

    if args.rerank:
        print(f"----------BGE Reranker Result:-")
        pprint(bge_reranker_metrics, sort_dicts=False)
        print()

        print(f"----------M3 Reranker Result:----------")
        pprint(m3_reranker_metrics, sort_dicts=False)
        print()

        print(f"----------GEMMA Reranker Result:----------")
        pprint(gemma_reranker_metrics, sort_dicts=False)
        print()

        print(f"----------MiniCPM Reranker Result:----------")
        pprint(minicpm_reranker_metrics, sort_dicts=False)
        print()

    # metrics2 = evaluate(sorted_retrieval_results, ground_truths)
    # print("-------------Result w/ reranker-----------------")
    # pprint(metrics2, sort_dicts=False)
    # print("------------------------------------------------")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    # 计算运行时间
    run_time = round(end_time - start_time, 3)
    # 打印运行时间
    print("Total cost: [", run_time, "] seconds")
    print("------------------------------------------------")
    print()
