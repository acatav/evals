import os
from pydantic import BaseModel

from evals.cli.oaieval import main


class CanopyArguments(BaseModel):
    top_k: int
    reranker: str
    top_n: int


args = [CanopyArguments(top_k=100 if reranker == 'cohere' else top_k,
                        reranker=reranker,
                        top_n=top_k if reranker == 'cohere' else -1)
        for top_k in range(1, 11)
        for reranker in ['cohere', "none"]]


if __name__ == '__main__':
    out_dir = "results_nq"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    index_name = "amnon-nq-evals-demo"

    for arg in args:
        out_path = f"{out_dir}/top_k={arg.top_k}_reranker={arg.reranker}_top_n={arg.top_n}.jsonl"
        if not os.path.isfile(out_path):
            main(["canopy",
              "nq-fact",
              "--completion_args", f"top_k={arg.top_k},index_name={index_name},reranker={arg.reranker},top_n={arg.top_n}",
              "--record_path", out_path])
