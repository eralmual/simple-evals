import os
import json
import torch
import argparse

import pandas as pd

from common import make_report
from mmlu_eval import MMLUEval

from samplers import ModelCompletionSampler, PipelineCompletionSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("xpu" if torch.xpu.is_available() else device)

def main(results_path: str, debug: bool, num_threads: int, sample_size: int):
    print(f"Using device: {device}")
    samplers = {
        "gemma-2-2b-it-bf16": ModelCompletionSampler(
            model_id="google/gemma-2-2b-it",
            max_tokens=1024,
            device=device,
            d_type=torch.bfloat16,
        ),
        "gemma-2-2b-it-4bit": ModelCompletionSampler(
            model_id="google/gemma-2-2b-it",
            max_tokens=1024,
            device=device,
            d_type=4,
        ),
        "Llama-3.2-3B-bf16": PipelineCompletionSampler(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=1024,
            d_type=torch.bfloat16,
            device=device,
        ),
        "Llama-3.2-3B-4bit": PipelineCompletionSampler(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            max_tokens=1024,
            d_type=4,
            device=device,
        ),
        "Phi-3.5-mini-instruct-bf16": PipelineCompletionSampler(
            model_id="microsoft/Phi-3.5-mini-instruct",
            max_tokens=1024,
            d_type=torch.bfloat16,
            device=device,
        ),
        "Phi-3.5-mini-instruct-4bit": PipelineCompletionSampler(
            model_id="microsoft/Phi-3.5-mini-instruct",
            max_tokens=1024,
            d_type=4,
            device=device,
        ),
    }

    def get_evals(eval_name):
        match eval_name:
            case "mmlu_EN-US":
                return MMLUEval(num_examples=10 if debug else sample_size, language="EN-US")
            case "mmlu_AR-XY":
                return MMLUEval(num_examples=10 if debug else sample_size, language="AR-XY")
            case "mmlu_BN-BD":
                return MMLUEval(num_examples=10 if debug else sample_size, language="BN-BD")
            case "mmlu_DE-DE":
                return MMLUEval(num_examples=10 if debug else sample_size, language="DE-DE")
            case "mmlu_ES-LA":
                return MMLUEval(num_examples=10 if debug else sample_size, language="ES-LA")
            case "mmlu_FR-FR":
                return MMLUEval(num_examples=10 if debug else sample_size, language="FR-FR")
            case "mmlu_HI-IN":
                return MMLUEval(num_examples=10 if debug else sample_size, language="HI-IN")
            case "mmlu_ID-ID":
                return MMLUEval(num_examples=10 if debug else sample_size, language="ID-ID")
            case "mmlu_IT-IT":
                return MMLUEval(num_examples=10 if debug else sample_size, language="IT-IT")
            case "mmlu_JA-JP":
                return MMLUEval(num_examples=10 if debug else sample_size, language="JA-JP")
            case "mmlu_KO-KR":
                return MMLUEval(num_examples=10 if debug else sample_size, language="KO-KR")
            case "mmlu_PT-BR":
                return MMLUEval(num_examples=10 if debug else sample_size, language="PT-BR")
            case "mmlu_ZH-CN":
                return MMLUEval(num_examples=10 if debug else sample_size, language="ZH-CN")
            case "mmlu_SW-KE":
                return MMLUEval(num_examples=10 if debug else sample_size, language="SW-KE")
            case "mmlu_YO-NG":
                return MMLUEval(num_examples=10 if debug else sample_size, language="YO-NG")
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    evals = [
        #"mmlu_AR-XY",
        #"mmlu_BN-BD",
        #"mmlu_DE-DE",
        "mmlu_EN-US",
        "mmlu_ES-LA",
        #"mmlu_FR-FR",
        #"mmlu_HI-IN",
        "mmlu_ID-ID",
        #"mmlu_IT-IT",
        #"mmlu_JA-JP",
        #"mmlu_KO-KR",
        #"mmlu_PT-BR",
        "mmlu_ZH-CN",
        #"mmlu_SW-KE",
        #"mmlu_YO-NG",
    ]
    
    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}
    for sampler_name, sampler in samplers.items():
        for eval_name in evals:
            eval_obj = get_evals(eval_name)
            result = eval_obj(sampler, debug=debug, num_threads=num_threads)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = os.path.join(results_path, f"{file_stem}{debug_suffix}.html")
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w", encoding="utf-8") as fh:
                fh.write(make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = os.path.join(results_path, f"{file_stem}{debug_suffix}.json")
            with open(result_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMMLU for a given set of models.")
    
    # Define the arguments
    parser.add_argument('-r', '--report', type=str, required=True, help="Path to the reports folder")
    parser.add_argument('-nt', '--num_threads', type=int, default=16, help="Number of threads")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('-sz', '--sample_size', type=int, default=None, help="Number of entries to sample")
    args = parser.parse_args()

    main(args.report, args.debug, args.num_threads, args.sample_size)

#"C:\\Users\\erick\\gdrive\\My Drive\\Maestría\\NLP\\Proyecto\\reports"