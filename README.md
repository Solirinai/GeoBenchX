# GeoBenchX
LLM-agents benchmark set of geospatial tasks requiring multi-step tool use; and LLM-as-Judge based evaluation framework.


In this work, we establish a benchmark for evaluating large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess seven leading commercial LLMs (Sonnet 3.5 and 3.7, Haiku 3.5, Gemini 2.0, GPT-4o, GPT-4o mini, and o3-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks across four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test hallucination rejection. We develop an LLM-as-Judge evaluation framework to compare agent solutions against reference implementations. Results show Sonnet 3.5 and GPT-4o achieve the best overall performance, with Claude models excelling on solvable tasks while OpenAI models better identify unsolvable scenarios. We observe significant differences in token usage, with Anthropic models consuming substantially more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. 
The resulting benchmark set, evaluation framework, and data generation pipeline will be released as open-source resources, providing one more standardized method for ongoing evaluation of LLMs for GeoAI.

Paper on Arxiv.org [GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks](https://arxiv.org/abs/2503.18129)
## Citation

If you use this benchmark or code in your research, please cite:

BibTeX:
```bibtex
@misc{krechetova2025geobenchxbenchmarkingllmsmultistep,
      title={GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks}, 
      author={Varvara Krechetova and Denis Kochedykov},
      year={2025},
      eprint={2503.18129},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.18129}, 
}